from typing import Optional, Any
from omegaconf import DictConfig
import time
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
import wandb
from PIL import Image

from .df_base import DiffusionForcingBase
from utils.logging_utils import (
    make_trajectory_images,
    get_random_start_goal,
    make_convergence_animation,
    make_mpc_animation
)
from .tree_node import TreeNode

OGBENCH_ENVS = [
    "pointmaze-medium-v0",
    "pointmaze-large-v0",
    "pointmaze-giant-v0",
    "pointmaze-teleport-v0",
    "antmaze-medium-v0",
    "antmaze-large-v0",
    "antmaze-giant-v0",
    "antmaze-teleport-v0",
    "cube-single-play-v0",
    "cube-double-play-v0",
    "cube-triple-play-v0",
    "cube-quadruple-play-v0",
]


class DiffusionForcingPlanning(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        self.env_id = cfg.env_id
        self.dataset = cfg.dataset
        self.action_dim = len(cfg.action_mean)
        self.observation_dim = len(cfg.observation_mean)
        self.use_reward = cfg.use_reward
        self.unstacked_dim = self.observation_dim + self.action_dim + int(self.use_reward)
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len
        self.n_tokens = self.episode_len // cfg.frame_stack + 1
        self.gamma = cfg.gamma
        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.observation_mean = np.array(cfg.observation_mean[: self.observation_dim])
        self.observation_std = np.array(cfg.observation_std[: self.observation_dim])
        self.action_mean = np.array(cfg.action_mean[: self.action_dim])
        self.action_std = np.array(cfg.action_std[: self.action_dim])
        self.open_loop_horizon = cfg.open_loop_horizon
        self.padding_mode = cfg.padding_mode
        self.interaction_seed = cfg.interaction_seed
        self.use_random_goals_for_interaction = cfg.use_random_goals_for_interaction
        self.task_id = cfg.task_id
        self.dql_model = cfg.dql_model
        self.val_max_steps = cfg.val_max_steps
        self.mctd = cfg.mctd
        self.mctd_guidance_scales = cfg.mctd_guidance_scales
        self.mctd_max_search_num = cfg.mctd_max_search_num
        self.mctd_num_denoising_steps = cfg.mctd_num_denoising_steps
        self.mctd_skip_level_steps = cfg.mctd_skip_level_steps
        self.jump = cfg.jump
        self.time_limit = cfg.time_limit
        self.parallel_search_num = cfg.parallel_search_num
        self.virtual_visit_weight = cfg.virtual_visit_weight
        self.warp_threshold = cfg.warp_threshold
        self.leaf_parallelization = cfg.leaf_parallelization
        self.parallel_multiple_visits = cfg.parallel_multiple_visits
        self.early_stopping_condition = cfg.early_stopping_condition
        self.num_tries_for_bad_plans = cfg.num_tries_for_bad_plans
        self.sub_goal_interval = cfg.sub_goal_interval
        self.sub_goal_threshold = cfg.sub_goal_threshold
        self.viz_plans = cfg.viz_plans
        self.cube_single_dql = cfg.cube_single_dql
        self.cube_viz = cfg.cube_viz
        super().__init__(cfg)
        self.plot_end_points = cfg.plot_start_goal and self.guidance_scale != 0

    def _build_model(self):
        mean = list(self.observation_mean) + list(self.action_mean)
        std = list(self.observation_std) + list(self.action_std)
        if self.use_reward:
            mean += [self.reward_mean]
            std += [self.reward_std]
        self.cfg.data_mean = np.array(mean).tolist()
        self.cfg.data_std = np.array(std).tolist()
        super()._build_model()

    def _preprocess_batch(self, batch):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]

        observations = observations[..., : self.observation_dim]
        actions = actions[..., : self.action_dim]

        if (n_frames - 1) % self.frame_stack != 0:
            raise ValueError("Number of frames - 1 must be divisible by frame stack size")

        nonterminals = torch.cat([torch.ones_like(nonterminals[:, : self.frame_stack]), nonterminals[:, :-1]], dim=1)
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        # masks = torch.cat([masks[:-self.frame_stack:self.jump], masks[-self.frame_stack:]], dim=0)

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1)
        bundles = self._normalize_x(self.make_bundle(observations, actions, rewards))  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = 0  # zero out actions and rewards after normalization
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b t c)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = None
        # bundles = bundles[::self.jump]
        return bundles, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

        n_tokens, batch_size = xs.shape[:2]

        weights = masks.float()
        if not self.causal:
            # manually mask out entries to train for varying length
            random_terminal = torch.randint(2, n_tokens + 1, (batch_size,), device=self.device)
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[:, :n_tokens].bool()
            random_terminal = repeat(random_terminal, "b t -> (t fs) b", fs=self.frame_stack)
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs, masks=masks))

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            o, a, r = self.split_bundle(xs_pred)
            trajectory = o.detach().cpu().numpy()[:-1, :8]  # last observation is dummy, sample 8
            images = make_trajectory_images(self.env_id, trajectory, trajectory.shape[1], None, None, False)
            for i, img in enumerate(images):
                self.log_image(
                    f"training_visualization/sample_{i}",
                    Image.fromarray(img),
                )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        if self.guidance_scale == 0:
            namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        self.interact(batch_size, conditions, namespace)  # interact if environment is installation

    def plan(self, start: torch.Tensor, goal: torch.Tensor, horizon: int, conditions: Optional[Any] = None,
        guidance_scale: int = None, noise_level: Optional[torch.Tensor] = None, plan: Optional[torch.Tensor] = None):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan

        batch_size = start.shape[0]

        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                dist = nn.functional.mse_loss(pred, target, reduction="none")  # (t fs) b c
                # guidance weight for observation and action
                weight = np.array(
                    [20] * (self.frame_stack)  # conditoning (aka reconstruction guidance)

                    + [1 for _ in range(horizon)]  # try to reach the goal at any horizon
                    #+ [0 for _ in range(horizon-1)] + [1]  # Diffuer guidance
                    + [0] * (h_padded - horizon)  # don't guide padded entries due to horizon % frame_stack != 0
                )
                # mathematically, one may also try multiplying weight by sqrt(alpha_cum)
                # this means you put higher weight to less noisy terms
                # which might be better but we haven't tried yet
                weight = torch.from_numpy(weight).float().to(self.device)
                
                dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                dist_o = dist_o[:, :, : 2]
                #dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=self.observation_dim // 2).sqrt()
                dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=1).sqrt()
                dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                #dist = torch.cat([dist_o, dist_a], -1)
                dist = dist_o
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]

                episode_return = -(dist * weight).mean() * 1000 * dist.shape[1] / 16 # considering the batch size
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")
                rewards = pred[:, :, -1]
                weight = np.array([10] * self.frame_stack + [0.997**j for j in range(h)] + [0] * h_padded)
                weight = torch.from_numpy(weight).float().to(self.device)
                episode_return = rewards * weight[:, None]

            #return self.guidance_scale * episode_return
            return guidance_scale * episode_return

        #guidance_fn = goal_guidance if self.guidance_scale else None
        guidance_fn = goal_guidance if guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        #pad_tokens = 0 # To be more efficient
        if noise_level is None:
            scheduling_matrix = self._generate_scheduling_matrix(plan_tokens)
        else: # if noise_level is given, use it
            scheduling_matrix = noise_level
        if plan is None:
            chunk = torch.randn((plan_tokens, batch_size, *self.x_stacked_shape), device=self.device)
            chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        else: # if plan is given, use it
            chunk = plan
            chunk = rearrange(chunk, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)
        pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0)

        plan_hist = [plan.detach().clone()[: self.n_tokens - pad_tokens]]
        stabilization = 0
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            to_noise_levels = np.concatenate(
                [
                    np.array((stabilization,), dtype=np.int64),
                    scheduling_matrix[m + 1],
                    np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
                ]
            )
            from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
            from_noise_levels = repeat(from_noise_levels, "t -> t b", b=batch_size)
            to_noise_levels = repeat(to_noise_levels, "t -> t b", b=batch_size)
            plan[1 : self.n_tokens - pad_tokens] = self.diffusion_model.sample_step(
                plan, conditions, from_noise_levels, to_noise_levels, guidance_fn=guidance_fn
            )[1 : self.n_tokens - pad_tokens]
            plan_hist.append(plan.detach().clone()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack)
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]

        return plan_hist

    def parallel_plan(self, start: torch.Tensor, goal: torch.Tensor, horizon: int, conditions: Optional[Any] = None,
        guidance_scale: int = None, noise_level: Optional[torch.Tensor] = None, plan: Optional[torch.Tensor] = None):
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan
        # assert batch_size == 1, "parallel planning only supports batch size 1"

        batch_size = len(plan)
        start = torch.cat([start] * batch_size, 0)
        goal = torch.cat([goal] * batch_size, 0)            

        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        else:
            if "cube" in self.env_id:
                # object-wise guidance
                cube_indices = []
                new_guidance_scale = []
                for b in range(batch_size):
                    _cube_idx, _guidance_scale = guidance_scale[b].split("-")
                    cube_indices.append(int(_cube_idx))
                    new_guidance_scale.append(float(_guidance_scale))
                guidance_scale = torch.tensor(new_guidance_scale).to(self.device)
            else:
                guidance_scale = torch.tensor(guidance_scale).to(start.device) # (batch_size,)

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                dist = nn.functional.mse_loss(pred, target, reduction="none")  # (t fs) b c

                if "cube" in self.env_id:
                    for b in range(batch_size):
                        _cube_idx = cube_indices[b]
                        for i in range(dist.shape[-1]//3):
                            if i != _cube_idx-1:
                                dist[:, b, i*3:(i+1)*3] = 0

                # guidance weight for observation and action
                weight = np.array(
                    [20] * (self.frame_stack)  # conditoning (aka reconstruction guidance)

                    + [1 for _ in range(horizon)]  # try to reach the goal at any horizon
                    #+ [0 for _ in range(horizon-1)] + [1]  # Diffuer guidance
                    + [0] * (h_padded - horizon)  # don't guide padded entries due to horizon % frame_stack != 0
                )
                weight = torch.from_numpy(weight).float().to(self.device)
                
                dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                #dist_o = dist_o[:, :, : 2]
                dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=1).sqrt()
                dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = dist_o
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]
                #episode_return = -(dist * weight).mean(dim=(0, 2)) * 1000 * dist.shape[1] / 16
                episode_return = -(dist * weight).mean() * 1000 * dist.shape[1] / 16
            else:
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")

            return (guidance_scale * episode_return).mean()

        guidance_fn = goal_guidance if guidance_scale is not None else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        try:
            scheduling_matrix = noise_level
        except:
            raise ValueError("noise_level is required for parallel planning")
        # if None in plan: 
        #     chunk = torch.randn((plan_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        #     chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise) 
        # else:
        #     chunk = torch.stack(plan).squeeze(dim=2)
        #     chunk = rearrange(chunk, "b (t fs) c -> t b (fs c)", fs=self.frame_stack) # 5, 500, 2 =>  50, 5, 20

        chunk = []
        for i in range(batch_size):
            if plan[i] == None:
                c = torch.randn((plan_tokens, 1, *self.x_stacked_shape), device=self.device)
                c = torch.clamp(c, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
            else:
                c = rearrange(plan[i], "(t fs) 1 c -> t 1 (fs c)", fs=self.frame_stack)
            chunk.append(c)
        chunk = torch.cat(chunk, 1)
        if len(chunk.shape) == 2:
            chunk = chunk.unsqueeze(0)
        pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0)

        plan_hist = [plan.detach().clone()[: self.n_tokens - pad_tokens]]
        stabilization = 0

        for m in range(scheduling_matrix.shape[1] - 1):
            from_noise_levels = np.concatenate(
                [
                    np.full((batch_size, 1), stabilization, dtype=np.int64),  # Shape (batch_size, 1)
                    scheduling_matrix[:,m],
                    np.full((batch_size, pad_tokens), self.sampling_timesteps, dtype=np.int64),  # Shape (batch_size, pad_tokens)
                ]
                , axis=1
            )
            to_noise_levels = np.concatenate(
                [
                    np.full((batch_size, 1), stabilization, dtype=np.int64),  # Shape (batch_size, 1)
                    scheduling_matrix[:,m+1],
                    np.full((batch_size, pad_tokens), self.sampling_timesteps, dtype=np.int64),  # Shape (batch_size, pad_tokens)
                ]
                , axis=1
            )
            from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
            from_noise_levels = rearrange(from_noise_levels, "b t -> t b", b=batch_size)
            to_noise_levels = rearrange(to_noise_levels, "b t -> t b", b=batch_size)
            plan[1 : self.n_tokens - pad_tokens] = self.diffusion_model.sample_step(
                plan, conditions, from_noise_levels, to_noise_levels, guidance_fn=guidance_fn
            )[1 : self.n_tokens - pad_tokens]
            plan_hist.append(plan.detach().clone()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack)
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]
        return plan_hist

    def interact(self, batch_size: int, conditions=None, namespace="validation"):

        assert batch_size == 1, f"Batch size must be 1 for Cube, got {batch_size}"

        try:
            import gym
            import ogbench
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print("d4rl import not successful, skipping environment interaction. Check d4rl installation.")
            return

        print("Interacting with environment... This may take a couple minutes.")

        use_diffused_action = False

        if self.env_id in OGBENCH_ENVS:
            if "pointmaze" in self.env_id:
                env = ogbench.make_env_and_datasets(self.dataset)[0]
                if self.action_dim == 2:
                    use_diffused_action = True
            elif "antmaze" in self.env_id:
                env = ogbench.make_env_and_datasets(self.dataset)[0]
                #use_diffused_action = True
                from dql.main_Antmaze import hyperparameters
                from dql.agents.ql_diffusion import Diffusion_QL as Agent
                params = hyperparameters[self.dataset]
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                max_action = float(env.action_space.high[0])
                agent = Agent(
                    env_name=self.env_id,
                    state_dim=state_dim*2,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=0,
                    discount=0.99,
                    tau=0.005,
                    max_q_backup=params["max_q_backup"],
                    beta_schedule="vp",
                    n_timesteps=5,
                    eta=params["eta"],
                    lr=params["lr"],
                    lr_decay=False,
                    lr_maxt=params["num_epochs"],
                    grad_norm=params["gn"],
                    goal_dim=2,
                    lcb_coef=4.0,
                )
                # pretrained agent loading
                if self.dataset == "antmaze-medium-navigate-v0":
                    dql_folder = "antmaze-medium-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                elif self.dataset == "antmaze-large-navigate-v0":
                    dql_folder = "antmaze-large-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                elif self.dataset == "antmaze-giant-navigate-v0":
                    dql_folder = "antmaze-giant-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                import os
                agent.load_model(os.path.join(os.getcwd(), "dql", "results", dql_folder), id=200)

            elif "cube" in self.env_id:
                assert batch_size == 1, f"Batch size must be 1 for Cube, got {batch_size}"
                env = ogbench.make_env_and_datasets(self.dataset)[0]
                from dql.main_Cube import hyperparameters
                from dql.agents.ql_diffusion import Diffusion_QL as Agent
                params = hyperparameters[self.dataset]
                state_dim = env.observation_space.shape[0] if not self.cube_single_dql else 28
                action_dim = env.action_space.shape[0]
                max_action = float(env.action_space.high[0])
                agent = Agent(
                    env_name=self.env_id,
                    state_dim=state_dim*2,
                    action_dim=action_dim,
                    max_action=max_action,
                    device=0,
                    discount=0.99,
                    tau=0.005,
                    max_q_backup=params["max_q_backup"],
                    beta_schedule="vp",
                    n_timesteps=5,
                    eta=params["eta"],
                    lr=params["lr"],
                    lr_decay=False,
                    lr_maxt=params["num_epochs"],
                    grad_norm=params["gn"],
                    goal_dim=params["goal_dim"] if self.cube_single_dql else self.observation_dim,
                    lcb_coef=4.0,
                )
                # pretrained agent loading
                import os
                if self.dataset == "cube-single-play-v0" or self.cube_single_dql:
                    dql_folder = "cube-single-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|3|1.0|False|cql_antmaze|0.2|4.0|10"
                    agent.load_model(os.path.join(os.getcwd(), "dql", "results", dql_folder), id=200)
                else:
                    if self.dataset == "cube-double-play-v0":
                        dql_folder = "cube-double-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|6|1.0|False|cql_antmaze|0.2|4.0|10"
                        agent.load_model(os.path.join(os.getcwd(), "dql", "results", dql_folder), id=2000)
                    elif self.dataset == "cube-triple-play-v0":
                        dql_folder = "cube-triple-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|9|1.0|False|cql_antmaze|0.2|4.0|10"
                        agent.load_model(os.path.join(os.getcwd(), "dql", "results", dql_folder), id=2000)
                    elif self.dataset == "cube-quadruple-play-v0":
                        dql_folder = "cube-quadruple-play-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|12|1.0|False|cql_antmaze|0.2|4.0|10"
                        agent.load_model(os.path.join(os.getcwd(), "dql", "results", dql_folder), id=2000)
        else:
            raise NotImplementedError(f"Environment interaction not implemented for this environment {self.env_id}")

        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        obs, info = env.reset(options=dict(task_id=self.task_id))
        if "cube" in self.env_id:
            num_cubes = (obs.shape[0] - 19) // 9
        goal = info['goal']

        obs = obs.reshape(1, -1)
        goal = np.array(goal).reshape(1, -1)

        origin_obs_numpy = obs.copy()
        obs = torch.from_numpy(obs).float().to(self.device)
        if "cube" in self.env_id:
            start = torch.cat([obs[:,19+i*9:19+i*9+3] for i in range(num_cubes)], dim=-1).detach()
            obs_normalized = ((torch.cat([obs[:,19+i*9:19+i*9+3] for i in range(num_cubes)], dim=-1) - obs_mean[None])/obs_std[None]).detach()
        else:
            start = obs[:, : self.observation_dim]
            obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

        goal = torch.Tensor(goal).float().to(self.device)
        if "cube" in self.env_id:
            goal = torch.cat([goal[:,19+i*9:19+i*9+3] for i in range(num_cubes)], dim=-1).detach()
        else:
            goal = goal[:, : self.observation_dim]
        goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach() 

        terminate = False
        steps = 0
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        reached = np.zeros(batch_size, dtype=bool)
        first_reach = np.zeros(batch_size)

        trajectory = []  # actual trajectory
        all_plan_hist = []  # a list of plan histories, each history is a collection of m diffusion steps

        # Cube task visualization
        if "cube" in self.env_id and self.cube_viz:
            cube_visual_obss = [env.render()]

        succeed_cube_list = []
        target_cube_idx = -1

        planning_time = []
        while not terminate and steps < self.val_max_steps:
            # list of the cubes achieved
            if "cube" in self.env_id:
                start_numpy = start.cpu().numpy()[:, :self.observation_dim]
                goal_numpy = goal.cpu().numpy()[:, :self.observation_dim]   
                for i in range(start_numpy.shape[1]//3):
                    if np.linalg.norm(start_numpy[0, i*3:i*3+3] - goal_numpy[0, i*3:i*3+3]) < 0.15:
                        succeed_cube_list.append(i)
            else:
                succeed_cube_list = None
            print("#########################")
            print(f"Plan at step {steps}")
            print("#########################")
            planning_start_time = time.time()
            if self.mctd:
                plan_hist = self.p_mctd_plan(obs_normalized, goal_normalized, self.episode_len, conditions, start.cpu().numpy()[:, :self.observation_dim], goal.cpu().numpy()[:, :self.observation_dim], steps, succeed_cube_list, target_cube_idx) # fake plan_hist
                plan_hist = self._unnormalize_x(plan_hist)
                plan = plan_hist[-1] # (t b c)
            else:
                plan_hist = self.plan(obs_normalized, goal_normalized, self.episode_len, conditions)
                plan_hist = self._unnormalize_x(plan_hist)  # (m t b c)
                plan = plan_hist[-1]  # (t b c)

            # In Cube case, cropping the plan to the successful trajectory
            if "cube" in self.env_id:
                successes = [False] * num_cubes
                plan_numpy = plan.detach().cpu().numpy()[:, :, :self.observation_dim]
                goal_numpy = goal.cpu().numpy()[:, : self.observation_dim]
                for t in range(plan.shape[0]):
                    for i in range(num_cubes):
                        if np.linalg.norm(plan_numpy[t, :, i*3:i*3+3] - goal_numpy[:, i*3:i*3+3], axis=-1) < 0.4:
                            successes[i] = True
                    if all(successes):
                        break
                plan = plan[:t+1, :, :]

            # Getting the target_cube_idx for Object-wise plan
            if "cube" in self.env_id and self.cube_single_dql:
                target_cube_idx = -1
                cube_stds = []
                for i in range(plan.shape[-1]//3):
                    if i in succeed_cube_list:
                        cube_stds.append(0)
                    else:
                        cube_stds.append(torch.mean(torch.std(plan[:, :, 3*i:3*(i+1)], dim=0)).item())
                target_cube_idx = np.argmax(cube_stds)
                print(f"Target cube idx: {target_cube_idx}")

            # Visualization
            start_numpy = start.cpu().numpy()[:, : self.observation_dim]
            goal_numpy = goal.cpu().numpy()[:, : self.observation_dim]
            image = make_trajectory_images(self.env_id, plan[:, :, :self.observation_dim].detach().cpu().numpy(), 1, start_numpy, goal_numpy, self.plot_end_points)[0]
            self.log_image(f"plan/plan_at_{steps}", Image.fromarray(image))

            planning_end_time = time.time()
            planning_time.append(planning_end_time - planning_start_time)

            # jumpy case (fill the gap)
            if self.jump > 1:
                _plan = []
                for t in range(plan.shape[0]):
                    for j in range(self.jump):
                        _plan.append(plan[t, : , :self.observation_dim])
                plan = torch.stack(_plan)

            all_plan_hist.append(plan_hist.cpu())

            if self.cube_single_dql:
                for release_idx in range(30): # Release cube
                    # move to the next cube position
                    obs_numpy = np.concatenate([origin_obs_numpy[:,:19]] + [origin_obs_numpy[:,19+target_cube_idx*9:19+target_cube_idx*9+9]], axis=-1)
                    subgoal = plan[0, :, 3*target_cube_idx:3*target_cube_idx+3].detach().cpu()
                    action = agent.sample_action(obs_numpy, subgoal)
                    action[2] = 0.02
                    action = torch.from_numpy(action).float().reshape(1, -1)
                    origin_obs_numpy, reward, terminated, truncated, info = env.step(np.nan_to_num(action.numpy())[0])
                    if self.cube_viz:
                        cube_visual_obss.append(env.render())

                    reached = np.logical_or(reached, reward >= 1.0)
                    episode_reward += reward
                    episode_reward_if_stay += np.where(~reached, reward, 1)
                    first_reach += ~reached
                    done = terminated or truncated
                    if done:
                        print(f"Terminated at step {t}")
                        terminate = True
                        break

                    obs, reward, done = [torch.from_numpy(item).float() for item in [origin_obs_numpy, np.array(reward), np.array(done)]]
                    obs = obs[None]
                    bundle = self.make_bundle(obs, action, reward[..., None])
                    trajectory.append(bundle)
                    obs = obs.to(self.device)
                    obs = torch.cat([obs[:,19+i*9:19+i*9+3] for i in range(num_cubes)], dim=-1)
                    start = obs
                    obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()
                    origin_obs_numpy = origin_obs_numpy[None]

            if "antmaze" in self.env_id or "cube" in self.env_id:
                if plan.shape[0] > self.sub_goal_interval:
                    sub_goal = plan[self.sub_goal_interval, :, :self.observation_dim].detach().cpu().numpy()
                    sub_goal_step = self.sub_goal_interval
                else:
                    sub_goal = plan[-1, :, :self.observation_dim].detach().cpu().numpy()
                    sub_goal_step = plan.shape[0]

            for t in range(self.open_loop_horizon):
                if terminate:
                    break
                if use_diffused_action:
                    _, action, _ = self.split_bundle(plan[t])
                else:
                    if "antmaze" in self.env_id or "cube" in self.env_id:
                        obs_numpy = obs.detach().cpu().numpy()
                        if np.linalg.norm(obs_numpy[0, :self.observation_dim] - sub_goal[0, :self.observation_dim]) < 1.0:
                            print(f"sub_goal_step {sub_goal_step} achieved at step {t}")
                            sub_goal_step += self.sub_goal_interval
                            if plan.shape[0] - sub_goal_step <= 0:
                                sub_goal = plan[-1, :, :self.observation_dim].detach().cpu().numpy()
                            else:
                                sub_goal = plan[sub_goal_step, :, :self.observation_dim].detach().cpu().numpy()
                        assert obs_numpy.shape[0] == 1, f"Batch size must be 1 for AntMaze, got {obs_numpy.shape[0]}"
                        if "cube" in self.env_id and self.cube_single_dql:
                            _sub_goal = sub_goal[:, 3*target_cube_idx:3*target_cube_idx+3]
                            _obs_numpy = np.concatenate([origin_obs_numpy[:,:19]] + [origin_obs_numpy[:,19+target_cube_idx*9:19+target_cube_idx*9+9]], axis=-1)
                            action = agent.sample_action(_obs_numpy, _sub_goal)
                        else:
                            action = agent.sample_action(origin_obs_numpy, sub_goal)
                        action = torch.from_numpy(action).float().reshape(1, -1)
                    else:
                        if t == 0:
                            plan_vel = plan[t, :, :self.observation_dim] - obs[:, :self.observation_dim]
                        else:
                            if t < plan.shape[0]:
                                plan_vel = plan[t, :, :self.observation_dim] - plan[t - 1, :, :self.observation_dim]
                            else:
                                plan_vel = 0
                        if t < plan.shape[0]:
                            action = 12.5 * (plan[t, :, :self.observation_dim] - obs[:, :self.observation_dim]) + 1.2 * (plan_vel - obs[:, self.observation_dim:])
                        else:
                            action = 12.5 * (plan[-1, :, :self.observation_dim] - obs[:, :self.observation_dim]) + 1.2 * (plan_vel - obs[:, self.observation_dim:])
                action = torch.clip(action, -1, 1).detach().cpu()
                origin_obs_numpy, reward, terminated, truncated, info = env.step(np.nan_to_num(action.numpy())[0])
                if "cube" in self.env_id and self.cube_viz:
                    cube_visual_obss.append(env.render())

                reached = np.logical_or(reached, reward >= 1.0)
                episode_reward += reward
                episode_reward_if_stay += np.where(~reached, reward, 1)
                first_reach += ~reached

                done = terminated or  truncated
                if done:
                    terminate = True
                    break

                obs, reward, done = [torch.from_numpy(item).float() for item in [origin_obs_numpy, np.array(reward), np.array(done)]]
                obs = obs[None]
                bundle = self.make_bundle(obs, action, reward[..., None])
                trajectory.append(bundle)
                obs = obs.to(self.device)
                if "cube" in self.env_id:
                    obs = torch.cat([obs[:,19+i*9:19+i*9+3] for i in range(num_cubes)], dim=-1)
                start = obs
                obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()
                origin_obs_numpy = origin_obs_numpy[None]
                steps += 1

                if "cube" in self.env_id and self.cube_single_dql:
                    if np.linalg.norm(origin_obs_numpy[:,19+target_cube_idx*9:19+target_cube_idx*9+3] - goal_numpy[:, 3*target_cube_idx:3*target_cube_idx+3]) < 0.2:
                        print(f"Cube {target_cube_idx} reached at step {steps}")
                        succeed_cube_list.append(target_cube_idx) # update the list when the cube is reached
                        break

            if self.cube_single_dql and target_cube_idx in succeed_cube_list: # when the target cube is reached
                if not terminate:
                    for release_idx in range(30): # Release cube
                        if release_idx < 10:
                            action = np.nan_to_num([0, 0, 1.0, 0, -1.0]).reshape(1,-1) # Open the gripper and move up
                        else:
                            action = np.nan_to_num([1.0, 0, 0, 0, -1.0]).reshape(1,-1) # Move back
                        action = torch.from_numpy(action).float().reshape(1, -1)
                        origin_obs_numpy, reward, terminated, truncated, info = env.step(np.nan_to_num(action.numpy())[0])
                        if self.cube_viz:
                            cube_visual_obss.append(env.render())

                        reached = np.logical_or(reached, reward >= 1.0)
                        episode_reward += reward
                        episode_reward_if_stay += np.where(~reached, reward, 1)
                        first_reach += ~reached
                        done = terminated or truncated
                        #done = terminated
                        if done:
                            print(f"Terminated at step {steps}")
                            terminate = True
                            break

                        #obs, reward, done = [torch.from_numpy(item).float() for item in [obs, reward, done]]
                        #bundle = self.make_bundle(obs, action, reward[..., None])
                        obs, reward, done = [torch.from_numpy(item).float() for item in [origin_obs_numpy, np.array(reward), np.array(done)]]
                        obs = obs[None]
                        bundle = self.make_bundle(obs, action, reward[..., None])
                        trajectory.append(bundle)
                        obs = obs.to(self.device)
                        if "cube" in self.env_id:
                            obs = torch.cat([obs[:,19+i*9:19+i*9+3] for i in range(num_cubes)], dim=-1)
                        start = obs
                        obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()
                        origin_obs_numpy = origin_obs_numpy[None]
                        steps += 1
                target_cube_idx = -1

        self.log(f"{namespace}/planning_time", np.sum(planning_time))
        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())
        self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0) / batch_size)

        # Visualization
        #samples = min(16, batch_size)
        samples = min(32, batch_size)
        trajectory = torch.stack(trajectory)
        if "cube" in self.env_id:
            trajectory = torch.cat([trajectory[:, :, 19+i*9:19+i*9+3] for i in range(num_cubes)], dim=-1)
        start = start[:, :self.observation_dim].cpu().numpy().tolist()
        goal = goal[:, :self.observation_dim].cpu().numpy().tolist()
        images = make_trajectory_images(self.env_id, trajectory, samples, start, goal, self.plot_end_points)

        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_interaction/sample_{i}",
                Image.fromarray(img),
            )

        if "cube" in self.env_id and self.cube_viz:
            # Visual observation
            visual_obss = np.array(cube_visual_obss).transpose(0, 3, 1, 2) # (t, c, h, w)
            self.logger.experiment.log({
                        f"validation_interaction_video/video": wandb.Video(visual_obss, fps=24),
                        f"trainer/global_step": self.global_step,
            })

        if self.debug:
            samples = min(16, batch_size)
            indicies = list(range(samples))

            for i in indicies:
                filename = make_convergence_animation(
                    self.env_id,
                    all_plan_hist,
                    trajectory,
                    start,
                    goal,
                    self.open_loop_horizon,
                    namespace=namespace,
                    batch_idx=i,
                )
                self.logger.experiment.log(
                    {
                        f"convergence/{namespace}_{i}": wandb.Video(filename, fps=4),
                        f"trainer/global_step": self.global_step,
                    }
                )

                filename = make_mpc_animation(
                    self.env_id,
                    all_plan_hist,
                    trajectory,
                    start,
                    goal,
                    self.open_loop_horizon,
                    namespace=namespace,
                    batch_idx=i,
                )
                self.logger.experiment.log(
                    {
                        f"mpc/{namespace}_{i}": wandb.Video(filename, fps=24),
                        f"trainer/global_step": self.global_step,
                    }
                )

    def pad_init(self, x, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            x[: self.frame_stack - 1] = 0
        elif self.padding_mode != "same":
            raise ValueError("init_pad must be 'zero' or 'same'")
        if batch_first:
            x = rearrange(x, "fs b ... -> b fs ...")

        return x

    def split_bundle(self, bundle):
        if self.use_reward:
            return torch.split(bundle, [self.observation_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [self.observation_dim, self.action_dim], -1)
            return o, a, None

    def make_bundle(
        self,
        obs: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if action is not None and valid_value is not None:
            valid_value = action
        if reward is not None and valid_value is not None:
            valid_value = reward
        if valid_value is None:
            raise ValueError("At least one of obs, action, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise_levels = super()._generate_noise_levels(xs, masks)
        _, batch_size, *_ = xs.shape

        # first frame is almost always known, this reflect that
        if random() < 0.5:
            noise_levels[0] = torch.randint(0, self.timesteps // 4, (batch_size,), device=xs.device)

        return noise_levels

    def visualize_node_value_plans(self, steps, search_num, values, names, plans, value_plans, starts, goals):
        if plans.shape[1] != starts.shape[0]:
            starts = starts.repeat(plans.shape[1], axis=0)
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0)
        plans = self._unnormalize_x(plans)
        plan_obs, _, _ = self.split_bundle(plans)
        plan_obs = plan_obs.detach().cpu().numpy()[:-1]
        plan_images = make_trajectory_images(self.env_id, plan_obs, plan_obs.shape[1], starts, goals, self.plot_end_points)
        value_plans = self._unnormalize_x(value_plans)
        value_plan_obs, _, _ = self.split_bundle(value_plans)
        value_plan_obs = value_plan_obs.detach().cpu().numpy()[:-1]
        value_plan_images = make_trajectory_images(self.env_id, value_plan_obs, value_plan_obs.shape[1], starts, goals, self.plot_end_points)
        for i in range(len(plan_images)):
            plan_image = plan_images[i]
            value_plan_image = value_plan_images[i]
            img = np.concatenate([plan_image, value_plan_image], axis=0)
            self.log_image(
                f"mcts_plan/{steps}_{search_num+i+1}_{names[i]}_V{values[i]}",
                Image.fromarray(img),
            )

    def calculate_values(self, plans, starts, goals):
        if plans.shape[1] != starts.shape[0]:
            starts = starts.repeat(plans.shape[1], axis=0)
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0)

        plans = self._unnormalize_x(plans)
        obs, _, _ = self.split_bundle(plans)
        obs = obs.detach().cpu().numpy()[:-1, :]  # last observation is dummy
        values = np.zeros(plans.shape[1])
        infos = np.array(["NotReached"] * plans.shape[1])
        achieved_ts = np.array([None] * plans.shape[1])
        for t in range(obs.shape[0]):
            if t == 0:
                pos_diff = np.linalg.norm(obs[t] - starts, axis=-1)
            else:
                pos_diff = np.linalg.norm(obs[t] - obs[t-1], axis=-1)
            infos[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = "Warp"
            values[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = 0
            diff_from_goal = np.linalg.norm(obs[t] - goals, axis=-1)
            values[(diff_from_goal < 2.0) * (infos == "NotReached")] = (plans.shape[0] - t) / plans.shape[0]
            achieved_ts[(diff_from_goal < 2.0) * (infos == "NotReached")] = t
            infos[(diff_from_goal < 2.0) * (infos == "NotReached")] = "Achieved"

        return values, infos, achieved_ts

    def calculate_values_cube(self, plans, starts, goals, obs_normalized, goal_normalized, succeed_cube_list):
        if plans.shape[1] != starts.shape[0]:
            starts = starts.repeat(plans.shape[1], axis=0)
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0)

        normalized_plans = plans.clone()
        plans = self._unnormalize_x(plans)
        obs, _, _ = self.split_bundle(plans)
        obs = obs.detach().cpu().numpy()[:-1, :]  # last observation is dummy
        values = np.zeros(plans.shape[1])
        infos = np.array(["NotReached"] * plans.shape[1])
        achieved_ts = np.array([None] * plans.shape[1])
        succeed_cube_indices = np.array([-1] * plans.shape[1])

        for b in range(obs.shape[1]): # for each sample
            fully_success = False
            partial_success = False
            not_stacked_list = []
            terminal_ts = -1
            for t in range(obs.shape[0]): # for each step
                newly_succeeded = False
                _successes = [False] * (plans.shape[-1]//3)
                if t == 0:
                    pos_diff = np.linalg.norm(obs[t, b, :] - starts[b, :], axis=-1)
                else:
                    pos_diff = np.linalg.norm(obs[t, b, :] - obs[t-1, b, :], axis=-1)
                if pos_diff > self.warp_threshold:
                    infos[b] = "Warp"
                    values[b] = 0
                    break
                for i in range(plans.shape[-1]//3):
                    if i in not_stacked_list:
                        continue
                    if np.linalg.norm(obs[t, b, 3*i:3*(i+1)] - goals[b, 3*i:3*(i+1)], axis=-1) < self.sub_goal_threshold:
                        if i in succeed_cube_list: 
                                _successes[i] = True
                                continue
                        if self.jump > 1: # When jump > 1, the value calculation is not correct due to the large jump
                            obs_backup = obs[t, b, 3*i:3*(i+1)].copy()
                            obs[t, b, 3*i:3*(i+1)] = goals[b, 3*i:3*(i+1)]
                        if i not in succeed_cube_list:
                            # Stacking check
                            check_count = 1
                            is_stacked = True
                            below_cube_z = obs[t, b, 3*i+2].copy()
                            while True:
                                below_cube_z -= 0.5 * check_count
                                if below_cube_z < -0.1:
                                    #if check_count == 1:
                                    #    print("The cube is on the floor: ", i, obs[t, b, 3*i:3*i+3])
                                    break
                                found_cube = False
                                for j in range(plans.shape[-1]//3):
                                    if i == j:
                                        continue
                                    if j in succeed_cube_list:
                                        j_t = 0
                                    else:
                                        j_t = t
                                    if abs(obs[j_t, b, 3*j+2] - below_cube_z) < 0.1 and np.linalg.norm(obs[j_t, b, 3*j:3*j+3] - obs[t, b, 3*i:3*i+3], axis=-1) < 0.6:
                                        found_cube = True
                                        print(f"Found below cube: {i}, {obs[t, b, 3*i:3*i+3]} - {j}, {obs[j_t, b, 3*j:3*j+3]}")
                                        break
                                    else:
                                        print(f"Not Found below cube: {i}, {obs[t, b, 3*i:3*i+3]} compared with {j}, {obs[j_t, b, 3*j:3*j+3]}, z-axis diff: {abs(obs[j_t, b, 3*j+2] - below_cube_z)}, pos-diff: {np.linalg.norm(obs[j_t, b, 3*j:3*j+3] - obs[t, b, 3*i:3*i+3], axis=-1)}")
                                if not found_cube:
                                    is_stacked = False
                                    break
                                check_count += 1
                            # Occlusion check
                            is_occluded = False
                            for j in range(plans.shape[-1]//3):
                                if i == j:
                                    continue
                                #print(f"Check Occluded: {i}, {goals[b, 3*i:3*i+3]} - {j}, {obs[t, b, 3*j:3*j+3]}, {np.linalg.norm(obs[t, b, 3*j:3*j+3] - goals[b, 3*i:3*i+3], axis=-1)}")
                                if np.linalg.norm(obs[t, b, 3*j:3*j+3] - goals[b, 3*i:3*i+3], axis=-1) < 0.15:
                                    is_occluded = True
                                    print(f"Occluded: {i}, {obs[t, b, 3*i:3*i+3]}")
                                    break
                            # If there is another cube on the top of the initial cube, then not success
                            upper_cube_z = starts[b, 3*i+2] + 0.5
                            cannot_move = False
                            for j in range(plans.shape[-1]//3):
                                if i == j:
                                    continue
                                #print(f"Check Cannot Move: {i}, {starts[b, 3*i:3*i+3]} - {j}, {starts[b, 3*j:3*j+3]}, {np.linalg.norm(starts[b, 3*j:3*j+3] - starts[b, 3*i:3*i+3], axis=-1)}")
                                if abs(starts[b, 3*j+2] - upper_cube_z) < 0.1 and np.linalg.norm(starts[b, 3*j:3*j+3] - starts[b, 3*i:3*i+3], axis=-1) < 0.6:
                                        cannot_move = True
                                        print(f"Cannot Move: {i}, {starts[b, 3*i:3*i+3]} - {j}, {starts[b, 3*j:3*j+3]}")
                                        break
                            if is_stacked and (not is_occluded) and (not cannot_move):
                                _successes[i] = True
                                succeed_cube_indices[b] = i
                                terminal_ts = t
                                newly_succeeded = True
                            else:
                                print(f"Not Stacked: {i}, {obs[t, b, 3*i:3*i+3]}")
                                not_stacked_list.append(i)
                        if self.jump > 1:
                            obs[t, b, 3*i:3*(i+1)] = obs_backup
                fully_success = all(_successes)
                partial_success = any(_successes) and not fully_success
                if self.open_loop_horizon < self.val_max_steps and newly_succeeded: # In replanning scenario, it is okay to achieve partial success
                    break
                if fully_success:
                    break
            if fully_success: # Fully solved
                values[b] = (obs.shape[0] - terminal_ts) / obs.shape[0]
                infos[b] = "Achieved"
                achieved_ts[b] = terminal_ts
                break
            if partial_success:
                values[b] = 0.1 * sum(_successes)
                infos[b] = "PartialAchieved"
            if (self.open_loop_horizon < self.val_max_steps) and newly_succeeded:
                infos[b] = "Achieved" # Record the partial success as achieved
                achieved_ts[b] = terminal_ts
                break

        # Post-processing
        for i in range(values.shape[0]):
            info = infos[i]
            achieved_t = achieved_ts[i]
            if info == "Achieved": # Object-wise planning
                if self.open_loop_horizon < self.val_max_steps: # When doing replanning, only one cube is handled per each planning
                    for j in range(plans.shape[-1]//3):
                        if j != succeed_cube_indices[i]:
                            normalized_plans[:, i, 3*j:3*j+3] = obs_normalized[0, 3*j:3*j+3]
                else: # If the object is almost not moving, then we use the goal position
                    for t in range(plans.shape[0]):
                        for j in range(plans.shape[-1]//3):
                            if t == 0:
                                pos_diff = torch.norm(goal_normalized[0, 3*j:(3*j+3)] - obs_normalized[0, 3*j:(3*j+3)], dim=-1)
                            else:
                                pos_diff = torch.norm(normalized_plans[t, i, 3*j:(3*j+3)] - normalized_plans[t-1, i, 3*j:(3*j+3)], dim=-1)
                            if pos_diff < 0.08:
                                normalized_plans[t, i, 3*j:(3*j+3)] = goal_normalized[0, 3*j:(3*j+3)]

        return normalized_plans, values, infos, achieved_ts, succeed_cube_indices

    def p_mctd_plan(self, obs_normalized, goal_normalized, horizon, conditions, start, goal, steps, succeed_cube_list=None, target_cube_idx=-1):
        assert start.shape[0] == 1, "the batch size must be 1"
        assert (not self.leaf_parallelization) or (self.parallel_search_num % len(self.mctd_guidance_scales) == 0), f"Parallel search num must be divisible by the number of guidance scales: {self.parallel_search_num} % {len(self.mctd_guidance_scales)} != 0"
        assert ("cube" in self.env_id) == (succeed_cube_list is not None), "succeed_cube_list must be provided for cube task"

        horizon = self.episode_len if horizon is None else horizon
        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        noise_level = self._generate_scheduling_matrix(plan_tokens)
        if "cube" in self.env_id:
            if target_cube_idx == -1:
                children_node_guidance_scales = []
                for i in range(obs_normalized.shape[-1]//3):
                    if i in succeed_cube_list:
                        continue
                    for j in list(self.mctd_guidance_scales):
                        children_node_guidance_scales.append(f"{i+1}-{j}")
            else:
                children_node_guidance_scales = [f"{target_cube_idx+1}-{j}" for j in list(self.mctd_guidance_scales)]
            print(f"Object-wise children_node_guidance_scales: {children_node_guidance_scales}")
        else:
            children_node_guidance_scales = self.mctd_guidance_scales
        max_search_num = self.mctd_max_search_num
        num_denoising_steps = self.mctd_num_denoising_steps
        skip_level_steps = self.mctd_skip_level_steps
        terminal_depth = np.ceil((noise_level.shape[0] - 1) / num_denoising_steps).astype(int)
        # Root Node (name, depth, parent_node, children_node_guidance_scale, plan_history)
        root_node = TreeNode('0', 0, None, children_node_guidance_scales, [], terminal_depth=terminal_depth, virtual_visit_weight=self.virtual_visit_weight)
        root_node.set_value(0) # Initialize the value of the root node

        # Search
        search_num, p_search_num, solved, achieved = 0, 0, False, False
        achieved_plans = [] # the plans that achieved the goal through the rollout
        not_reached_plans = [] # the plans that did not achieve the goal, but there is no warp through the rollout
        # lists for logging time
        selection_time, expansion_time, simulation_time, backprop_time, early_termination_time = [], [], [], [], [] # sum of the time for each batch
        simul_noiselevel_zero_padding_time = []
        simul_value_estimation_time = []
        simul_value_calculation_time = []
        simul_node_allocation_time = []
        while True:
            if self.time_limit is not None:
                if time.time() - self.start_time > self.time_limit:
                    break
            else:
                if p_search_num >= max_search_num:
                    break

            ## For checking the virtual visit count
            #root_node.check_virtual_visit_count()

            ###############################
            # Selection
            #  When leaf parallelization is True, then the selection is done in partially parallel (the children nodes from same parent node are selected at the same time)
            #  When leaf parallelization is False, then the selection is done in fully sequential (only one node is selected at a time)
            if not self.parallel_multiple_visits: # If parallel multiple visits is False, then we need to list all the nodes to expand
                expandable_node_names = root_node.get_expandable_node_names()
                #print(f"Expandable node names: {expandable_node_names}")
            selection_start_time = time.time()
            #print("============ Selection Start ============")
            psn = self.parallel_search_num
            selected_nodes, expanded_node_candidates = [], []
            while psn > 0:
                selected_node = root_node
                while (not selected_node.is_expandable(consider_virtually_visited=(not self.parallel_multiple_visits))) and (not selected_node.is_terminal()) and (selected_node.is_selectable()):
                    selected_node = selected_node.select(leaf_parallelization=self.leaf_parallelization)
                if selected_node.is_terminal() or (not selected_node.is_selectable() and not selected_node.is_expandable(consider_virtually_visited=(not self.parallel_multiple_visits))):
                    psn -= 1 if not self.leaf_parallelization else len(children_node_guidance_scales)
                    continue
                if self.leaf_parallelization:
                    for i in range(len(children_node_guidance_scales)):
                        # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                        expanded_node_candidate = selected_node.get_expandable_candidate(index=i, consider_virtually_visited=(not self.parallel_multiple_visits))
                        selected_nodes.append(selected_node)
                        expanded_node_candidates.append(expanded_node_candidate)
                        if not self.parallel_multiple_visits:
                            if not expanded_node_candidate['name'] in expandable_node_names:
                                raise ValueError(f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names")
                            expandable_node_names.remove(expanded_node_candidate['name'])
                        #print(f"Expanded node candidate {expanded_node_candidate['name']} is selected")
                        psn -= 1
                else:
                    # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                    expanded_node_candidate = selected_node.get_expandable_candidate(index=None, consider_virtually_visited=(not self.parallel_multiple_visits))
                    selected_nodes.append(selected_node)
                    expanded_node_candidates.append(expanded_node_candidate)
                    if not self.parallel_multiple_visits:
                        if not expanded_node_candidate['name'] in expandable_node_names:
                            raise ValueError(f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names")
                        expandable_node_names.remove(expanded_node_candidate['name'])
                    #print(f"Expanded node candidate {expanded_node_candidate['name']} is selected")
                    psn -= 1
                if not self.parallel_multiple_visits:
                    if len(expandable_node_names) == 0:
                        print("No more expandable nodes")
                        break
            if len(selected_nodes) == 0:
                print("No more selected nodes")
                break
            #print("============ Selection End ============")
            selection_end_time = time.time()
            selection_time.append(selection_end_time - selection_start_time)

            filtered_expanded_node_plan_hists = [None] * len(expanded_node_candidates)
            filtered_value_estimation_plan_hists = [None] * len(expanded_node_candidates)
            for _ in range(self.num_tries_for_bad_plans): # Trick used in MCTD to resample when the generated plan is terrible (e.g., not moving plans)
                ###############################
                # Expansion
                expansion_start_time = time.time()
                #print("============ Expansion Start ============")
                expanded_node_plans = []
                expanded_node_noise_levels = []
                expanded_node_guidance_scales = []
                for info in expanded_node_candidates:
                    if len(info["plan_history"]) == 0:
                        expanded_node_plans.append(None)
                    else:
                        expanded_node_plans.append(info["plan_history"][-1][-1].unsqueeze(1).to(self.device))
                    _noise_level = noise_level[(info["depth"] - 1) * num_denoising_steps : (info["depth"] * num_denoising_steps + 1)]
                    #if info["depth"] == terminal_depth:
                    _noise_level = np.concatenate([_noise_level] + [noise_level[-1:]]*(num_denoising_steps - _noise_level.shape[0]+1))
                    expanded_node_noise_levels.append(_noise_level)
                    expanded_node_guidance_scales.append(info["guidance_scale"])
                expanded_node_noise_levels = np.array(expanded_node_noise_levels, dtype=np.int32) # (batch_size, height, width)
                expanded_node_plan_hists = self.parallel_plan(
                    obs_normalized, goal_normalized, horizon, conditions,
                    guidance_scale=expanded_node_guidance_scales,
                    noise_level=expanded_node_noise_levels,
                    plan=expanded_node_plans,
                )
                #print(f"Expanded node plan hists: {expanded_node_plan_hists.shape}")
                #print("============ Expansion End ============")
                expansion_end_time = time.time()
                expansion_time.append(expansion_end_time - expansion_start_time)

                ###############################
                # Simulation
                #  It includes the noise level zero-padding, finding the max denoising steps, simulation, value calculation and node allocation
                simulation_start_time = time.time()
                #print("============ Simulation Start ============")

                # Pad the noise levels - Sequential
                simul_noiselevel_zero_padding_start = time.time()
                value_estimation_plans, value_estimation_noise_levels = [], []
                max_denoising_steps = 0
                for i in range(len(expanded_node_candidates)): # find the max denoising steps
                    _noise_level = np.concatenate(
                        [noise_level[(expanded_node_candidates[i]["depth"] * num_denoising_steps)::skip_level_steps],
                        noise_level[-1:]], axis=0)
                    # update max denoising steps
                    if _noise_level.shape[0] > max_denoising_steps:
                        max_denoising_steps = _noise_level.shape[0]
                    value_estimation_noise_levels.append(_noise_level)
                    value_estimation_plans.append(expanded_node_plan_hists[-1, :, i].unsqueeze(1))
                for i in range(len(expanded_node_candidates)): # zero-padding
                    length = value_estimation_noise_levels[i].shape[0]                
                    if length < max_denoising_steps:
                        value_estimation_noise_levels[i] = np.concatenate([
                            value_estimation_noise_levels[i], 
                            np.zeros((max_denoising_steps - length, value_estimation_noise_levels[i].shape[1]), dtype=np.int32)], 
                            axis=0) # zero-padding
                simul_noiselevel_zero_padding_end = time.time()
                simul_noiselevel_zero_padding_time.append(simul_noiselevel_zero_padding_end - simul_noiselevel_zero_padding_start)

                # Simulation - Value Estimation
                simul_value_estimation_start = time.time()
                value_estimation_noise_levels = np.array(value_estimation_noise_levels, dtype=np.int32)
                value_estimation_plan_hists = self.parallel_plan(
                    obs_normalized, goal_normalized, horizon, conditions,
                    guidance_scale=expanded_node_guidance_scales,
                    noise_level=value_estimation_noise_levels,
                    plan=value_estimation_plans,
                )
                simul_value_estimation_end = time.time()
                #print(f"Value estimation plan hist: {value_estimation_plan_hists.shape}")

                # check if any plan is good
                #if "cube" in self.env_id and (self.open_loop_horizon < self.val_max_steps): # In cube case, testing through value
                #    value_estimation_plans, values, infos, achieved_ts, succeed_cube_indices = self.calculate_values_cube(value_estimation_plan_hists[-1], start, goal, obs_normalized, goal_normalized, succeed_cube_list) # (plan_len, N, D), (N, D), (N, D)
                #    for i in range(len(infos)):
                #        if filtered_expanded_node_plan_hists[i] is None and infos[i] == "Achieved" and values[i] > 0.5:
                #            filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                #            filtered_value_estimation_plan_hists[i] = value_estimation_plan_hists[:, :, i]
                #else:
                plans = self._unnormalize_x(value_estimation_plan_hists[-1])[:-1].detach().cpu().numpy()
                diffs = np.linalg.norm(plans[1:] - plans[:-1], axis=-1) # (plan_len-1, N)
                for i in range(diffs.shape[1]):
                    if filtered_expanded_node_plan_hists[i] is None and not np.all(diffs[:, i] < 0.1):
                        filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                        filtered_value_estimation_plan_hists[i] = value_estimation_plan_hists[:, :, i]

                if None in filtered_expanded_node_plan_hists:
                    print("No good plan found, resampling")
                    simulation_end_time = time.time()
                    simulation_time.append(simulation_end_time - simulation_start_time)
                    continue
                else:
                    break
            for i in range(len(filtered_expanded_node_plan_hists)):
                if filtered_expanded_node_plan_hists[i] is None:
                    filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                    filtered_value_estimation_plan_hists[i] = value_estimation_plan_hists[:, :, i]
            expanded_node_plan_hists = torch.stack(filtered_expanded_node_plan_hists, dim=2)
            value_estimation_plan_hists = torch.stack(filtered_value_estimation_plan_hists, dim=2)

            # Value Calculation
            simul_value_calculation_start = time.time()
            if "cube" in self.env_id:
                value_estimation_plans, values, infos, achieved_ts, succeed_cube_indices = self.calculate_values_cube(value_estimation_plan_hists[-1], start, goal, obs_normalized, goal_normalized, succeed_cube_list) # (plan_len, N, D), (N, D), (N, D)
            else:
                values, infos, achieved_ts = self.calculate_values(value_estimation_plan_hists[-1], start, goal) # (plan_len, N, D), (N, D), (N, D)
                value_estimation_plans = value_estimation_plan_hists[-1]

            viz_value_estimation_plans = []
            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    viz_value_estimation_plans.append(value_estimation_plans[:achieved_t, i])
                else:
                    viz_value_estimation_plans.append(value_estimation_plans[:, i])

            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    achieved_plans.append([value_estimation_plans[:achieved_t, i], values[i]])
                    achieved = True
                elif info == "NotReached":
                    not_reached_plans.append([value_estimation_plans[:, i], values[i]])
            print(f"Value Calculation: {values}, {infos}")
            simul_value_calculation_end = time.time()

            # Node Allocation
            simul_node_allocation_start = time.time()
            selected_nodes_for_expansion = {}
            expanded_node_infos = {}
            for i in range(len(expanded_node_candidates)):
                name = expanded_node_candidates[i]["name"]
                if name not in expanded_node_infos:
                    selected_nodes_for_expansion[name] = selected_nodes[i]
                    expanded_node_infos[name] = expanded_node_candidates[i]
                    expanded_node_infos[name]["plan_history"].append([])
                value = values[i]
                plan_hist = expanded_node_plan_hists[:, :, i]
                value_estimation_plan = value_estimation_plans[:, i]
                if expanded_node_infos[name]["value"] is None:
                    expanded_node_infos[name]["value"] = value
                    expanded_node_infos[name]["value_estimation_plan"] = value_estimation_plan.detach().cpu()
                    expanded_node_infos[name]["plan_history"][-1] = plan_hist.detach().cpu()
                else:
                    if value > expanded_node_infos[name]["value"]:
                        expanded_node_infos[name]["value"] = value
                        expanded_node_infos[name]["value_estimation_plan"] = value_estimation_plan.detach().cpu()
                        expanded_node_infos[name]["plan_history"][-1] = plan_hist.detach().cpu()
            for name in selected_nodes_for_expansion:
                selected_nodes_for_expansion[name].expand(**expanded_node_infos[name])
            simul_node_allocation_end = time.time()
            simul_node_allocation_time.append(simul_node_allocation_end - simul_node_allocation_start)

            #print("============ Simulation End ============")
            simulation_end_time = time.time()
            simulation_time.append(simulation_end_time - simulation_start_time)

            ######################
            # Backpropagation
            #  When leaf parallelization is True, then the backpropagation is done in partially parallel (the leafs from same parent node are backpropagated at the same time)
            #  When leaf parallelization is False, then the backpropagation is done in fully sequential (only one node is backpropagated at a time)
            backprop_start_time = time.time()
            #print("============ Backpropagation Start ============")

            distinct_selected_nodes = np.unique(selected_nodes)
            for selected_node in distinct_selected_nodes:
                selected_node.backpropagate()

            #print("============ Backpropagation End ============")
            backprop_end_time = time.time()
            backprop_time.append(backprop_end_time - backprop_start_time)

            ######################
            # Early Termination
            early_termination_start_time = time.time()
            #print("============ Early Termination Start ============")

            plans = torch.stack([info["plan_history"][-1][-1] for info in expanded_node_infos.values()], dim=1).to(self.device)
            if "cube" in self.env_id:
                plans, _, infos, achieved_ts, succeed_cube_indices = self.calculate_values_cube(plans, start, goal, obs_normalized, goal_normalized, succeed_cube_list) # (plan_len, N, D), (N, D), (N, D)
            else:
                _, infos, achieved_ts = self.calculate_values(plans, start, goal) # (plan_len, N, D), (N, D), (N, D)
            print(f"Early Termination: {infos}, {achieved_ts}")
            solved = False
            viz_plans = []
            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    terminal_ts = achieved_t
                    viz_plans.append(plans[:terminal_ts, i])
                else:
                    viz_plans.append(plans[:, i])

            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    solved = True
                    terminal_ts = achieved_t
                    solved_plan = plans[:terminal_ts, i]
                    break

            #print("============ Early Termination End ============")
            early_termination_end_time = time.time()
            early_termination_time.append(early_termination_end_time - early_termination_start_time)

            if self.viz_plans:
                for i in range(len(viz_plans)):
                    names = [info["name"] for info in expanded_node_infos.values()]
                    self.visualize_node_value_plans(steps, search_num+i+1, values[i][None], [names], viz_plans[i][:,None],
                        viz_value_estimation_plans[i][:,None], start, goal)

            search_num += 1
            p_search_num += len(expanded_node_candidates)
            if search_num % 100 == 0:
                print(f"search_num: {search_num}, p_search_num: {p_search_num}")

            if (self.early_stopping_condition == "solved" and solved) or (self.early_stopping_condition == "achieved" and achieved):
                break

        if solved:
            output_plan = torch.cat([solved_plan[:,None], goal_normalized[None]], dim=0)[None] # (1, t, 1, c)
        else:
            if len(achieved_plans) != 0:
                max_value = -1
                max_plan = None
                for plan, value in achieved_plans:
                    assert value >= 0, f"The value is negative: {value}"
                    if value > max_value:
                        max_value = value
                        max_plan = plan
                output_plan = torch.cat([max_plan[:,None], goal_normalized[None]], dim=0)[None] # (1, t, 1, c)
            elif len(not_reached_plans) != 0:
                max_value = -1
                max_plan = None
                for plan, value in not_reached_plans:
                    assert value >= 0, f"The value is negative: {value}"
                    if value > max_value:
                        max_value = value
                        max_plan = plan
                output_plan = max_plan[None,:,None] # (1, t, 1, c)
            else:
                print("Failed to find the plan")
                output_plan = torch.cat([obs_normalized[None]]*horizon, dim=0)[None] # (1, t, 1, c) failed to find the plan

        self.log(f"validation/search_num", search_num)
        self.log(f"validation/p_search_num", p_search_num)

        self.log(f"validation_time/selection_time", np.sum(selection_time))
        self.log(f"validation_time/expansion_time", np.sum(expansion_time))
        self.log(f"validation_time/simulation_time", np.sum(simulation_time))
        self.log(f"validation_time/backprop_time", np.sum(backprop_time))
        self.log(f"validation_time/early_termination_time", np.sum(early_termination_time))

        self.log(f"validation_time/simul_noiselevel_zero_padding_time", np.sum(simul_noiselevel_zero_padding_time))
        self.log(f"validation_time/simul_value_estimation_time", np.sum(simul_value_estimation_time))
        self.log(f"validation_time/simul_value_calculation_time", np.sum(simul_value_calculation_time))
        self.log(f"validation_time/simul_node_allocation_time", np.sum(simul_node_allocation_time))
        return output_plan
