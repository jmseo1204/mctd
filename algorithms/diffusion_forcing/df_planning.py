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
        self.warp_threshold = cfg.warp_threshold * self.jump
        self.leaf_parallelization = cfg.leaf_parallelization
        self.parallel_multiple_visits = cfg.parallel_multiple_visits
        self.early_stopping_condition = cfg.early_stopping_condition
        self.num_tries_for_bad_plans = cfg.num_tries_for_bad_plans
        self.sub_goal_interval = cfg.sub_goal_interval
        self.viz_plans = cfg.viz_plans
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
        nonterminals = nonterminals.bool().permute(1, 0) # (T, B)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        # masks = torch.cat([masks[:-self.frame_stack:self.jump], masks[-self.frame_stack:]], dim=0)

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1) # (b t c_o)
        bundles = self._normalize_x(self.make_bundle(observations, actions, rewards))  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = 0  # zero out actions and rewards after normalization
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b t c)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous() # t b fs c -> t b fs*c

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
        horizon = int(horizon)
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
        horizon = int(horizon)
        # start and goal are numpy arrays of shape (b, obs_dim)
        # start and goal are assumed to be normalized
        # returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan
        # assert batch_size == 1, "parallel planning only supports batch size 1"

        batch_size = len(plan)
        start = torch.cat([start] * batch_size, 0)
        goal = torch.cat([goal] * batch_size, 0)            

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
                weight = torch.from_numpy(weight).float().to(self.device)
                
                dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                dist_o = dist_o[:, :, : 2]
                dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=1).sqrt()
                dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = dist_o
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]
                episode_return = -(dist * weight).mean(dim=(0, 2)) * 1000 * dist.shape[1] / 16
            else:
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")

            return (guidance_scale * episode_return).mean()

        def particle_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            # Implementation of Particle Guidance (PG) from "Tree-Guided Diffusion Planner (TDP)"
            # This function computes a diversity score based on an RBF kernel to repel particles from each other.
            b = x.shape[1]
            if b <= 1:
                return torch.tensor(0.0, device=x.device)

            # Flatten trajectories to [batch_size, trajectory_dim]
            # Each particle's trajectory is treated as a single point in high-dimensional space.
            x_flat = rearrange(x, "t b (fs c) -> b (t fs c)", fs=self.frame_stack)
            
            # Compute pairwise squared Euclidean distances between all particles in the batch
            # Shape: [b, b]
            dist_sq = torch.cdist(x_flat, x_flat, p=2).pow(2)
            
            # Median trick for the RBF kernel bandwidth (h) to ensure it's scale-invariant.
            # We detach it to avoid backpropagation through the bandwidth itself.
            h = torch.median(dist_sq.detach())
            if h == 0: 
                h = 1.0 # Fallback to avoid division by zero
            
            # Compute RBF Kernel matrix K_ij = exp(-||x_i - x_j||^2 / h)
            kernel_matrix = torch.exp(-dist_sq / h)
            
            # The similarity score is the average of the off-diagonal elements of the kernel matrix.
            # We want to minimize this sum to maximize diversity.
            similarity = (kernel_matrix.sum() - b) / (b * (b - 1))
            
            # Return negative similarity so that the gradient points towards higher diversity.
            return -similarity

        guidance_fn = particle_guidance if guidance_scale is not None else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1 # 1 means init_token
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
        chunk = torch.cat(chunk, 1) # (T,B, fs*c)
        if len(chunk.shape) == 2:
            chunk = chunk.unsqueeze(0)
        pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0) # (n_tokens, B, fs*c)

        plan_hist = [plan.detach().clone()[: self.n_tokens - pad_tokens]]
        stabilization = 0

        for m in range(scheduling_matrix.shape[1] - 1): # (B, M, T)
            from_noise_levels = np.concatenate(
                [
                    np.full((batch_size, 1), stabilization, dtype=np.int64),  # Shape (batch_size, 1)
                    scheduling_matrix[:,m], # initially (B, M, T) -> (B, T)
                    np.full((batch_size, pad_tokens), self.sampling_timesteps, dtype=np.int64),  # Shape (batch_size, pad_tokens), num_denoising_steps means MAX_NOISE_LEVEL
                ]
                , axis=1
            ) # (B, T)
            to_noise_levels = np.concatenate(
                [
                    np.full((batch_size, 1), stabilization, dtype=np.int64),  # Shape (batch_size, 1)
                    scheduling_matrix[:,m+1], # (batch_size, T)
                    np.full((batch_size, pad_tokens), self.sampling_timesteps, dtype=np.int64),  # Shape (batch_size, pad_tokens)
                ]
                , axis=1
            ) # (B, T)
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
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon] # (m, init_token 제외한 1~T 개의 fs, b, c)
        return plan_hist

    def _generate_plan_between_points(self, start_normalized, goal_normalized, start_raw, goal_raw, conditions, horizon_scale=0.4, tag="mcts_plan"):
        """
        Helper function to generate a plan between two points.
        
        Args:
            start_normalized: Normalized start observation
            goal_normalized: Normalized goal observation
            start_raw: Raw (unnormalized) start observation for MCTD
            goal_raw: Raw (unnormalized) goal observation for MCTD
            conditions: Planning conditions
            horizon_scale: Multiplier for episode_len (default: 0.4)
        
        Returns:
            plan: Unnormalized plan trajectory (t b c)
            plan_hist: Full plan history (m t b c) or (D, M, T, B, C) for MCTD
        """
        if self.mctd:
            plan_hist = self.p_mctd_plan(
                start_normalized, goal_normalized, 
                self.episode_len * horizon_scale, conditions, 
                start_raw[:, :self.observation_dim], 
                goal_raw[:, :self.observation_dim],
                tag=tag
            )
            plan_hist = self._unnormalize_x(plan_hist)
            plan = plan_hist[-1]  # (t b c)
        else:
            plan_hist = self.plan(
                start_normalized, goal_normalized, 
                self.episode_len * horizon_scale, conditions
            )
            plan_hist = self._unnormalize_x(plan_hist)  # (m t b c)
            plan = plan_hist[-1]  # (t b c)
        
        return plan, plan_hist

    def interact(self, batch_size: int, conditions=None, namespace="validation"):
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
                envs = DummyVecEnv([lambda: ogbench.locomaze.maze.make_maze_env('point','maze',maze_type=self.env_id.split("-")[1])] * batch_size)
                if self.action_dim == 2:
                    use_diffused_action = True
            elif "antmaze" in self.env_id:
                envs = DummyVecEnv([lambda: ogbench.locomaze.maze.make_maze_env('ant','maze',maze_type=self.env_id.split("-")[1])] * batch_size)
                #use_diffused_action = True
                from dql.main_Antmaze import hyperparameters
                from dql.agents.ql_diffusion import Diffusion_QL as Agent
                params = hyperparameters[self.dataset]
                state_dim = envs.observation_space.shape[0]
                action_dim = envs.action_space.shape[0]
                max_action = float(envs.action_space.high[0])
                agent = Agent(
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
                else:
                    raise ValueError(f"Dataset {self.dataset} not supported")
                
                import os
                agent.load_model(os.path.join(os.getcwd(), "dql", "results", dql_folder), id=200)
            for i, env in enumerate(envs.envs):
                env.set_task(self.task_id + i)
                #env.set_seed(self.interaction_seed)
        else:
            envs = DummyVecEnv([lambda: gym.make(self.env_id)] * batch_size)
            envs.seed(self.interaction_seed)

        terminate = False
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        obs = envs.reset()
        # Randomize the goal for each environment
        if self.env_id in OGBENCH_ENVS: # OGBench goal setting is already done through set_task()   
            pass
        else:
            if self.use_random_goals_for_interaction:
                for env in envs.envs:
                    env.set_target()

        obs = torch.from_numpy(obs).float().to(self.device)
        start = obs.detach()
        obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

        if self.env_id in OGBENCH_ENVS: # OGBench
            goal = np.vstack([envs.reset_infos[i]['goal'] for i in range(len(envs.reset_infos))])
        else:
            goal = np.concatenate([[env.env._target] for env in envs.envs])
        goal = torch.Tensor(goal).float().to(self.device)
        goal = torch.cat([goal, torch.zeros_like(goal)], -1)
        goal = goal[:, : self.observation_dim]
        goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()

        steps = 0
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        reached = np.zeros(batch_size, dtype=bool)
        first_reach = np.zeros(batch_size)

        trajectory = []  # actual trajectory
        all_plan_hist = []  # a list of plan histories, each history is a collection of m diffusion steps

        # run mpc with diffused actions
        planning_time = []
        while not terminate and steps < self.val_max_steps:
            planning_start_time = time.time()
            
            # Generate forward plan (start → goal)
            plan, plan_hist = self._generate_plan_between_points(
                obs_normalized, goal_normalized,
                start.cpu().numpy(), goal.cpu().numpy(),
                conditions, horizon_scale=0.4, tag="mcts_plan_from_start"
            )
            
            # Generate reverse plan (goal → start) for visualization only
            reverse_plan, _ = self._generate_plan_between_points(
                goal_normalized, obs_normalized,
                goal.cpu().numpy(), start.cpu().numpy(),
                conditions, horizon_scale=0.4, tag="mcts_plan_from_goal"
            )
            
           # Visualization with both forward and reverse trajectories
            start_numpy = start.cpu().numpy()[:, :2]
            goal_numpy = goal.cpu().numpy()[:, : self.observation_dim]
            
            # Create forward trajectory image
            forward_image = make_trajectory_images(
                self.env_id, 
                plan[:, :, :2].detach().cpu().numpy(), 
                1, start_numpy, goal_numpy, 
                self.plot_end_points
            )[0]
            self.log_image(f"plan/plan_at_{steps}_from_start", Image.fromarray(forward_image))

            # Create reverse trajectory image (swap start and goal for visualization)
            reverse_image = make_trajectory_images(
                self.env_id, 
                reverse_plan[:, :, :2].detach().cpu().numpy(), 
                1, goal_numpy[:, :2], start_numpy, 
                self.plot_end_points
            )[0]
            self.log_image(f"plan/plan_at_{steps}_from_goal", Image.fromarray(reverse_image))

            planning_end_time = time.time()
            planning_time.append(planning_end_time - planning_start_time)


            # TODO: we don't have to do below process if the output plan is infeasible(unachieved) (break or continue)


            # jumpy case (fill the gap)
            if self.jump > 1:
                _plan = []
                for t in range(plan.shape[0]):
                    for j in range(self.jump):
                        _plan.append(plan[t, : , :2])
                plan = torch.stack(_plan)

            all_plan_hist.append(plan_hist.cpu())

            obs_numpy = obs.detach().cpu().numpy()
            if "antmaze" in self.env_id:
                #sub_goal = plan[self.open_loop_horizon - 1, :, :2].detach().cpu().numpy()
                sub_goal = plan[self.sub_goal_interval, :, :2].detach().cpu().numpy()
                sub_goal_step = self.sub_goal_interval
            for t in range(self.open_loop_horizon):
                if use_diffused_action:
                    _, action, _ = self.split_bundle(plan[t])
                else:
                    if "antmaze" in self.env_id:
                        if np.linalg.norm(obs_numpy[0, :2] - sub_goal[0]) < 1.0:
                            if sub_goal_step < plan.shape[0] - self.sub_goal_interval:
                                print(f"sub_goal_step {sub_goal_step} achieved...")
                                sub_goal_step += self.sub_goal_interval
                                sub_goal = plan[sub_goal_step, :, :2].detach().cpu().numpy()
                            else:
                                sub_goal = plan[-1, :, :2].detach().cpu().numpy()
                        assert obs_numpy.shape[0] == 1, f"Batch size must be 1 for AntMaze, got {obs_numpy.shape[0]}"
                        action = agent.sample_action(obs_numpy, sub_goal)
                        action = torch.from_numpy(action).float().reshape(1, -1)
                    else:
                        if t == 0:
                            plan_vel = plan[t, :, :2] - obs[:, :2]
                        else:
                            if t < plan.shape[0]:
                                plan_vel = plan[t, :, :2] - plan[t - 1, :, :2]
                            else:
                                plan_vel = 0
                        if t < plan.shape[0]:
                            action = 12.5 * (plan[t, :, :2] - obs[:, :2]) + 1.2 * (plan_vel - obs[:, 2:])
                        else:
                            action = 12.5 * (plan[-1, :, :2] - obs[:, :2]) + 1.2 * (plan_vel - obs[:, 2:])
                action = torch.clip(action, -1, 1).detach().cpu()
                obs_numpy, reward, done, _ = envs.step(np.nan_to_num(action.numpy()))

                reached = np.logical_or(reached, reward >= 1.0)
                episode_reward += reward
                episode_reward_if_stay += np.where(~reached, reward, 1)
                first_reach += ~reached

                if done.any():
                    terminate = True
                    break

                obs, reward, done = [torch.from_numpy(item).float() for item in [obs_numpy, reward, done]]
                bundle = self.make_bundle(obs, action, reward[..., None])
                trajectory.append(bundle)
                obs = obs.to(self.device)
                obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

                steps += 1
        self.log(f"{namespace}/planning_time", np.sum(planning_time))
        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())
        self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0) / batch_size)

        # Visualization
        #samples = min(16, batch_size)
        samples = min(32, batch_size)
        trajectory = torch.stack(trajectory)
        start = start[:, :2].cpu().numpy().tolist()
        goal = goal[:, :2].cpu().numpy().tolist()
        images = make_trajectory_images(self.env_id, trajectory, samples, start, goal, self.plot_end_points)

        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_interaction/sample_{i}",
                Image.fromarray(img),
            )

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

    def visualize_node_value_plans(self, search_num, values, names, plans, value_plans, starts, goals, tag="mcts_plan"):
        if value_plans.shape[1] != starts.shape[0]:
            starts = starts.repeat(value_plans.shape[1], axis=0)
        if value_plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(value_plans.shape[1], axis=0)
        
        value_plans = self._unnormalize_x(value_plans)
        value_plan_obs, _, _ = self.split_bundle(value_plans)
        value_plan_obs = value_plan_obs.detach().cpu().numpy()[:-1]
        value_plan_images = make_trajectory_images(self.env_id, value_plan_obs, value_plan_obs.shape[1], starts, goals, self.plot_end_points)
        for i in range(len(value_plan_images)):
            img = value_plan_images[i]
            self.log_image(
                f"{tag}/{search_num+i+1}_{names[i]}_V{values[i]}",
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


    def p_mctd_plan(self, obs_normalized, goal_normalized, horizon, conditions, start, goal, tag="mcts_plan"):
        assert start.shape[0] == 1, "the batch size must be 1"
        assert (not self.leaf_parallelization) or (self.parallel_search_num % len(self.mctd_guidance_scales) == 0), f"Parallel search num must be divisible by the number of guidance scales: {self.parallel_search_num} % {len(self.mctd_guidance_scales)} != 0"
        horizon = int(self.episode_len if horizon is None else horizon)
        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        noise_level = self._generate_scheduling_matrix(plan_tokens)
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
                #if search_num >= max_search_num:
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
            print("============ Selection Start ============")
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
            print("============ Selection End ============")
            selection_end_time = time.time()
            selection_time.append(selection_end_time - selection_start_time)

            filtered_expanded_node_plan_hists = [None] * len(expanded_node_candidates) # the elements can be left as None is every states are at the same point
            filtered_value_estimation_plan_hists = [None] * len(expanded_node_candidates)
            for _ in range(self.num_tries_for_bad_plans): # Trick used in MCTD to resample when the generated plan is terrible (e.g., not moving plans)
                ###############################
                # Expansion
                expansion_start_time = time.time()
                print("============ Expansion Start ============")
                expanded_node_plans = []
                expanded_node_noise_levels = []
                expanded_node_guidance_scales = [] 
                for info in expanded_node_candidates:
                    if len(info["plan_history"]) == 0:
                        expanded_node_plans.append(None)
                    else:
                        expanded_node_plans.append(info["plan_history"][-1][-1].unsqueeze(1)) # plan_history shape: (D, M, T, B, C)-> MCA & fully denoised
                    _noise_level = noise_level[(info["depth"] - 1) * num_denoising_steps : (info["depth"] * num_denoising_steps + 1)]
                    #if info["depth"] == terminal_depth:
                    _noise_level = np.concatenate([_noise_level] + [noise_level[-1:]]*(num_denoising_steps - _noise_level.shape[0]+1)) # (num_denoising_steps, T)
                    expanded_node_noise_levels.append(_noise_level)
                    expanded_node_guidance_scales.append(info["guidance_scale"])
                expanded_node_guidance_scales = torch.tensor(expanded_node_guidance_scales).to(obs_normalized.device) # (batch_size,)
                expanded_node_noise_levels = np.array(expanded_node_noise_levels, dtype=np.int32) # (batch_size, height, width)
                expanded_node_plan_hists = self.parallel_plan(
                    obs_normalized, goal_normalized, horizon, conditions,
                    guidance_scale=expanded_node_guidance_scales,
                    noise_level=expanded_node_noise_levels,
                    plan=expanded_node_plans,
                ) # shape: m (t fs) b c         but the last trajectory is what we want.
                print(f"Expanded node plan hists: {expanded_node_plan_hists.shape}")
                print("============ Expansion End ============")
                expansion_end_time = time.time()
                expansion_time.append(expansion_end_time - expansion_start_time)

                ###############################
                # Simulation
                #  It includes the noise level zero-padding, finding the max denoising steps, simulation, value calculation and node allocation
                simulation_start_time = time.time()
                print("============ Simulation Start ============")

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
                    value_estimation_plans.append(expanded_node_plan_hists[-1, :, i].unsqueeze(1)) # added one dim: (t fs) 1 c
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
                ) # shape: m (t fs) b c         but the last trajectory is what we want.
                simul_value_estimation_end = time.time()
                print(f"Value estimation plan hist: {value_estimation_plan_hists.shape}")

                # check if any plan is good
                plans = self._unnormalize_x(value_estimation_plan_hists[-1])[:-1].detach().cpu().numpy() # (t fs) b c
                diffs = np.linalg.norm(plans[1:] - plans[:-1], axis=-1) # (plan_len-1, N) # N is the number of expanded nodes(=batch_size)
                for i in range(diffs.shape[1]):
                    if filtered_expanded_node_plan_hists[i] is None and not np.all(diffs[:, i] < 0.1):
                        filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i] # b m (t fs) c
                        filtered_value_estimation_plan_hists[i] = value_estimation_plan_hists[:, :, i] # b m (t fs) c
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
            expanded_node_plan_hists = torch.stack(filtered_expanded_node_plan_hists, dim=2) # m (t fs) b c
            value_estimation_plan_hists = torch.stack(filtered_value_estimation_plan_hists, dim=2) # m (t fs) b c

            # Value Calculation
            simul_value_calculation_start = time.time()
            values, infos, achieved_ts = self.calculate_values(value_estimation_plan_hists[-1], start, goal) # (plan_len, N, D), (N, D), (N, D)
            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    achieved_plans.append([value_estimation_plan_hists[-1, :achieved_t, i], values[i]])
                    achieved = True
                elif info == "NotReached":
                    not_reached_plans.append([value_estimation_plan_hists[-1, :, i], values[i]])
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
                value_estimation_plan = value_estimation_plan_hists[-1, :, i]
                if expanded_node_infos[name]["value"] is None:
                    expanded_node_infos[name]["value"] = value
                    expanded_node_infos[name]["value_estimation_plan"] = value_estimation_plan
                    expanded_node_infos[name]["plan_history"][-1] = plan_hist
                else:
                    if value > expanded_node_infos[name]["value"]:
                        expanded_node_infos[name]["value"] = value
                        expanded_node_infos[name]["value_estimation_plan"] = value_estimation_plan
                        expanded_node_infos[name]["plan_history"][-1] = plan_hist
            for name in selected_nodes_for_expansion:
                selected_nodes_for_expansion[name].expand(**expanded_node_infos[name])
            simul_node_allocation_end = time.time()
            simul_node_allocation_time.append(simul_node_allocation_end - simul_node_allocation_start)

            print("============ Simulation End ============")
            simulation_end_time = time.time()
            simulation_time.append(simulation_end_time - simulation_start_time)

            ######################
            # Backpropagation
            #  When leaf parallelization is True, then the backpropagation is done in partially parallel (the leafs from same parent node are backpropagated at the same time)
            #  When leaf parallelization is False, then the backpropagation is done in fully sequential (only one node is backpropagated at a time)
            backprop_start_time = time.time()
            print("============ Backpropagation Start ============")

            distinct_selected_nodes = np.unique(selected_nodes)
            for selected_node in distinct_selected_nodes:
                selected_node.backpropagate()

            print("============ Backpropagation End ============")
            backprop_end_time = time.time()
            backprop_time.append(backprop_end_time - backprop_start_time)

            ######################
            # Early Termination
            early_termination_start_time = time.time()
            print("============ Early Termination Start ============")

            plans = torch.stack([info["plan_history"][-1][-1] for info in expanded_node_infos.values()], dim=1) # plan_history shape: (D, M, T, B, C)-> MCA & fully denoised
            _, infos, achieved_ts = self.calculate_values(plans, start, goal) # (plan_len, N, D), (N, D), (N, D)
            print(f"Early Termination: {infos}, {achieved_ts}")
            solved = False
            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    solved = True
                    terminal_ts = achieved_t
                    solved_plan = plans[:terminal_ts, i]
                    break

            print("============ Early Termination End ============")
            early_termination_end_time = time.time()
            early_termination_time.append(early_termination_end_time - early_termination_start_time)

            if self.viz_plans:
                terminal_indices = [i for i, info in enumerate(expanded_node_candidates) if info["depth"] == terminal_depth]
                if len(terminal_indices) > 0:
                    terminal_values = values[terminal_indices]
                    terminal_names = [expanded_node_candidates[i]["name"] for i in terminal_indices]
                    terminal_expanded_hists = expanded_node_plan_hists[:, :, terminal_indices]
                    terminal_estimation_hists = value_estimation_plan_hists[:, :, terminal_indices]
                    self.visualize_node_value_plans(search_num, terminal_values, terminal_names, 
                        terminal_expanded_hists[-1], terminal_estimation_hists[-1], start, goal, tag=tag)

            search_num += 1
            p_search_num += len(expanded_node_candidates)

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
