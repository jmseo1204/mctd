from typing import Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from omegaconf import DictConfig
from tqdm import tqdm
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


@dataclass
class MCTSTreeState:
    """Container holding all state for a single MCTS tree instance."""
    # --- Static config (set at init, never mutated) ---
    root_node: TreeNode
    plan_tokens: int
    terminal_depth: int
    noise_level: Optional[np.ndarray]          # unidirectional only; None when bidirectional_search=True
    children_node_guidance_scales: list
    max_search_num: int
    num_denoising_steps: int
    skip_level_steps: int
    from_start: bool
    tag: str
    # Root observation (unnormalized): start for tree1, goal for tree2.
    # Used to track agent positions across bidirectional expansion rounds.
    tree_root_obs: Optional[np.ndarray] = None  # shape (obs_dim,)
    # --- Mutable search state (updated by _run_mcts_search) ---
    search_num: int = 0
    p_search_num: int = 0
    max_depth: int = 0
    solved: bool = False
    achieved: bool = False
    solved_plan: Optional[torch.Tensor] = None
    achieved_plans: List = field(default_factory=list)
    not_reached_plans: List = field(default_factory=list)
    pbar: Any = None
    # --- Timing lists ---
    selection_time: List = field(default_factory=list)
    expansion_time: List = field(default_factory=list)
    simulation_time: List = field(default_factory=list)
    backprop_time: List = field(default_factory=list)
    early_termination_time: List = field(default_factory=list)
    simul_noiselevel_zero_padding_time: List = field(default_factory=list)
    simul_value_estimation_time: List = field(default_factory=list)
    simul_value_calculation_time: List = field(default_factory=list)
    simul_node_allocation_time: List = field(default_factory=list)



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
        
        # Manually initialize frame_stack as requested to solve dependency order
        self.frame_stack = cfg.frame_stack
        assert self.episode_len % self.frame_stack == 0, "Episode length must be divisible by frame stack size"
        self.n_tokens = self.episode_len // self.frame_stack
        
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
        self.mctd_denoising_steps_per_segment = cfg.mctd_denoising_steps_per_segment
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
        self.bidirectional_search = cfg.bidirectional_search
        self.meeting_delta = cfg.get('meeting_delta', 0.5)
        self.debug = cfg.get('DEBUG', False)
        self.sequence_dividing_factor = cfg.sequence_dividing_factor
        self.is_unknown_final_token = cfg.get('is_unknown_final_token', False)
        self.horizon_scale = cfg.horizon_scale
        self.pyramid = cfg.get('pyramid', False)
        
        # HILP value function guidance
        self.use_hilp_guidance = cfg.get('use_hilp_guidance', False)
        self.hilp_checkpoint_path = cfg.get('hilp_checkpoint_path', 'td_models/hilp_ckpt_latest.pt')
        self.hilp_obs_dim = cfg.get('hilp_obs_dim', 29)
        self.hilp_skill_dim = cfg.get('hilp_skill_dim', 32)
        self.hilp_value_fn = None  # Will be loaded lazily when needed
        self.anchor_guidance_scale = cfg.get('anchor_guidance_scale', 40.0)
        self.rdf_guidance_scale = cfg.get('rdf_guidance_scale', 2.0)
        self.mcts_use_sim = cfg.get('mcts_use_sim', True)
        
        super().__init__(cfg)
        self.plot_end_points = cfg.plot_start_goal and self.guidance_scale != 0

    def _get_hilp_value_fn(self):
        """Lazy loader for HILP value function model."""
        if self.hilp_value_fn is None:
            import sys
            import os
            # Add algorithms directory to path to import cleandiffuser_ex
            algorithms_dir = os.path.join(os.path.dirname(__file__), '..')
            if algorithms_dir not in sys.path:
                sys.path.insert(0, algorithms_dir)
            from cleandiffuser_ex.hilp import HILP
            
            # Load HILP model
            self.hilp_value_fn = HILP(
                obs_dim=self.hilp_obs_dim,
                skill_dim=self.hilp_skill_dim,
                device=self.device,
                value_hidden_dims=(512, 512, 512),
                use_layer_norm=True
            )
            self.hilp_value_fn.load(self.hilp_checkpoint_path)
            self.hilp_value_fn.eval()
            
            # Freeze all parameters to prevent gradient updates
            for param in self.hilp_value_fn.parameters():
                param.requires_grad = False
            
            print(f"[HILP] Loaded HILP value function from {self.hilp_checkpoint_path}")
                
        return self.hilp_value_fn

    def _compute_hilp_values(
        self, 
        obs: Union[np.ndarray, torch.Tensor], 
        goal: Union[np.ndarray, torch.Tensor], 
        use_no_grad: bool = True
    ) -> torch.Tensor:
        """
        Unified helper to compute pessimistic HILP values (min(v1, v2)).
        STRICT: Only supports matching shapes (N, D) or (D,).
        
        Args:
            obs: (N, D) or (D,)
            goal: (N, D) or (D,)
            use_no_grad: Whether to use torch.no_grad()
            
        Returns:
            min_values: Tensor of pessimistic values, shape (N,).
        """
        hilp_value_fn = self._get_hilp_value_fn()
        
        # 1. Convert to torch and move to device
        def _to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float().to(self.device)
            return x.float().to(self.device)

        obs_t = _to_tensor(obs)
        goal_t = _to_tensor(goal)

        # 2. Add batch dimension if 1D
        if obs_t.ndim == 1: obs_t = obs_t.unsqueeze(0)
        if goal_t.ndim == 1: goal_t = goal_t.unsqueeze(0)

        # 3. STRICT SHAPE ASSERTION
        assert obs_t.shape == goal_t.shape, (
            f"[HILP Shape Error] obs and goal must have matching shapes. "
            f"Got obs: {obs_t.shape}, goal: {goal_t.shape}. "
            f"Broadcasting/Expansion must be handled by the caller (e.g. for guidance (T,B,D))."
        )
        assert obs_t.ndim == 2, f"[HILP Shape Error] Expected 2D tensors (N, D), got {obs_t.shape}"

        # 4. Padding/Cropping to self.hilp_obs_dim
        def _pad(x):
            if x.shape[-1] < self.hilp_obs_dim:
                padding = torch.zeros((*x.shape[:-1], self.hilp_obs_dim - x.shape[-1]), device=x.device)
                return torch.cat([x, padding], dim=-1)
            return x[..., :self.hilp_obs_dim]

        obs_t = _pad(obs_t)
        goal_t = _pad(goal_t)

        # 5. Compute values
        if use_no_grad:
            with torch.no_grad():
                v1, v2 = hilp_value_fn(obs_t, goal_t)
                res = torch.min(v1, v2)
        else:
            if hasattr(hilp_value_fn, 'value'):
                v1, v2 = hilp_value_fn.value(obs_t, goal_t)
            else:
                v1, v2 = hilp_value_fn(obs_t, goal_t)
            res = torch.min(v1, v2)
            
        return res

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

        self.log("training/loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)

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



    # DEPRECATED
    """
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
    """

    def process_segment_noise_levels(self, level_array: np.ndarray, sequence_dividing_factor: int, from_start: bool, reduction_amount: Optional[int] = None) -> np.ndarray:
        
        plan_tokens = len(level_array) # T
        assert plan_tokens % sequence_dividing_factor == 0, f"Plan tokens must be divisible by sequence dividing factor, but got {plan_tokens} and {sequence_dividing_factor}"
        segment_size = plan_tokens // sequence_dividing_factor
        
        # Work with a copy
        steps = [level_array.copy()]
        
        work_array = level_array.copy()
        if not from_start:
            work_array = np.flip(work_array)
        
        non_zero_indices = np.where(work_array > 0)[0]
        if len(non_zero_indices) == 0:
            return np.expand_dims(level_array, 0)
        
        start_idx = non_zero_indices[0]
        end_idx = min(start_idx + segment_size, plan_tokens)
        
        if self.pyramid:
            local_horizon = end_idx - start_idx
            uncertainty_scale = getattr(self, 'uncertainty_scale', 1) 
            
            initial_levels = steps[0][start_idx:end_idx]
            base_val = initial_levels[0]
            indices = np.arange(local_horizon)
            
            while np.any(work_array[start_idx:end_idx] > 0):
                current_step_count = len(steps)
                
                target_levels = base_val + indices * uncertainty_scale - current_step_count * reduction_amount
                target_levels = np.maximum(0, target_levels).astype(work_array.dtype)
                
                work_array[start_idx:end_idx] = np.minimum(work_array[start_idx:end_idx], target_levels)
                
                step_to_add = work_array.copy()
                if not from_start:
                    step_to_add = np.flip(step_to_add)
                steps.append(step_to_add)
                
        else:
            while np.any(work_array[start_idx:end_idx] > 0):
                work_array[start_idx:end_idx] = np.maximum(0, work_array[start_idx:end_idx] - reduction_amount)
                
                step_to_add = work_array.copy()
                if not from_start:
                    step_to_add = np.flip(step_to_add)
                steps.append(step_to_add)
        
        return np.stack(steps, axis=0) # (M, T)
    
    def _construct_noise_levels(
        self,
        levels: np.ndarray,
        batch_size: int,
        stabilization: int,
        pad_tokens: int,
        from_start: bool,
        include_final_token: bool,
        include_init_token: bool = True,
    ) -> torch.Tensor:
        """
        Construct noise levels for middle tokens, optional init/final tokens, and padding.

        Args:
            batch_size: Batch size
            levels: Noise levels for plan tokens (batch_size, plan_tokens)
            stabilization: Noise level for stabilized tokens (init and final)
            pad_tokens: Number of padding tokens
            include_final_token: Whether to include final_token (bidirectional mode)
            include_init_token: Whether to prepend init_token slot (legacy format).
                                Set False for pre-built sequences (new format).

        Returns:
            Noise levels array (batch_size, total_tokens)
        """
        if not from_start:
            levels = np.flip(levels, axis=1)

        components = []
        if include_init_token:
            components.append(np.full((batch_size, 1), stabilization, dtype=np.int64))  # init_token
        components.append(levels)  # plan tokens

        if include_final_token:
            components.append(np.full((batch_size, 1), stabilization, dtype=np.int64))  # final_token

        components.append(np.full((batch_size, pad_tokens), self.sampling_timesteps, dtype=np.int64))  # padding
        components = torch.from_numpy(np.concatenate(components, axis=1)).to(self.device)

        return rearrange(components, "b t -> t b", b=batch_size)
    

    # DEPRECATED: This function is no longer called inside parallel_plan.
    """
    def _construct_sequence(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        plan: list,
        plan_tokens: int,
        from_start: bool,
        reserve_final_token_space: bool,
    ) -> tuple:
        # Kept for reference only. New format pre-builds the full sequence in
        # _build_plan_from_leaf and cats plans directly in parallel_plan.
        # input plan.shape: b (t fs) 1 c

        chunk = []
        batch_size = len(plan) 
        for i in range(batch_size):
            if plan[i] == None:
                c = torch.randn((plan_tokens, 1, *self.x_stacked_shape), device=self.device)
                c = torch.clamp(c, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
            else:
                c = rearrange(plan[i] if from_start else torch.flip(plan[i], [0]), "(t fs) 1 c -> t 1 (fs c)", fs=self.frame_stack)
            chunk.append(c)
        chunk = torch.cat(chunk, 1) # (T,B, fs*c)
        if len(chunk.shape) == 2:
            chunk = chunk.unsqueeze(0)

        init_token_raw = self.pad_init(start)  # (fs, b, c)
        final_token_raw = self.pad_init(goal, is_start=False)  # (fs, b, c)
        
            # 1 for init_token, 1 for final_token
            
        if from_start:
            processed_init_token = rearrange(init_token_raw, "fs b c -> 1 b (fs c)")
            processed_final_token = rearrange(final_token_raw, "fs b c -> 1 b (fs c)")
            # Normal order: [init(start), chunk, final(goal), pad]
        else:
            # Flip along frame_stack dimension
            init_token_flipped_raw = torch.flip(init_token_raw, [0])  # (fs, b, c)
            final_token_flipped_raw = torch.flip(final_token_raw, [0])  # (fs, b, c)
            
            # Now rearrange
            processed_init_token = rearrange(final_token_flipped_raw, "fs b c -> 1 b (fs c)")
            processed_final_token = rearrange(init_token_flipped_raw, "fs b c -> 1 b (fs c)")
                
        if reserve_final_token_space:
            pad_tokens = max(0, self.n_tokens - plan_tokens - 2)
            pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
            plan_with_given_tokens = torch.cat([processed_init_token, chunk, processed_final_token, pad], 0)
        else:
            pad_tokens = max(0, self.n_tokens - plan_tokens - 1)
            pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
            plan_with_given_tokens = torch.cat([processed_init_token, chunk, pad], 0)




        return plan_with_given_tokens # t b (fs c)
 
    """



    def _generate_bidirectional_schedule(
        self, 
        start_levels: np.ndarray, 
        complete_denoising: bool = False, 
        from_start: bool = True
    ) -> np.ndarray:
        """
        Generates the N-step denoising schedule for bidirectional search.
        Returns a tensor of shape (Batch, Steps, Tokens) representing the sequence of noise levels.
        """
        # start_levels shape: (B, plan_tokens)

        batch_size = start_levels.shape[0]
        current_levels = start_levels.copy()
        schedule = [current_levels.copy()]
        
        assert self.sampling_timesteps >= self.mctd_num_denoising_steps, "sampling_timesteps must be greater than or equal to mctd_num_denoising_steps"
        chunk_of_sampling_timesteps_for_one_denoising = self.sampling_timesteps // self.mctd_num_denoising_steps

        while True:
            # Process each batch to denoise ONE segment
            to_levels_list = []
            for b in range(batch_size):
                to_levels_b = self.process_segment_noise_levels(
                    current_levels[b], self.sequence_dividing_factor, from_start, reduction_amount=chunk_of_sampling_timesteps_for_one_denoising
                )
                to_levels_list.append(to_levels_b)
            
            # Verify that all particles in the batch have the same number of steps (M)
            # assert all(len(steps) == len(to_levels_list[0]) for steps in to_levels_list), \
            #     f"Schedules in batch have inconsistent lengths: {[len(s) for s in to_levels_list]}"
                
                
            # Determine the maximum number of steps (M) in this segment across the batch
            max_m = max(len(steps) for steps in to_levels_list)
            
            # Pad schedules to max_m by repeating the last step
            for b in range(batch_size):
                if len(to_levels_list[b]) < max_m:
                    padding = np.tile(to_levels_list[b][-1:], (max_m - len(to_levels_list[b]), 1))
                    to_levels_list[b] = np.concatenate([to_levels_list[b], padding], axis=0)


            batch_steps = np.stack(to_levels_list, axis=1) # (M, B, T)
            
            # Append subsequent steps (index 0 is current_levels which is already in schedule)
            for m in range(1, batch_steps.shape[0]):
                schedule.append(batch_steps[m].copy())
            
            current_levels = batch_steps[-1]

            if np.all(current_levels == 0):
                break
            
            if not complete_denoising:
                break
        
        return np.stack(schedule, axis=0).transpose(1, 0, 2) # (B, TotalSteps, T)




    def parallel_plan(
        self, 
        start: torch.Tensor, 
        goal: torch.Tensor, 
        horizon: int, 
        conditions: Optional[Any] = None,
        guidance_scale: Optional[int] = None, 
        noise_level: Optional[np.ndarray] = None, 
        plans: Optional[list] = None, 
        from_start: bool = True, 
        is_unknown_final_token: bool = False
    ):


        horizon = int(horizon)
        # start and goal are numpy arrays of shape (b, obs_dim)

        batch_size = len(plans)
        if start.dim() == 2 and start.shape[0] == batch_size:
            pass
        else:
            start = torch.cat([start] * batch_size, 0)
        
        if goal.dim() == 2 and goal.shape[0] == batch_size:
            pass
        else:
            goal = torch.cat([goal] * batch_size, 0)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        def weigheted_loss(dist: torch.Tensor, weight: Optional[torch.Tensor] = None, dim: tuple = (0, 2)) -> torch.Tensor:
            """Helper function to compute weighted loss from distance tensor."""
            dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
            # dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
            dist_o = dist_o[:, :, : 2]
            dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=1)
            dist_o = (dist_o + 1e-6).sqrt()
            # dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
            dist = dist_o
            if weight is None:
                weight = torch.ones_like(dist)
            else:
                assert len(weight.shape) == 1, f"weight shape {weight.shape} is not 1D"
                weight = repeat(weight, "t -> t n", n=dist.shape[-1])
                weight = torch.ones_like(dist) * weight[:, None] #  t b n
            return (dist * weight).mean(dim=dim) # * dist.shape[1] DO NOT DELETE THIS COMMENT

        def _prepare_pred(x: torch.Tensor) -> torch.Tensor:
            """Helper to rearrange and unnormalize predictions."""
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            return self._unnormalize_x(pred)

        hilp_value_fn = self._get_hilp_value_fn() if self.use_hilp_guidance else None

        def goal_guidance(x: torch.Tensor) -> torch.Tensor:
            """Target guidance to reach goal/start."""
            # print(f"\n[GUIDANCE DEBUG] goal_guidance called with x.shape={x.shape}")
            # print(f"[GUIDANCE DEBUG] guidance_scale={guidance_scale}")
            # x is a tensor of shape [t b (fs c)]
            pred = _prepare_pred(x)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # Temporal consistency guidance via shifted predictions
                # pred: (t fs) b c
                
                target = goal if from_start else start
                
                # Unnormalize target to match the scale of pred (which is unnormalized above)
                target = self._unnormalize_x(target)
                
                target_guidance = torch.stack([target] * pred.shape[0])
                
                # Compute distance: either HILP value function or MSE
                if self.use_hilp_guidance:
                    T, B = pred.shape[0], pred.shape[1]
                    # Shape strictness: Flatten (T, B, 2) and expand (B, 2) to match
                    obs_flat = pred[:, :, :2].reshape(T * B, 2)
                    goal_expanded = target[:, :2].unsqueeze(0).expand(T, -1, -1).reshape(T * B, 2)
                    
                    v_flat = self._compute_hilp_values(obs_flat, goal_expanded, use_no_grad=False)
                    v = v_flat.reshape(T, B)  # (T, B)
                    
                    # Convert value to distance: negate since v is negative distance
                    dist_values = -v  # (T, B)
                    dist_target_hilp = dist_values.unsqueeze(-1)  # (T, B, 1)
                    
                    # Replicate to match MSE output shape (T, B, C)
                    dist_target = dist_target_hilp.expand(-1, -1, pred.shape[-1])
                    
                    # print(f"[HILP DEBUG] Using HILP value function with zero-padding")
                    # print(f"[HILP DEBUG] pred_xy shape: {pred_xy.shape}, zeros_rest shape: {zeros_rest.shape}")
                    # print(f"[HILP DEBUG] pred_obs shape: {pred_obs.shape}, target_obs shape: {target_obs.shape}")
                    # print(f"[HILP DEBUG] v1 range: [{v1.min():.4f}, {v1.max():.4f}], v2 range: [{v2.min():.4f}, {v2.max():.4f}]")
                    # print(f"[HILP DEBUG] dist_target shape: {dist_target.shape}, range: [{dist_target.min():.4f}, {dist_target.max():.4f}]")
                else:
                    # Use traditional MSE distance
                    dist_target = nn.functional.mse_loss(pred, target_guidance, reduction="none")

                target_weight = np.array(
                    [0] * (self.frame_stack)  # conditoning (aka reconstruction guidance)

                    + [1 for _ in range(horizon)]  # try to reach the goal at any horizon
                    # + [0 for _ in range(horizon-1)] + [1]  # Diffuer guidance
                    + [0] * (h_padded - horizon)  # don't guide padded entries due to horizon % frame_stack != 0
                )
                target_weight = torch.from_numpy(target_weight).float().to(self.device)
                weighted_dist_target = weigheted_loss(dist_target, target_weight)

                dist_per_batch = guidance_scale * weighted_dist_target

                # Specifically for dist_left, the last token is the most important
                # dist is (t fs) b n
                last_token_dist = weigheted_loss(dist_target, weight=None, dim=-1)[-1]
                
                print(f"[SCALE IMPACT] Dist per batch: {dist_per_batch.tolist()}")
                print(f"[SCALE IMPACT] Final token dist: {last_token_dist.tolist()}")
                print(f"[SCALE IMPACT] Scales: {guidance_scale.tolist()}")
                
            else:
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")

            return -(dist_per_batch).mean()

        def anchor_dist_guidance(x: torch.Tensor) -> torch.Tensor:
            """Anchor distance regularization using segment heads."""
            # x is a tensor of shape [t b (fs c)]
            pred = _prepare_pred(x)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            pred_detached = pred.detach()
            
            segment_size = horizon // self.sequence_dividing_factor
            head_of_each_segments = pred_detached[self.frame_stack-1 : self.frame_stack + horizon - 1 : segment_size]
            anchor_plan = torch.zeros_like(pred_detached)
            anchor_plan[self.frame_stack:self.frame_stack + horizon:segment_size] = head_of_each_segments
            dist_anchor = nn.functional.mse_loss(pred, anchor_plan, reduction="none")
            
            anchor_weight = torch.zeros_like(pred_detached[:, 0, 0])
            anchor_weight[self.frame_stack:self.frame_stack + horizon:segment_size] = 1
            weighted_dist_anchor = weigheted_loss(dist_anchor, anchor_weight)
            
            return -(weighted_dist_anchor).mean()


        def segment_rdf_guidance(x: torch.Tensor) -> torch.Tensor:
            """
            Temporal consistency guidance using RDF kernel with a sliding window.
            Repels current state from states in the window [idx-7-segment_size, idx-7].
            """
            # x is a tensor of shape [t b (fs c)]
            pred = _prepare_pred(x)
            total_T = pred.shape[0]
            
            # Extract observation part (first 2 dimensions for position)
            pred_obs = pred[:, :, :2]  # Shape: [T, B, 2]
            
            # Calculate segment size for window width
            # segment_size = horizon // self.sequence_dividing_factor
            segment_size = float('inf')
            
            # Create indices for pairwise comparison
            indices = torch.arange(total_T, device=x.device)
            j_idx = indices.view(-1, 1)
            k_idx = indices.view(1, -1)
            
            # Sliding window mask: k is between [j-7-segment_size, j-7]
            ignore_latest = 5 * self.frame_stack
            pair_mask = (k_idx <= j_idx - ignore_latest) & (k_idx >= j_idx - ignore_latest - segment_size)
            
            # Only apply to states within the planning horizon (after conditioning frames)
            planning_mask = (j_idx >= self.frame_stack) & (j_idx < self.frame_stack + horizon)
            pair_mask = pair_mask & planning_mask
            
            if not pair_mask.any():
                return torch.tensor(0.0, device=x.device, requires_grad=True)
            
            # Pairwise squared distances [B, T, T]
            pred_obs_b = pred_obs.transpose(0, 1) # [B, T, 2]
            dist_sq = torch.cdist(pred_obs_b, pred_obs_b, p=2).pow(2)
            
            # RDF kernel matrix [B, T, T]
            h = 2.0 # bandwidth
            rdf_matrix = torch.exp(-dist_sq / h)
            
            # Apply mask: set invalid pairs to 0
            masked_rdf = rdf_matrix * pair_mask.unsqueeze(0).float()
            
            # For each j (dim 1), find mean of top 3 RDF among valid k's (dim 2)
            topk_rdf, _ = torch.topk(masked_rdf, k=3, dim=2)
            topk_rdf_mean_per_j = topk_rdf.mean(dim=2)
            
            # Average over j's that have at least one valid candidate k
            j_has_candidates = pair_mask.any(dim=1)
            if not j_has_candidates.any():
                return torch.tensor(0.0, device=x.device, requires_grad=True)
                
            mean_loss = topk_rdf_mean_per_j[:, j_has_candidates].sum() / 1000
            
            # Return negative loss (gradient descent will minimize repulsion)
            return -mean_loss

        def particle_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            # Implementation of Particle Guidance (PG) from "Tree-Guided Diffusion Planner (TDP)"
            # This function computes a diversity score based on an RBF kernel to repel particles from each other.
            b = x.shape[1]
            if b <= 1:
                return x.sum() * 0.0

            x_flat = rearrange(x, "t b (fs c) -> b (t fs c)", fs=self.frame_stack)
            
            # Shape: [b, b]
            dist_sq = torch.cdist(x_flat, x_flat, p=2).pow(2)
            
            h = torch.median(dist_sq.detach())
            if h == 0: 
                h = 1.0 # Fallback to avoid division by zero
            
            kernel_matrix = torch.exp(-dist_sq / h)
            
            similarity = (kernel_matrix.sum() - b) / (b * (b - 1))
            
            return -similarity


        def combined_guidance(x_start: torch.Tensor) -> dict:
            return {
                "anchor":anchor_dist_guidance(x_start) * self.anchor_guidance_scale,
                "goal": guidance_scale * goal_guidance(x_start),
                "rdf": segment_rdf_guidance(x_start) * self.rdf_guidance_scale
            }
        
        guidance_fn = combined_guidance




        assert horizon % self.frame_stack == 0, "horizon must be a multiple of frame_stack"

        plan_tokens = horizon // self.frame_stack

        # Detect if plans are pre-built (new format: each plan is (n_tokens, 1, fs*c)).
        # _build_plan_from_leaf returns this format for bidirectional search.
        # Legacy format: plans is a list of None or (plan_tokens*fs, 1, c) tensors.
        is_prebuilt = (
            plans is not None
            and len(plans) == batch_size
            and all(
                isinstance(p, torch.Tensor) and p.ndim == 3 and p.shape[0] == self.n_tokens
                for p in plans
            )
        )

        if is_prebuilt:
            # ---- New format: plans are fully pre-built (n_tokens, 1, fs*c) ----
            assert self.n_tokens >= plan_tokens, f"too long horizon (n_tokens={self.n_tokens} < plan_tokens={plan_tokens})"
            pad_tokens = max(0, self.n_tokens - plan_tokens)
            plan_start_offset = 0      # no init_token slot
            use_init_token_in_noise = False
            use_final_token_in_noise = False

            plan_with_given_tokens = torch.cat(plans, dim=1)  # (n_tokens, b, fs*c)

        else:
            # ---- Legacy format: inline equivalent of deprecated _construct_sequence ----
            use_bidirectional_sequence = self.bidirectional_search and not is_unknown_final_token
            assert self.n_tokens - plan_tokens >= (1 if is_unknown_final_token else 2), \
                f"too long horizon (n_tokens - plan_tokens < {1 if is_unknown_final_token else 2})"
            pad_tokens = max(0, self.n_tokens - plan_tokens - 2) if use_bidirectional_sequence else max(0, self.n_tokens - plan_tokens - 1)
            plan_start_offset = 1      # init_token occupies position 0
            use_init_token_in_noise = True
            use_final_token_in_noise = use_bidirectional_sequence

            chunk = []
            for i in range(batch_size):
                if plans is None or plans[i] is None:
                    c = torch.randn((plan_tokens, 1, *self.x_stacked_shape), device=self.device)
                    c = torch.clamp(c, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
                else:
                    c = rearrange(
                        plans[i] if from_start else torch.flip(plans[i], [0]),
                        "(t fs) 1 c -> t 1 (fs c)", fs=self.frame_stack,
                    )
                chunk.append(c)
            chunk = torch.cat(chunk, 1)  # (plan_tokens, b, fs*c)

            init_token_raw = self.pad_init(start)       # (fs, b, c)
            final_token_raw = self.pad_init(goal, is_start=False)  # (fs, b, c)
            if from_start:
                processed_init_token = rearrange(init_token_raw, "fs b c -> 1 b (fs c)")
                processed_final_token = rearrange(final_token_raw, "fs b c -> 1 b (fs c)")
            else:
                init_token_flipped_raw = torch.flip(init_token_raw, [0])
                final_token_flipped_raw = torch.flip(final_token_raw, [0])
                processed_init_token = rearrange(final_token_flipped_raw, "fs b c -> 1 b (fs c)")
                processed_final_token = rearrange(init_token_flipped_raw, "fs b c -> 1 b (fs c)")

            pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
            if use_bidirectional_sequence:
                plan_with_given_tokens = torch.cat([processed_init_token, chunk, processed_final_token, pad], 0)
            else:
                plan_with_given_tokens = torch.cat([processed_init_token, chunk, pad], 0)

        # input plan.shape: b (t fs) 1 c  (legacy) or pre-built (n_tokens, 1, fs*c)
        # output plan_hist.shape: m (t fs) b c

        def flip_plan_for_insert_hist(processed_plan_for_diffusion, plan_tokens, from_start, fs, offset):
            chunk_tokens = processed_plan_for_diffusion[offset : offset + plan_tokens].detach().clone()
            if not from_start:
                chunk_tokens = torch.flip(rearrange(chunk_tokens, "t b (fs c) -> (t fs) b c", fs=fs), [0])
                chunk_tokens = rearrange(chunk_tokens, "(t fs) b c -> t b (fs c)", fs=fs)
            return chunk_tokens

        plan_hist = [flip_plan_for_insert_hist(plan_with_given_tokens, plan_tokens, from_start, self.frame_stack, plan_start_offset)]

        stabilization = 0

        for m in range(noise_level.shape[1] - 1):
            # noise_level.shape: b, m, plan_tokens(=t)
            from_noise_levels = self._construct_noise_levels(
                noise_level[:, m], batch_size, stabilization, pad_tokens,
                from_start, use_final_token_in_noise, include_init_token=use_init_token_in_noise,
            )
            to_noise_levels = self._construct_noise_levels(
                noise_level[:, m + 1], batch_size, stabilization, pad_tokens,
                from_start, use_final_token_in_noise, include_init_token=use_init_token_in_noise,
            )

            sample = self.diffusion_model.sample_step(
                plan_with_given_tokens, conditions, from_noise_levels, to_noise_levels, guidance_fn=guidance_fn
            )  # (n_tokens, b, fs*c)

            if is_prebuilt:
                # Update only tokens whose noise level is actively decreasing this step.
                # This preserves denoised_prefix (level=0) and obs_parent_token (level=0).
                update_mask = (from_noise_levels > to_noise_levels).unsqueeze(-1)  # (n_tokens, b, 1)
                plan_with_given_tokens = torch.where(update_mask, sample, plan_with_given_tokens)
            else:
                plan_with_given_tokens[plan_start_offset: plan_start_offset + plan_tokens] = \
                    sample[plan_start_offset: plan_start_offset + plan_tokens]

            plan_hist.append(flip_plan_for_insert_hist(plan_with_given_tokens, plan_tokens, from_start, self.frame_stack, plan_start_offset))

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack)

        return plan_hist  # m (t fs) b c


    def interact(
        self, 
        batch_size: int, 
        conditions: Optional[Any] = None, 
        namespace: str = "validation"
    ) -> None:
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
                if self.dataset == "antmaze-medium-navigate-v0" or self.dataset == "antmaze-medium-stitch-v0":
                    dql_folder = "antmaze-medium-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                elif self.dataset == "antmaze-large-navigate-v0" or self.dataset == "antmaze-large-stitch-v0":
                    dql_folder = "antmaze-large-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                elif self.dataset == "antmaze-giant-navigate-v0" or self.dataset == "antmaze-giant-stitch-v0":
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

        # ----------------------------------------------------------------
        # Bidirectional MCTS: initialize tree1/tree2 once before MPC loop.
        # These trees are maintained across MPC steps and expanded
        # alternately within each planning call.
        # ----------------------------------------------------------------
        horizon: int = int(self.episode_len * self.horizon_scale)
        if self.bidirectional_search and self.mctd:
            _bidir_start_np = start.cpu().numpy()[:, :self.observation_dim]  # (1, obs_dim)
            _bidir_goal_np  = goal.cpu().numpy()[:, :self.observation_dim]   # (1, obs_dim)
            # Capture initial physical state if available
            initial_sim_state = self._get_sim_state(envs)

            # Derive heuristic goal simulation state from initial state
            goal_sim_state = {
                "qpos": initial_sim_state["qpos"].copy(),
                "qvel": np.zeros_like(initial_sim_state["qvel"])  # Goal is assumed static
            }
            # Replace x, y coordinates with goal coordinates
            goal_sim_state["qpos"][:2] = _bidir_goal_np[0][:2]

            bidir_tree1 = self._init_mcts_tree(
                horizon, from_start=True,
                tag="bidir_mcts_from_start",
                root_obs=_bidir_start_np[0],
                root_sim_state=initial_sim_state,
            )
            bidir_tree2 = self._init_mcts_tree(
                horizon, from_start=True,
                tag="bidir_mcts_from_goal",
                root_obs=_bidir_goal_np[0],
                root_sim_state=goal_sim_state,
            )
            # Flag: 0 → expand tree1 next, 1 → expand tree2 next
            expanded_tree_idx: int = 0
            # Configurable meeting threshold (Euclidean distance in unnormalized obs space)
            _meeting_delta: float = getattr(self.cfg.planning, "meeting_delta", 2.0)
        else:
            bidir_tree1 = bidir_tree2 = None

        while not terminate and steps < self.val_max_steps:
            planning_start_time = time.time()
            
            # Generate plan (start → goal)
            # _generate_plan_between_points has been inlined here.
            
            if self.mctd and self.bidirectional_search:
                # ------------------------------------------------------------------
                # Bidirectional alternating MCTS planning
                # ------------------------------------------------------------------
                _start_np = start.cpu().numpy()[:, :self.observation_dim]
                _goal_np  = goal.cpu().numpy()[:, :self.observation_dim]

                # Collect opposite tree leaf nodes for dynamic goal selection and plan extraction
                def _get_leaf_nodes(root_node: "TreeNode") -> List["TreeNode"]:
                    leaves: List["TreeNode"] = []
                    stack = [root_node]
                    while stack:
                        n = stack.pop()
                        is_leaf = all(c["node"] is None for c in n._children_nodes)
                        if is_leaf:
                            leaves.append(n)
                        else:
                            for c in n._children_nodes:
                                if c["node"] is not None:
                                    stack.append(c["node"])
                    return leaves

                t1_leaf_nodes: List["TreeNode"] = _get_leaf_nodes(bidir_tree1.root_node)
                t2_leaf_nodes: List["TreeNode"] = _get_leaf_nodes(bidir_tree2.root_node)
                
                # (leaf node lists are passed directly to _run_mcts_search as opposite_leaf_nodes)

                # Use flag to decide whether to use simulation in search
                _use_sim = getattr(self, "mcts_use_sim", True)

                # Initialize infos dicts so {**infos1, **infos2} is safe even on the first step
                expanded_node_infos1: Dict[str, dict] = {}
                expanded_node_infos2: Dict[str, dict] = {}

                # Alternate expansion: one single_step per MPC iteration
                if expanded_tree_idx == 0:
                    bidir_tree1, expanded_node_infos1 = self._run_mcts_search(
                        bidir_tree1, horizon, conditions,
                        _start_np, _goal_np,
                        opposite_leaf_nodes=t2_leaf_nodes,
                        single_step=True,
                        use_simulation=_use_sim,
                    )
                    expanded_node_infos = expanded_node_infos1
                else:
                    bidir_tree2, expanded_node_infos2 = self._run_mcts_search(
                        bidir_tree2, horizon, conditions,
                        _goal_np, _start_np,
                        opposite_leaf_nodes=t1_leaf_nodes,
                        single_step=True,
                        use_simulation=_use_sim,
                    )
                    expanded_node_infos = expanded_node_infos2

                if self.debug:
                    print(f"[DEBUG] [Step {step}] Bidir Turn: {'Tree1 (Forward)' if expanded_tree_idx == 0 else 'Tree2 (Backward)'}")
                
                # Per-leaf MPC rollout: update obs_pos and sim_state for newly expanded leaves
                # Backup current physical state to avoid cross-leaf contamination
                # _original_sim_state = self._get_sim_state(envs)

                active_tree = bidir_tree1 if expanded_tree_idx == 0 else bidir_tree2

                for info in expanded_node_infos.values():
                    parent_node: "TreeNode" = info["parent_node"]
                    _child: Optional["TreeNode"] = info.get("node")  # set by expand()
                    if _child is None:
                        continue

                    # Recompute plan tensor and denoised index range from stored plan_history
                    plan_hist_last: torch.Tensor = info["plan_history"][-1][-1]  # (T_total*fs, c)
                    plan: torch.Tensor = self._unnormalize_x(plan_hist_last.unsqueeze(1))  # (T_fs, 1, c)

                    seg_size: int = active_tree.plan_tokens // self.sequence_dividing_factor
                    prefix_len = parent_node.depth * seg_size
                    # +1 accounts for obs_parent_token
                    new_denoised_start: int = prefix_len + 1
                    new_denoised_end: int = prefix_len + 1 + seg_size

                    _last_obs, _new_sim_state = self._rollout_leaf_plan(
                        leaf_plan_tokens=plan,
                        new_denoised_start_idx=new_denoised_start,
                        new_denoised_end_idx=new_denoised_end,
                        agent=agent,
                        envs=envs,
                        parent_sim_state=parent_node.sim_state,
                    )
                    _child.sim_state = _new_sim_state
                    # Update obs_pos from reached sim_state if possible, else fallback to observation
                    _child.obs_pos = _new_sim_state["qpos"][:2] if _new_sim_state else _last_obs[:2]

                    if self.debug:
                        print(f"  [DEBUG] Expanded Leaf '{_child.name}' updated with sim_state. Pos: {_child.obs_pos}")

                # Restore original environment state after all individual rollouts
                # if _original_sim_state is not None:
                #     self._set_sim_state(envs, _original_sim_state)

                # Extract plan by selecting best leaf and combining plans
                best_info: dict = self._select_best_leaf(expanded_node_infos)
                best_node: "TreeNode" = best_info["node"]
                output_plan = self._extract_output_plan(best_node, plan_tokens=active_tree.plan_tokens, is_tree1=(expanded_tree_idx == 0))
                
                plan_hist = output_plan.unsqueeze(0)  # (1, T, 1, c)
                plan_hist = self._unnormalize_x(plan_hist)
                plan = plan_hist[-1]  # (T, 1, c)

                # Flip for the next MPC step to alternate trees
                expanded_tree_idx = (expanded_tree_idx + 1) % 2

            elif self.mctd:
                # --- Unidirectional MCTS (flag=False, original behavior) ---
                plan_hist = self.p_mctd_plan(
                    obs_normalized, goal_normalized,
                    horizon, conditions,
                    start.cpu().numpy()[:, :self.observation_dim],
                    goal.cpu().numpy()[:, :self.observation_dim],
                    tag="mcts_plan",
                    from_start=True,
                )
                plan_hist = self._unnormalize_x(plan_hist)
                plan = plan_hist[-1]  # (t b c)

            # Visualization with both forward and reverse trajectories
            start_numpy = start.cpu().numpy()[:, :2]
            goal_numpy = goal.cpu().numpy()[:, : 2]
            
            # Create forward trajectory image
            forward_image = make_trajectory_images(
                self.env_id, 
                plan[:, :, :2].detach().cpu().numpy(), 
                1, start_numpy, goal_numpy, 
                self.plot_end_points
            )[0]
            self.log_image(f"plan/plan_at_{steps}_from_start", Image.fromarray(forward_image))

            # Create reverse trajectory image (swap start and goal for visualization)
            """
            reverse_image = make_trajectory_images(
                self.env_id, 
                reverse_plan[:, :, :2].detach().cpu().numpy(), 
                1, start_numpy, goal_numpy, 
                self.plot_end_points
            )[0]
            self.log_image(f"plan/plan_at_{steps}_from_goal", Image.fromarray(reverse_image))
            """
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
                sub_goal_idx = min(self.sub_goal_interval, plan.shape[0] - 1)
                sub_goal = plan[sub_goal_idx, :, :2].detach().cpu().numpy()
                sub_goal_step = sub_goal_idx
            
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

    def pad_init(self, x, is_start=True, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            if is_start:
                x[: self.frame_stack - 1] = 0
            else:
                x[1:] = 0
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

    def visualize_node_value_plans(self, search_num, values, names, plans, starts, goals, tag="mcts_plan"):
        # plans: (t fs) b c 

        if plans.shape[1] != starts.shape[0]:
            starts = starts.repeat(plans.shape[1], axis=0) # b
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0) # b
        
        plans = self._unnormalize_x(plans)
        plan_obs, _, _ = self.split_bundle(plans)  # (t fs) b c -> [(t fs) b c1, (t fs) b c2, (t fs) b c3]
        plan_obs = plan_obs.detach().cpu().numpy()# [:-1]
        
        if plan_obs.ndim == 2:
            # (t fs) c -> (t fs) 1 c
            plan_obs = plan_obs[:, None, :]

        plan_images = make_trajectory_images(self.env_id, plan_obs, plan_obs.shape[1], starts, goals, self.plot_end_points)
        for i in range(len(plan_images)):
            img = plan_images[i]
            self.log_image(
                # f"{tag}/{search_num+i+1}_{names[i]}_V{values[i]}",
                f"{tag}/{names[i]}_V{values[i]}",
                Image.fromarray(img),
            )

    def calculate_values(self, plans, starts, goals, from_start):
        # plans: (t fs) b c

        if not from_start:
            starts, goals = goals, starts
            plans = torch.flip(plans.clone(), [0])

        if plans.shape[1] != starts.shape[0]: # b
            starts = starts.repeat(plans.shape[1], axis=0) # (b, c1)
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0)

        plans = self._unnormalize_x(plans)
        obs, _, _ = self.split_bundle(plans) # (t fs) b c -> [(t fs) b c1, (t fs) b c2, (t fs) b c3]
        obs = obs.detach().cpu().numpy()#[:-1, :]  # last observation is dummy
        values = np.zeros(plans.shape[1])
        infos = np.array(["NotReached"] * plans.shape[1]) # B
        achieved_ts = np.array([None] * plans.shape[1])
        for t in range(obs.shape[0]): # (t fs)
            if t == 0:
                pos_diff = np.linalg.norm(obs[t] - starts, axis=-1) # b c1 -> b
            else:
                pos_diff = np.linalg.norm(obs[t] - obs[t-1], axis=-1)
            infos[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = "Warp" # batch-wise indexing
            values[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = 0
            diff_from_goal = np.linalg.norm(obs[t] - goals, axis=-1)
            values[(diff_from_goal < self.meeting_delta) * (infos == "NotReached")] = (plans.shape[0] - t) / plans.shape[0]
            achieved_ts[(diff_from_goal < self.meeting_delta) * (infos == "NotReached")] = t
            infos[(diff_from_goal < self.meeting_delta) * (infos == "NotReached")] = "Achieved"

        return values, infos, achieved_ts


    def calculate_values_bidir(
        self,
        expanded_node_candidates: List[dict],
        expanded_plan_hists: torch.Tensor,
        tree: "MCTSTreeState",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-node values for bidirectional search by pairing current plan with
        the target opposite-tree leaf node's plan.

        For each expanded candidate i:
          1. Slice plan_A from expanded_plan_hists (current tree, depth-based length).
          2. Slice plan_B from target_node.plan_history (opposite tree, depth-based length).
          3. Flip plan_B and concatenate: [plan_A_sliced | flip(plan_B_sliced)].
          4. Delegate Warp/Achieved detection to calculate_values.

        Args:
            expanded_node_candidates: List of candidate dicts (each has 'parent_node').
            expanded_plan_hists: Tensor of shape (m, T_total*fs, B, c) — fully denoised plan hists.
            tree: MCTSTreeState for the current tree (provides plan_tokens, sequence_dividing_factor).

        Returns:
            values: np.ndarray shape (B,)
            infos:  np.ndarray shape (B,), dtype str
            achieved_ts: np.ndarray shape (B,)
        """
        seg_size: int = tree.plan_tokens // self.sequence_dividing_factor
        B: int = len(expanded_node_candidates)

        values: np.ndarray = np.zeros(B)
        infos: np.ndarray = np.array(["NotReached"] * B)
        achieved_ts: np.ndarray = np.array([None] * B)

        for i, candidate in enumerate(expanded_node_candidates):
            parent_node: "TreeNode" = candidate["parent_node"]
            target_node: Optional["TreeNode"] = parent_node.target_node

            assert target_node is not None, (
                f"[BiDir Value] parent_node '{parent_node.name}' has no target_node. "
                "target_node must be set by _select_dynamic_goal() before value calculation."
            )
            assert target_node.plan_history, (
                f"[BiDir Value] target_node '{target_node.name}' has empty plan_history."
            )

            # --- Plan A: current tree's denoised plan, sliced to parent depth --- #
            # expanded_plan_hists: (m, T_total*fs, B, c) -> last denoising step for candidate i
            plan_a_full: torch.Tensor = expanded_plan_hists[-1, :, i]  # (T_total*fs, c)
            a_len: int = parent_node.depth * seg_size * self.frame_stack
            plan_a_sliced: torch.Tensor = plan_a_full[:a_len]  # (A_len, c)

            # --- Plan B: opposite tree's last denoised plan, sliced to target depth --- #
            plan_b_full: torch.Tensor = target_node.plan_history[-1][-1]  # (T_total*fs, c)
            b_len: int = target_node.depth * seg_size * self.frame_stack
            plan_b_sliced: torch.Tensor = plan_b_full[:b_len]  # (B_len, c)
            # Flip plan_B so it runs from target_node -> root of opposite tree
            plan_b_flipped: torch.Tensor = torch.flip(plan_b_sliced, [0])

            # --- Concat and add batch dim -> (T_combined, 1, c) --- #
            combined: torch.Tensor = torch.cat([plan_a_sliced, plan_b_flipped], dim=0).unsqueeze(1)

            # --- Delegate Warp/Achieved detection to calculate_values --- #
            # start = parent_node's physical position, goal = target_node's physical position
            start_np: np.ndarray = parent_node.obs_pos[None, :self.observation_dim]  # (1, obs_dim)
            goal_np: np.ndarray = target_node.obs_pos[None, :self.observation_dim]   # (1, obs_dim)
            # Always evaluate as forward (plan_A is forward, plan_B already flipped)
            _vals, _infos, _achieved_ts = self.calculate_values(
                combined, start_np, goal_np, from_start=True
            )
            values[i] = _vals[0]
            infos[i] = _infos[0]
            achieved_ts[i] = _achieved_ts[0]

        return values, infos, achieved_ts

    def _init_mcts_tree(
        self,
        horizon: int,
        from_start: bool,
        tag: str,
        root_obs: Optional[np.ndarray] = None,
        root_sim_state: Optional[dict] = None,
    ) -> MCTSTreeState:
        """
        (A function) Initialize a single MCTS tree and return its full state.

        Args:
            horizon: Planning horizon (must be divisible by frame_stack)
            from_start: Direction flag (True: start→goal, False: goal→start)
            tag: Tag string for tqdm progress bar labeling
            root_obs: Unnormalized root observation, shape (obs_dim,).
                      For tree1 (from_start=True) pass start coords;
                      for tree2 (from_start=False) pass goal coords.
                      Stored in root_node.obs_pos and tree.tree_root_obs.

        Returns:
            MCTSTreeState: Fully initialized tree state ready for _run_mcts_search
        """
        plan_tokens: int = horizon // self.frame_stack
        children_node_guidance_scales: list = self.mctd_guidance_scales
        max_search_num: int = self.mctd_max_search_num
        num_denoising_steps: int = self.mctd_num_denoising_steps
        skip_level_steps: int = self.mctd_skip_level_steps

        if self.bidirectional_search:
            assert plan_tokens <= self.n_tokens - (1 if self.is_unknown_final_token else 2), \
                f"Plan tokens must be <= {self.n_tokens - (1 if self.is_unknown_final_token else 2)}, but got {plan_tokens}"
            H: int = self.sequence_dividing_factor
            terminal_depth: int = H
            noise_level: Optional[np.ndarray] = None  # bidirectional uses dynamic schedule
        else:
            # Unidirectional mode: use original scheduling matrix
            noise_level = self._generate_scheduling_matrix(plan_tokens)
            terminal_depth = int(np.ceil((noise_level.shape[0] - 1) / num_denoising_steps))

        # Root Node initialization
        # Initialize root's current_levels for bidirectional search
        if self.bidirectional_search:
            # current_levels only contains middle tokens (init/final tokens handled separately)
            root_current_levels: Optional[np.ndarray] = np.full((1, plan_tokens), self.sampling_timesteps, dtype=np.int64)
        else:
            root_current_levels = None

        root_node = TreeNode(
            '0', 0, None, children_node_guidance_scales, [],
            terminal_depth=terminal_depth, virtual_visit_weight=self.virtual_visit_weight,
            current_levels=root_current_levels,
            obs_pos=root_obs,  # Store root actual position for bidirectional meeting detection
            sim_state=root_sim_state,
        )
        root_node.set_value(0)  # Initialize the value of the root node

        pbar = tqdm(total=max_search_num, desc=f"MCTS Search ({tag})", leave=False, dynamic_ncols=True)

        return MCTSTreeState(
            root_node=root_node,
            plan_tokens=plan_tokens,
            terminal_depth=terminal_depth,
            noise_level=noise_level,
            children_node_guidance_scales=children_node_guidance_scales,
            max_search_num=max_search_num,
            num_denoising_steps=num_denoising_steps,
            skip_level_steps=skip_level_steps,
            from_start=from_start,
            tag=tag,
            pbar=pbar,
            tree_root_obs=root_obs,
        )

    def _run_mcts_search(
        self,
        tree: MCTSTreeState,
        horizon: int,
        conditions: Optional[Any],
        start: np.ndarray,
        goal: np.ndarray,
        opposite_leaf_nodes: Optional[List["TreeNode"]] = None,
        single_step: bool = False,
        use_simulation: bool = True,
    ) -> Tuple[MCTSTreeState, Dict[str, dict]]:
        """
        (B function) Run the MCTS search loop for a given tree state.

        When `single_step=False` (default), runs until max_search_num or time_limit.
        When `single_step=True`, executes exactly one Selection→Expansion→Simulation→
        Backpropagation→EarlyTermination cycle and returns.

        In bidirectional mode, `opposite_leaf_positions` provides the leaf positions
        from the other tree so that dynamic goal selection can be performed via HILP.

        Args:
            tree: MCTSTreeState initialized by _init_mcts_tree
            obs_normalized: Normalized start observation
            goal_normalized: Normalized goal observation
            horizon: Planning horizon
            conditions: Planning conditions
            start: Raw (unnormalized) start observation, shape (1, obs_dim)
            goal: Raw (unnormalized) goal observation, shape (1, obs_dim)
            opposite_leaf_nodes: List of TreeNode objects from the opposite tree's
                                     current leaf nodes (used for dynamic goal selection).
                                     None → use the fixed `goal` as target.
            single_step: If True, run only one iteration of the MCTS loop then return.

        Returns:
            (MCTSTreeState, expanded_node_infos):
                - updated tree state after search
                - dict keyed by node name, each value is the candidate info dict with fields:
                    {
                      'node': TreeNode,            # the newly created child TreeNode (set after expand())
                      'value': float,              # best value found across denoising steps
                      'plan_history': list,        # nested plan tensors
                      'parent_node': TreeNode,     # the parent node that was expanded
                      'target_node': TreeNode|None,# dynamically selected opposite-tree leaf (bidir only)
                      ... (other TreeNode constructor fields)
                    }
                  Empty dict when no expansion occurred (e.g. all candidates already expanded).
        """
        # Unpack frequently used tree fields for readability
        root_node = tree.root_node
        children_node_guidance_scales = tree.children_node_guidance_scales
        num_denoising_steps = tree.num_denoising_steps
        skip_level_steps = tree.skip_level_steps
        noise_level = tree.noise_level
        from_start = tree.from_start
        terminal_depth = tree.terminal_depth

        # Variable to hold expanded_node_updated_levels across the loop
        expanded_node_updated_levels: Optional[np.ndarray] = None

        # Holds the expanded node infos from the latest iteration (reset each iteration)
        expanded_node_infos: Dict[str, dict] = {}

        while True:
            if self.time_limit is not None:
                if time.time() - self.start_time > self.time_limit:
                    break
            else:
                #if search_num >= max_search_num:
                if tree.p_search_num >= tree.max_search_num:
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

                # [DEBUG]
                # nz = np.count_nonzero(selected_node.current_levels) if selected_node.current_levels is not None else "N/A"
                # print(f"[DEBUG] Selected Node: {selected_node.name}, Depth: {selected_node.depth}/{selected_node.terminal_depth}, Non-zero tokens: {nz}")

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
            tree.selection_time.append(selection_end_time - selection_start_time)

            # ------------------------------------------------------------------
            # Dynamic Start & Goal Selection for each expansion candidate
            # ------------------------------------------------------------------
            eff_obs_norm_list, eff_goal_norm_list = [], []
            eff_start_np_list, eff_goal_np_list = [], []

            for info in expanded_node_candidates:
                parent_node = info["parent_node"]
                parent_obs_pos = parent_node.obs_pos
                
                # Start: Normalized parent position for planning context
                eff_start_np_list.append(parent_obs_pos[None, :self.observation_dim])
                p_norm = torch.tensor(
                    (parent_obs_pos[:self.observation_dim] - self.obs_mean) / self.obs_std,
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                eff_obs_norm_list.append(p_norm)
                
                # Goal: Dynamic selection if bidirectional, else the tree's target (global goal or global start)
                if self.bidirectional_search:
                    assert opposite_leaf_nodes is not None and len(opposite_leaf_nodes) > 0, "opposite_leaf_nodes is empty"
                    target_node = self._select_dynamic_goal(
                        current_leaf_obs=parent_obs_pos,
                        opposite_leaf_nodes=opposite_leaf_nodes,
                    )
                    info["target_node"] = target_node  # Will be propagated to child TreeNode via expand()
                    target_pos = target_node.obs_pos
                else:
                    target_node = None
                    target_pos = goal[0] # The tree's target (global goal or global start)
                
                eff_goal_np_list.append(target_pos[None, :self.observation_dim])
                g_norm = torch.tensor(
                    (target_pos[:self.observation_dim] - self.obs_mean) / self.obs_std,
                    dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                eff_goal_norm_list.append(g_norm)
                
            effective_obs_normalized = torch.cat(eff_obs_norm_list, dim=0)    # (B, D)
            effective_goal_normalized = torch.cat(eff_goal_norm_list, dim=0)   # (B, D)
            effective_starts_np = np.concatenate(eff_start_np_list, axis=0)  # (B, D)
            effective_goals_np = np.concatenate(eff_goal_np_list, axis=0)    # (B, D)

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
                    if self.bidirectional_search:
                        # Build plan from history and intermediate obs_pos
                        seg_size = tree.plan_tokens // self.sequence_dividing_factor
                        initial_plan = self._build_plan_from_leaf(
                            parent_node=info["parent_node"],
                            plan_tokens=tree.plan_tokens,
                            segment_size=seg_size,
                            from_start=from_start,
                        ) # (plan_tokens * fs, 1, c)
                        expanded_node_plans.append(initial_plan) # (T*fs, 1, c)
                    else:
                        if len(info["plan_history"]) == 0:
                            expanded_node_plans.append(None)
                        else:
                            expanded_node_plans.append(info["plan_history"][-1][-1].unsqueeze(1))
                    expanded_node_guidance_scales.append(info["guidance_scale"])

                    if not self.bidirectional_search:
                        _noise_level = noise_level[(info["depth"] - 1) * num_denoising_steps : (info["depth"] * num_denoising_steps + 1)]
                        #if info["depth"] == terminal_depth:
                        _noise_level = np.concatenate([_noise_level] + [noise_level[-1:]]*(num_denoising_steps - _noise_level.shape[0]+1)) # (num_denoising_steps, T)
                        expanded_node_noise_levels.append(_noise_level)

                expanded_node_guidance_scales = torch.tensor(expanded_node_guidance_scales).to(obs_normalized.device) # (batch_size,)

                if not self.bidirectional_search:
                    expanded_node_noise_levels = np.array(expanded_node_noise_levels, dtype=np.int32) # (batch_size, height, width)


                if self.bidirectional_search:
                    parent_levels_list = []
                    for info in expanded_node_candidates: # b
                        parent_node = info["parent_node"]
                        if parent_node.current_levels is not None:
                            parent_levels_list.append(parent_node.current_levels)
                        else:
                            # Fallback: initialize fresh if parent doesn't have state
                            assert horizon % self.frame_stack == 0, "Horizon must be divisible by frame_stack"
                            plan_tokens = horizon // self.frame_stack
                            # current_levels only contains middle tokens (excluding init/final tokens)
                            init_levels = np.full((1, plan_tokens), self.sampling_timesteps, dtype=np.int64)
                            parent_levels_list.append(init_levels)

                    parent_levels = np.concatenate(parent_levels_list, axis=0)  # (b, plan_tokens)

                    # In the new sequence layout, obs_parent_token occupies position
                    # prefix_len = parent_node.depth * seg_size within plan_tokens.
                    # Set its noise level to 0 (it is a given/observed token, not to be denoised).
                    _seg_size = tree.plan_tokens // self.sequence_dividing_factor
                    for _b, _info in enumerate(expanded_node_candidates):
                        _prefix_len_b = _info["parent_node"].depth * _seg_size
                        if _prefix_len_b < parent_levels.shape[1]:
                            parent_levels[_b, _prefix_len_b] = 0

                    # Generate Schedule for Bidirectional
                    expanded_node_noise_levels = self._generate_bidirectional_schedule(
                        parent_levels, complete_denoising=False, from_start=from_start
                    ) # b, m, plan_tokens(=t)
                    expanded_node_updated_levels = expanded_node_noise_levels[:, -1, :] # b, plan_tokens

                # input plans.shape: b (t fs) 1 c
                # output plan_hist.shape: m (t fs) b c
                # plan_hist = expanded_node_plan_hists[:, :, i] <- m (t fs) c
                # expanded_node_infos[name]["plan_history"][-1] = plan_hist <- d m (t fs) c

                expanded_node_plan_hists = self.parallel_plan(
                    start=effective_obs_normalized, 
                    goal=effective_goal_normalized, 
                    horizon=horizon, 
                    conditions=conditions,
                    guidance_scale=expanded_node_guidance_scales,
                    noise_level=expanded_node_noise_levels,
                    plans=expanded_node_plans,
                    from_start=from_start,
                    is_unknown_final_token=self.is_unknown_final_token
                )
                if self.debug:
                    print(f"  [DEBUG] [{tree.root_node.name}-Search] Expansion completed for {len(expanded_node_candidates)} nodes. plan_hists shape: {expanded_node_plan_hists.shape}")

                print(f"Expanded node plan hists: {expanded_node_plan_hists.shape}")
                print("============ Expansion End ============")
                expansion_end_time = time.time()
                tree.expansion_time.append(expansion_end_time - expansion_start_time)

                ###############################
                # Simulation
                #  It includes the noise level zero-padding, finding the max denoising steps, simulation, value calculation and node allocation
                simulation_start_time = time.time()
                
                if use_simulation:
                    print("============ Simulation Start ============")
                    # Pad the noise levels - Sequential
                    simul_noiselevel_zero_padding_start = time.time()
                    value_estimation_plans, value_estimation_noise_levels = [], []
                    max_denoising_steps = 0
                    for i in range(len(expanded_node_candidates)): # find the max denoising steps
                        if not self.bidirectional_search:
                            _noise_level = np.concatenate(
                                [noise_level[(expanded_node_candidates[i]["depth"] * num_denoising_steps)::skip_level_steps],
                                noise_level[-1:]], axis=0)
                            # update max denoising steps
                            if _noise_level.shape[0] > max_denoising_steps:
                                max_denoising_steps = _noise_level.shape[0]
                            value_estimation_noise_levels.append(_noise_level)

                        # expanded_node_plan_hists: m (t fs) b c
                        # For bidirectional (pre-built format): wrap plan to (n_tokens, 1, fs*c).
                        # For legacy format: keep as (t fs) 1 c.
                        _plan_t_fs = expanded_node_plan_hists[-1, :, i].unsqueeze(1)  # (plan_tokens*fs, 1, c)
                        if self.bidirectional_search:
                            _plan_tokens_val = horizon // self.frame_stack
                            _plan_rearranged = rearrange(_plan_t_fs, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)  # (plan_tokens, 1, fs*c)
                            _sim_pad_tokens = self.n_tokens - _plan_tokens_val
                            _sim_pad = torch.zeros((_sim_pad_tokens, 1, _plan_rearranged.shape[-1]), device=self.device)
                            value_estimation_plans.append(torch.cat([_plan_rearranged, _sim_pad], dim=0))  # (n_tokens, 1, fs*c)
                        else:
                            value_estimation_plans.append(_plan_t_fs)  # (t fs) 1 c — legacy format

                    if not self.bidirectional_search:
                        for i in range(len(expanded_node_candidates)): # zero-padding
                            length = value_estimation_noise_levels[i].shape[0]
                            if length < max_denoising_steps:
                                value_estimation_noise_levels[i] = np.concatenate([
                                    value_estimation_noise_levels[i],
                                    np.zeros((max_denoising_steps - length, value_estimation_noise_levels[i].shape[1]), dtype=np.int32)],
                                    axis=0) # zero-padding
                    simul_noiselevel_zero_padding_end = time.time()
                    tree.simul_noiselevel_zero_padding_time.append(simul_noiselevel_zero_padding_end - simul_noiselevel_zero_padding_start)

                    # Simulation - Value Estimation
                    simul_value_estimation_start = time.time()
                    if not self.bidirectional_search:
                        value_estimation_noise_levels = np.array(value_estimation_noise_levels, dtype=np.int32)

                    # Prepare expanded node's denoising state for simulation
                    simulation_initial_levels_list = []
                    for i in range(len(expanded_node_candidates)):
                        if expanded_node_updated_levels is not None:
                            simulation_initial_levels_list.append(expanded_node_updated_levels[i:i+1])
                        else:
                            simulation_initial_levels_list.append(None)

                    if self.bidirectional_search:
                        if expanded_node_updated_levels is not None:
                             simulation_initial_levels = np.concatenate(simulation_initial_levels_list, axis=0) # b, plan_tokens
                             # Generate Schedule for Simulation (Complete Denoising)
                             value_estimation_noise_levels = self._generate_bidirectional_schedule(
                                simulation_initial_levels, complete_denoising=True, from_start=from_start
                            )
                        else:
                             assert 0, "Should not happen if bidirectional"

                    # input plans.shape: b (t fs) 1 c
                    # output plan_hist.shape: m (t fs) b c
                    value_estimation_plan_hists = self.parallel_plan(
                        effective_obs_normalized, effective_goal_normalized, horizon, conditions,
                        guidance_scale=expanded_node_guidance_scales,
                        noise_level=value_estimation_noise_levels,
                        plans=value_estimation_plans,
                        from_start=from_start,
                        is_unknown_final_token=self.is_unknown_final_token
                    )

                    simul_value_estimation_end = time.time()
                    print(f"Value estimation plan hist: {value_estimation_plan_hists.shape}")

                    # check if any plan is good
                    plans = self._unnormalize_x(value_estimation_plan_hists[-1])[:-1].detach().cpu().numpy() # (t fs) b c
                    diffs = np.linalg.norm(plans[1:] - plans[:-1], axis=-1) # (plan_len-1, N)
                    for i in range(diffs.shape[1]):
                        if filtered_expanded_node_plan_hists[i] is None and not np.all(diffs[:, i] < self.meeting_delta):
                            filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]  # m (t fs) b c -> m (t fs) c
                            filtered_value_estimation_plan_hists[i] = value_estimation_plan_hists[:, :, i]

                    if None in filtered_expanded_node_plan_hists:
                        print("No good plan found, resampling")
                        simulation_end_time = time.time()
                        tree.simulation_time.append(simulation_end_time - simulation_start_time)
                        continue
                    else:
                        break
                else:
                    # Skip simulation: use HILP value directly for expansion results
                    for i in range(len(expanded_node_candidates)):
                        filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                        # Create dummy value_estimation_plan_hists using expanded_node_plan_hists
                        filtered_value_estimation_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                    break


            #----------------------SIM (DDIM) LOOP END----------------------------------------

            for i in range(len(filtered_expanded_node_plan_hists)):
                if filtered_expanded_node_plan_hists[i] is None:
                    filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                    filtered_value_estimation_plan_hists[i] = value_estimation_plan_hists[:, :, i]
            expanded_node_plan_hists = torch.stack(filtered_expanded_node_plan_hists, dim=2) # m (t fs) 'B' c
            value_estimation_plan_hists = torch.stack(filtered_value_estimation_plan_hists, dim=2) # m (t fs) 'B' c

            # TODO: Value Calculation
            simul_value_calculation_start = time.time()
            achieved_sim_indices = []
            if self.bidirectional_search:
                values, infos, achieved_ts = self.calculate_values_bidir(
                    expanded_node_candidates, expanded_node_plan_hists, tree
                )
            else:
                values, infos, achieved_ts = self.calculate_values(value_estimation_plan_hists[-1], effective_starts_np, effective_goals_np, from_start=from_start) # (plan_len, N, D), (N, D), (N, D)
            for i in range(len(infos)): # B
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    tree.achieved_plans.append([value_estimation_plan_hists[-1, :achieved_t, i], values[i]])
                    tree.achieved = True
                    achieved_sim_indices.append(i)
                elif info == "NotReached":
                    tree.not_reached_plans.append([value_estimation_plan_hists[-1, :, i], values[i]])
            print(f"Value Calculation: {values}, {infos}")
            simul_value_calculation_end = time.time()

            # Node Allocation
            simul_node_allocation_start = time.time()
            selected_nodes_for_expansion = {}
            expanded_node_infos = {}
            for i in range(len(expanded_node_candidates)): # B
                name = expanded_node_candidates[i]["name"]
                if name not in expanded_node_infos:
                    selected_nodes_for_expansion[name] = selected_nodes[i]
                    expanded_node_infos[name] = expanded_node_candidates[i]
                    expanded_node_infos[name]["plan_history"].append([])
                    expanded_node_infos[name]["is_tree1"] = tree.from_start
                value = values[i]
                plan_hist = expanded_node_plan_hists[:, :, i] # m (t fs) c
                value_estimation_plan = value_estimation_plan_hists[-1, :, i]

                # Store updated denoising state for child node
                if expanded_node_updated_levels is not None:
                    updated_level = expanded_node_updated_levels[i:i+1]  # Shape: (1, plan_tokens)
                else:
                    updated_level = None

                if expanded_node_infos[name]["value"] is None:
                    expanded_node_infos[name]["value"] = value
                    expanded_node_infos[name]["value_estimation_plan"] = value_estimation_plan
                    expanded_node_infos[name]["plan_history"][-1] = plan_hist # d m (t fs) c
                    expanded_node_infos[name]["current_levels"] = updated_level
                else:
                    if value > expanded_node_infos[name]["value"]:
                        expanded_node_infos[name]["value"] = value
                        expanded_node_infos[name]["value_estimation_plan"] = value_estimation_plan
                        expanded_node_infos[name]["plan_history"][-1] = plan_hist
                        expanded_node_infos[name]["current_levels"] = updated_level

            for name in selected_nodes_for_expansion:
                child_node = selected_nodes_for_expansion[name].expand(**expanded_node_infos[name])
                expanded_node_infos[name]["node"] = child_node

            simul_node_allocation_end = time.time()
            tree.simul_node_allocation_time.append(simul_node_allocation_end - simul_node_allocation_start)

            print("============ Simulation End ============")
            simulation_end_time = time.time()
            tree.simulation_time.append(simulation_end_time - simulation_start_time)

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
            tree.backprop_time.append(backprop_end_time - backprop_start_time)

            ######################
            # Early Termination
            early_termination_start_time = time.time()
            print("============ Early Termination Start ============")

            # plan_history: d m (t fs) c
            # plans: (t fs) B c
            plans = torch.stack([info["plan_history"][-1][-1] for info in expanded_node_infos.values()], dim=1)
            if self.bidirectional_search:
                _, infos, achieved_ts = self.calculate_values_bidir(
                    expanded_node_candidates, expanded_node_plan_hists, tree
                )
            else:
                _, infos, achieved_ts = self.calculate_values(plans, start, goal, from_start=from_start) # (plan_len, N, D), (N, D), (N, D)
            print(f"Early Termination: {infos}, {achieved_ts}")
            tree.solved = False
            achieved_indices = []
            early_termination_achieved_plans = []
            for i in range(len(infos)):
                info = infos[i]
                achieved_t = achieved_ts[i]
                if info == "Achieved":
                    tree.solved = True
                    terminal_ts = achieved_t
                    # early_termination_achieved_plans.append(plans[:terminal_ts, i]) # b (t fs) c
                    # solved_plan = plans[:terminal_ts, i]
                    achieved_indices.append(i)

            if tree.solved:
                tree.solved_plan = plans[:, achieved_indices[0]]
            else:
                tree.solved_plan = None


            print("============ Early Termination End ============")
            early_termination_end_time = time.time()
            tree.early_termination_time.append(early_termination_end_time - early_termination_start_time)


            tree.search_num += 1
            tree.p_search_num += len(expanded_node_candidates)
            tree.pbar.update(len(expanded_node_candidates))
            tree.max_depth = max(tree.max_depth, max([info["depth"] for info in expanded_node_candidates]))
            is_early_termination = (self.early_stopping_condition == "solved" and tree.solved) or (self.early_stopping_condition == "achieved" and tree.achieved)

            if self.viz_plans:
                depths = [info["depth"] for info in expanded_node_candidates]
                terminal_indices = [i for i, info in enumerate(expanded_node_candidates) if info["depth"] == terminal_depth]

                if is_early_termination:
                    if self.early_stopping_condition == "solved" and tree.solved:
                        terminal_indices = list(set(terminal_indices) | set(achieved_indices))
                    elif self.early_stopping_condition == "achieved" and tree.achieved:
                        terminal_indices = list(set(terminal_indices) | set(achieved_sim_indices))
                    terminal_indices = sorted(terminal_indices)

                # print(f"[DEBUG] viz_plans=True at search_num={tree.search_num}")
                # print(f"[DEBUG] terminal_depth={terminal_depth}")
                # print(f"[DEBUG] expanded_node_candidates depths={depths}")
                # print(f"[DEBUG] terminal_indices count={len(terminal_indices)}")

                if len(terminal_indices) > 0:
                    terminal_values = values[terminal_indices]
                    terminal_names = [expanded_node_candidates[i]["name"] for i in terminal_indices]
                    terminal_expanded_hists = expanded_node_plan_hists[-1, :, terminal_indices]   # m (t fs) b c
                    # terminal_estimation_hists = value_estimation_plan_hists[-1, :, terminal_indices] # m (t fs) b c
                    self.visualize_node_value_plans(tree.search_num, terminal_values, terminal_names,
                        terminal_expanded_hists,
                        #terminal_estimation_hists[-1],
                        start, goal, tag=tree.tag)

                # elif is_early_termination and len(achieved_indices) > 0:
                #     achieved_values = values[achieved_indices]
                #     achieved_names = [expanded_node_candidates[i]["name"] for i in achieved_indices]
                #     self.visualize_node_value_plans(tree.search_num, achieved_values, achieved_names,
                #         plans[:, achieved_indices],
                #         start, goal, tag=tree.tag)

            if is_early_termination:
                break

            # ------------------------------------------------------------------
            # single_step mode: exit after 1 iteration (expanded_node_infos already set)
            # ------------------------------------------------------------------
            if single_step:
                break

        tree.pbar.close()
        return tree, expanded_node_infos

    # =========================================================================
    # Helper functions for bidirectional alternating MCTS
    # =========================================================================

    def _build_plan_from_leaf(
        self,
        parent_node: "TreeNode",
        plan_tokens: int,
        segment_size: int,
        from_start: bool,
    ) -> torch.Tensor:
        # Assembles a diffusion sequence: [prior trajectory | current obs | random noise | padding]
        """Construct the full plan_with_given_tokens for a new leaf node expansion.

        Returns a tensor of shape (n_tokens, 1, fs*c) with layout:
            [denoised_prefix(prefix_len) | obs_parent_token(1) | noisy_chunk | padding]
        When denoised_prefix is empty (root depth=0):
            [obs_parent_token(1) | noisy_chunk | padding]

        This output is ready to be passed directly to parallel_plan (pre-built format).
        """
        if self.debug:
            print(f"    [DEBUG] Building initial plan from leaf. Parent: {parent_node.name}, Depth: {parent_node.depth}, History Segments: {len(parent_node.plan_history)}")

        # Build obs_parent_token: the parent node's current observation, tokenised.
        parent_obs_pos = parent_node.obs_pos
        parent_obs_tensor = torch.tensor(parent_obs_pos, dtype=torch.float32, device=self.device)
        obs_parent_token_raw = self.pad_init(parent_obs_tensor.unsqueeze(0))  # (fs, 1, c)
        if not from_start:
            # Backward search: frames inside token are flipped to backward order.
            obs_parent_token_raw = torch.flip(obs_parent_token_raw, [0])
        obs_parent_token = rearrange(obs_parent_token_raw, "fs b c -> 1 b (fs c)")  # (1, 1, fs*c)

        # --- Build denoised prefix from parent's plan_history ---
        if parent_node.plan_history:
            # plan_history stores plans in canonical (forward) order via flip_plan_for_insert_hist.
            latest_plan_canonical = parent_node.plan_history[-1][-1]  # (plan_tokens*fs, c)
            prefix_len_frames = parent_node.depth * segment_size * self.frame_stack
            full_prefix_canonical = latest_plan_canonical[:prefix_len_frames].unsqueeze(1)  # (prefix_len*fs, 1, c)

            # Normalize for diffusion model input.
            full_prefix = self._normalize_x(full_prefix_canonical)  # (prefix_len*fs, 1, c)
            denoised_prefix = rearrange(full_prefix, "(t fs) b c -> t b (fs c)", fs=self.frame_stack)  # (prefix_len, 1, fs*c)
            prefix_len = denoised_prefix.shape[0]

            if not from_start and prefix_len > 0:
                # Canonical order is forward; convert back to backward order for the sequence.
                denoised_prefix_raw = rearrange(denoised_prefix, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
                denoised_prefix = rearrange(torch.flip(denoised_prefix_raw, [0]), "(t fs) b c -> t b (fs c)", fs=self.frame_stack)
        else:
            denoised_prefix = None
            prefix_len = 0

        # Layout within plan_tokens: [prefix(prefix_len) | obs_parent(1) | noisy(plan_tokens-1-prefix_len)]
        # obs_parent_token takes 1 slot, leaving plan_tokens-1-prefix_len for noisy content.
        noisy_total = plan_tokens - 1 - prefix_len
        assert noisy_total >= 0, f"Noisy total must be non-negative: {noisy_total}"

        batch_size = obs_parent_token.shape[1]  # always 1 per leaf
        noisy_parts = torch.randn(
            (noisy_total, batch_size, *self.x_stacked_shape),
            device=self.device,
        )
        noisy_parts = torch.clamp(noisy_parts, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)

        # Assemble plan_tokens-length chunk: [prefix | obs_parent | noisy]
        if denoised_prefix is not None:
            plan_chunk = torch.cat([denoised_prefix, obs_parent_token, noisy_parts], dim=0)
        else:
            plan_chunk = torch.cat([obs_parent_token, noisy_parts], dim=0)

        assert plan_chunk.shape[0] == plan_tokens, \
            f"Plan chunk length mismatch: {plan_chunk.shape[0]} != {plan_tokens}"

        # Append zero-padding to reach n_tokens.
        pad_tokens = self.n_tokens - plan_tokens
        assert pad_tokens >= 0, f"pad_tokens must be non-negative: {pad_tokens}"
        pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)

        return torch.cat([plan_chunk, pad], dim=0)  # (n_tokens, 1, fs*c)

    def _select_dynamic_goal(
        self,
        current_leaf_obs: np.ndarray,
        opposite_leaf_nodes: List["TreeNode"],
    ) -> "TreeNode":
        """Select the best goal from the opposite tree's leaf nodes using HILP value.

        Computes V(current_leaf_obs, candidate.obs_pos) for each candidate in
        `opposite_leaf_nodes` and returns the node with the highest value
        (i.e., temporally closest to `current_leaf_obs`).

        Args:
            current_leaf_obs: Unnormalized observation of the leaf node being expanded,
                              shape (obs_dim,).
            opposite_leaf_nodes: List of TreeNode objects from the opposite tree's leaf nodes.

        Returns:
            best_node: The TreeNode from opposite_leaf_nodes with the highest HILP value.
        """
        targets = np.stack([n.obs_pos for n in opposite_leaf_nodes]) # (N, D)
        obs_expanded = np.tile(current_leaf_obs, (targets.shape[0], 1)) # (N, D)
        values = self._compute_hilp_values(obs_expanded, targets, use_no_grad=True)
        
        best_idx = torch.argmax(values).item()
        best_value = values[best_idx].item()
        best_node = opposite_leaf_nodes[best_idx]
        
        if self.debug:
            print(f"      [DEBUG] Dynamic Goal Selection: Evaluated {len(opposite_leaf_nodes)} candidates. Best Value: {best_value:.4f}")
        return best_node

    def _rollout_leaf_plan(
        self,
        leaf_plan_tokens: torch.Tensor,
        new_denoised_start_idx: int,
        new_denoised_end_idx: int,
        agent: Any,
        envs: Any,
        parent_sim_state: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Execute a freshly denoised plan segment in the actual environment.
        Restores the parent's physical state before stepping to ensure consistency.

        Args:
            leaf_plan_tokens: Fully assembled plan tensor, shape (T, 1, c) unnormalized.
            new_denoised_start_idx: Start token index of the freshly denoised chunk.
            new_denoised_end_idx: End token index (exclusive) of the freshly denoised chunk.
            agent: RL agent (used for antmaze sub-goal following).
            envs: Vectorized environment.
            parent_sim_state: Physical state (qpos/qvel) of the parent node to restore.

        Returns:
            Tuple containing:
              - last_obs: shape (obs_dim,) — agent's position after the rollout.
              - final_sim_state: dictionary containing reached qpos/qvel.
        """
        # Restore parent's physical state before simulation
        assert parent_sim_state is not None, "Parent sim state must be provided for rollout"
        
        self._set_sim_state(envs, parent_sim_state)
        
        # Construct initial observation from restored sim state
        full_obs = np.concatenate([parent_sim_state["qpos"], parent_sim_state["qvel"]])
        obs_numpy = full_obs[:self.observation_dim][None, :]

        plan_slice = leaf_plan_tokens[new_denoised_start_idx:new_denoised_end_idx]  # (chunk_t, 1, c)
        last_obs = obs_numpy[0].copy()  # default: no movement

        if plan_slice.shape[0] == 0:
            return last_obs, self._get_sim_state(envs)

        plan_slice_np = plan_slice[:, 0, :].detach().cpu().numpy()  # (chunk_t, c)

        for t in range(plan_slice.shape[0]):
            if "antmaze" in self.env_id:
                sub_goal = plan_slice_np[t, :2]
                action = agent.sample_action(obs_numpy, sub_goal[None])
                action = torch.from_numpy(action).float().reshape(1, -1)
            else:
                if t == 0:
                    obs_t = torch.from_numpy(obs_numpy).float()
                    plan_vel = plan_slice[t, :, :2] - obs_t[:, :2]
                else:
                    plan_vel = plan_slice[t, :, :2] - plan_slice[t - 1, :, :2]
                action = 12.5 * (plan_slice[t, :, :2] - torch.from_numpy(obs_numpy).float()[:, :2]) + \
                         1.2 * (plan_vel - torch.from_numpy(obs_numpy).float()[:, 2:4])
                action = torch.clip(action, -1, 1)

            action_np = action.detach().cpu().numpy()
            obs_numpy, _, done, _ = envs.step(np.nan_to_num(action_np))
            last_obs = obs_numpy[0].copy()
            if done.any():
                break

        # Capture reached physical state
        final_sim_state = self._get_sim_state(envs)

        return last_obs, final_sim_state


    def _select_best_leaf(
        self,
        expanded_node_infos: Dict[str, dict],
    ) -> dict:
        """
        Select the best expanded node info from an expanded_node_infos dict.

        Selects the candidate with the highest 'value' field.
        The returned dict contains a 'node' key with the actual child TreeNode.

        Args:
            expanded_node_infos: Dict[name -> info_dict] as returned by _run_mcts_search.

        Returns:
            The info dict with the highest value.
        """
        return max(
            expanded_node_infos.values(),
            key=lambda info: info["value"] if info.get("value") is not None else float("-inf"),
        )

    def _extract_output_plan(
        self,
        best_node: "TreeNode",
        plan_tokens: int,
        is_tree1: bool,
    ) -> torch.Tensor:

        """
        Construct the final output plan from the best selected leaf TreeNode.

        In bidirectional mode (best_node.target_node is not None):
            - Takes plan_A from best_node (forward tree leaf) sliced by depth.
            - Takes plan_B from best_node.target_node (backward tree leaf) sliced by depth, then flipped.
            - Returns the concatenated plan: plan_A + flip(plan_B).

        In unidirectional mode (best_node.target_node is None):
            - Returns plan_A only (forward tree leaf sliced by depth).

        Args:
            best_node: The selected best leaf TreeNode (from _select_best_leaf).
            plan_tokens: Total number of plan tokens for the tree (determines seg_size).

        Returns:
            output_plan: Tensor of shape (T, 1, c), where T = combined path length.
        """
        seg_size: int = plan_tokens // self.sequence_dividing_factor

        # --- Plan A: forward tree leaf ---
        plan_a_full: torch.Tensor = best_node.plan_history[-1][-1]  # (T_total*fs, c)
        a_len: int = best_node.depth * seg_size * self.frame_stack
        t1_segments: torch.Tensor = plan_a_full[:a_len]  # (A_len, c)

        if best_node.target_node is None:
            # --- Unidirectional: use plan_A only ---
            combined = t1_segments
        else:
            # --- Bidirectional: flip plan_B and concat ---
            plan_b_full: torch.Tensor = best_node.target_node.plan_history[-1][-1]  # (T_total*fs, c)
            b_len: int = best_node.target_node.depth * seg_size * self.frame_stack
            t2_flipped: torch.Tensor = torch.flip(plan_b_full[:b_len], [0])  # (B_len, c)

            if self.debug:
                print(
                    f"[DEBUG] [Extract Plan] A_len={a_len}, B_len={b_len}, "
                    f"Combined={a_len + b_len}"
                )

            combined = torch.cat([t1_segments, t2_flipped], dim=0)

        if not is_tree1:
            combined = torch.flip(combined, [0])
        return combined.unsqueeze(1)  # (T_combined, 1, c)



############### DEPRECATED ###############
    def p_mctd_plan(
        self,
        obs_normalized: torch.Tensor,
        goal_normalized: torch.Tensor,
        horizon: int,
        conditions: Optional[Any],
        start: np.ndarray,
        goal: np.ndarray,
        tag: str = "mcts_plan",
        from_start: bool = True,
    ) -> torch.Tensor:
        """
        Orchestrator for MCTS-based diffusion planning.

        When bidirectional_search=False:
            - Initializes one tree (T1, from_start=True) and runs the full search.
        When bidirectional_search=True:
            - Initializes two trees: T1 (from_start=True) and T2 (from_start=False).
            - Runs the full search on T1, then the full search on T2 (each up to max_search_num).
            - TODO: bidirectional meeting point (connect T1 and T2 plans).
            - Currently returns T1's result as the output plan.

        Args:
            obs_normalized: Normalized start observation, shape (1, obs_dim)
            goal_normalized: Normalized goal observation, shape (1, obs_dim)
            horizon: Planning horizon (must be divisible by frame_stack)
            conditions: Planning conditions
            start: Raw (unnormalized) start observation, shape (1, obs_dim)
            goal: Raw (unnormalized) goal observation, shape (1, obs_dim)
            tag: Tag string for logging
            from_start: Ignored when bidirectional_search=True (both directions are used);
                        used as-is when bidirectional_search=False.

        Returns:
            output_plan: shape (1, t, 1, c)
        """
        assert start.shape[0] == 1, "the batch size must be 1"
        assert (not self.leaf_parallelization) or (self.parallel_search_num % len(self.mctd_guidance_scales) == 0), \
            f"Parallel search num must be divisible by the number of guidance scales: {self.parallel_search_num} % {len(self.mctd_guidance_scales)} != 0"

        assert horizon <= self.episode_len, f"Horizon must be less than or equal to episode length: {horizon} <= {self.episode_len}"
        assert horizon % self.frame_stack == 0, f"Horizon must be divisible by frame stack: {horizon} % {self.frame_stack} != 0"

        def _get_all_leaves(root: "TreeNode") -> List["TreeNode"]:
            res = []
            s = [root]
            while s:
                n = s.pop()
                if all(c["node"] is None for c in n._children_nodes): res.append(n)
                else:
                    for c in n._children_nodes:
                        if c["node"] is not None: s.append(c["node"])
            return res

        if self.bidirectional_search:
            # --- Bidirectional mode: two trees, each searched independently ---
            tree1 = self._init_mcts_tree(horizon, from_start=True,  tag=tag + "_from_start")
            tree2 = self._init_mcts_tree(horizon, from_start=False, tag=tag + "_from_goal")

            tree1, infos1 = self._run_mcts_search(tree1, horizon, conditions, start, goal)
            tree2, infos2 = self._run_mcts_search(tree2, horizon, conditions, start, goal)

            _all_infos: Dict[str, dict] = {}
            _all_infos.update(infos1 if infos1 else {})
            _all_infos.update(infos2 if infos2 else {})
            if _all_infos:
                _best_info = self._select_best_leaf(_all_infos)
                output_plan = self._extract_output_plan(_best_info["node"], plan_tokens=tree1.plan_tokens, is_tree1=_best_info["is_tree1"])
            else:
                # Fallback: no expansions occurred
                output_plan = torch.zeros((horizon, 1, tree1.plan_tokens // horizon * self.frame_stack), device=self.device)
            output_plan = output_plan.unsqueeze(0) # (1, T, 1, c)


            # Logging: sum both trees' stats
            self.log(f"validation/search_num",  tree1.search_num  + tree2.search_num)
            self.log(f"validation/p_search_num", tree1.p_search_num + tree2.p_search_num)
            self.log(f"validation/max_depth",    max(tree1.max_depth, tree2.max_depth))

            self.log(f"validation_time/selection_time",       np.sum(tree1.selection_time)       + np.sum(tree2.selection_time))
            self.log(f"validation_time/expansion_time",       np.sum(tree1.expansion_time)       + np.sum(tree2.expansion_time))
            self.log(f"validation_time/simulation_time",      np.sum(tree1.simulation_time)      + np.sum(tree2.simulation_time))
            self.log(f"validation_time/backprop_time",        np.sum(tree1.backprop_time)        + np.sum(tree2.backprop_time))
            self.log(f"validation_time/early_termination_time", np.sum(tree1.early_termination_time) + np.sum(tree2.early_termination_time))

            self.log(f"validation_time/simul_noiselevel_zero_padding_time", np.sum(tree1.simul_noiselevel_zero_padding_time) + np.sum(tree2.simul_noiselevel_zero_padding_time))
            self.log(f"validation_time/simul_value_estimation_time",        np.sum(tree1.simul_value_estimation_time)        + np.sum(tree2.simul_value_estimation_time))
            self.log(f"validation_time/simul_value_calculation_time",       np.sum(tree1.simul_value_calculation_time)       + np.sum(tree2.simul_value_calculation_time))
            self.log(f"validation_time/simul_node_allocation_time",         np.sum(tree1.simul_node_allocation_time)         + np.sum(tree2.simul_node_allocation_time))

        else:
            # --- Unidirectional mode: single tree, identical to original behavior ---
            tree1 = self._init_mcts_tree(horizon, from_start=from_start, tag=tag)
            tree1, infos1 = self._run_mcts_search(tree1, horizon, conditions, start, goal)
            if infos1:
                _best_info = self._select_best_leaf(infos1)
                output_plan = self._extract_output_plan(_best_info["node"], plan_tokens=tree1.plan_tokens, is_tree1=_best_info["is_tree1"])
            else:
                # Fallback: no expansions occurred
                output_plan = torch.zeros((horizon, 1, tree1.plan_tokens // horizon * self.frame_stack), device=self.device)
            output_plan = output_plan.unsqueeze(0) # (1, T, 1, c)

            # Logging: single tree stats
            self.log(f"validation/search_num",  tree1.search_num)
            self.log(f"validation/p_search_num", tree1.p_search_num)
            self.log(f"validation/max_depth",    tree1.max_depth)

            self.log(f"validation_time/selection_time",          np.sum(tree1.selection_time))
            self.log(f"validation_time/expansion_time",          np.sum(tree1.expansion_time))
            self.log(f"validation_time/simulation_time",         np.sum(tree1.simulation_time))
            self.log(f"validation_time/backprop_time",           np.sum(tree1.backprop_time))
            self.log(f"validation_time/early_termination_time",  np.sum(tree1.early_termination_time))

            self.log(f"validation_time/simul_noiselevel_zero_padding_time", np.sum(tree1.simul_noiselevel_zero_padding_time))
            self.log(f"validation_time/simul_value_estimation_time",        np.sum(tree1.simul_value_estimation_time))
            self.log(f"validation_time/simul_value_calculation_time",       np.sum(tree1.simul_value_calculation_time))
            self.log(f"validation_time/simul_node_allocation_time",         np.sum(tree1.simul_node_allocation_time))

        return output_plan

    def _get_sim_state(self, envs: Any) -> Optional[dict]:
        """Extract current qpos/qvel from envs (DummyVecEnv)."""
        try:
            # get_attr returns a list of attributes for each env in the vector
            # We assume batch size 1 for SimState restoration as per requirements
            data = envs.get_attr("data")
            if data and len(data) > 0:
                return {
                    "qpos": data[0].qpos.copy(),
                    "qvel": data[0].qvel.copy()
                }
        except Exception as e:
            if self.debug:
                print(f"  [DEBUG] Failed to get sim_state: {e}")
        return None

    def _set_sim_state(self, envs: Any, sim_state: Optional[dict]) -> None:
        """Restore qpos/qvel to envs (DummyVecEnv)."""
        if sim_state is None:
            return
        try:
            # env_method calls the method on each env in the vector
            envs.env_method("set_state", sim_state["qpos"], sim_state["qvel"])
        except Exception as e:
            if self.debug:
                print(f"  [DEBUG] Failed to set sim_state: {e}")

