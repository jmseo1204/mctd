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
    make_mpc_animation,
)
from utils.tracer import Tracer, set_default_tracer, get_tracer
from .tree_node import TreeNode
from . import guidance

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
    noise_level: Optional[
        np.ndarray
    ]  # always None; bidirectional dynamic schedule used
    children_node_guidance_scales: list
    max_search_num: int
    num_denoising_steps: int
    skip_level_steps: int
    tag: str
    is_tree1: bool = True  # True for start-rooted tree, False for goal-rooted tree
    # Root observation (unnormalized): start for tree1, goal for tree2.
    # Used to track agent positions across bidirectional expansion rounds.
    tree_root_obs: Optional[np.ndarray] = None  # shape (obs_dim,)
    # --- Mutable search state (updated by _run_mcts_search) ---
    search_num: int = 0
    p_search_num: int = 0
    max_depth: int = 0
    achieved: bool = False
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
        # [INSTRUMENTATION] Initialize tracer (will be set later in interact() or on-demand in parallel_plan())
        self.tracer = None
        
        self.env_id = cfg.env_id
        self.dataset = cfg.dataset
        self.action_dim = len(cfg.action_mean)
        self.observation_dim = len(cfg.observation_mean)
        self.use_reward = cfg.use_reward
        self.unstacked_dim = (
            self.observation_dim + self.action_dim + int(self.use_reward)
        )
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len

        # Manually initialize frame_stack as requested to solve dependency order
        self.frame_stack = cfg.frame_stack
        assert self.episode_len % self.frame_stack == 0, (
            "Episode length must be divisible by frame stack size"
        )
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
        self.val_max_loops = cfg.val_max_loops
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
        self.num_tries_for_bad_plans = cfg.num_tries_for_bad_plans
        self.sub_goal_interval = cfg.sub_goal_interval
        self.viz_plans = cfg.viz_plans
        self.meeting_delta = cfg.get("meeting_delta", 0.5)
        self.debug = cfg.get("DEBUG", False)
        self.debug_log_level = cfg.get("debug_log_level", 0)  # 0=off, 1=basic, 2=detailed, 3=verbose
        self.debug_log_interval = cfg.get("debug_log_interval", 10)  # Log every N iterations
        self.debug_memory_profile = cfg.get("debug_memory_profile", False)
        self.max_plan_hist_keep = cfg.get("max_plan_hist_keep", 1)  # Memory optimization: limit history
        self.sequence_dividing_factor = cfg.sequence_dividing_factor
        self.horizon_scale = cfg.horizon_scale
        self.pyramid = cfg.get("pyramid", False)

        # HILP value function guidance
        self.use_hilp_guidance = cfg.get("use_hilp_guidance", False)
        hilp_path = cfg.get("hilp_checkpoint_path", "td_models/hilp_ckpt_latest.pt")
        # Resolve path relative to repo root if relative
        import os
        if not os.path.isabs(hilp_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            hilp_path = os.path.join(repo_root, hilp_path)
        self.hilp_checkpoint_path = hilp_path
        self.hilp_obs_dim = cfg.get("hilp_obs_dim", 29)
        self.hilp_skill_dim = cfg.get("hilp_skill_dim", 32)
        # HILP value function instance will be loaded lazily and stored in _hilp_value_fn_instance
        # We don't initialize it here to prevent PyTorch from registering it as a submodule
        self.anchor_guidance_scale = cfg.get("anchor_guidance_scale", 40.0)
        self.rdf_guidance_scale = cfg.get("rdf_guidance_scale", 2.0)
        self.mcts_use_sim = cfg.get("mcts_use_sim", False)

        super().__init__(cfg)
        self.plot_end_points = cfg.plot_start_goal
        
        # Initialize memory profiler for debugging
        if self.debug_memory_profile:
            from utils.memory_profiler import init_profiler
            self.profiler = init_profiler(self.device, debug_level=self.debug_log_level)
            if self.debug_log_level >= 1:
                import sys
                print(f"[DEBUG] Memory profiler initialized (level={self.debug_log_level})", file=sys.stderr, flush=True)
        else:
            self.profiler = None
        
        # Log initialization config
        if self.debug_log_level >= 1:
            import sys
            print(f"[DEBUG] DiffusionForcingPlanning initialized:", file=sys.stderr, flush=True)
            print(f"  parallel_search_num: {self.parallel_search_num}", file=sys.stderr, flush=True)
            print(f"  mctd_max_search_num: {self.mctd_max_search_num}", file=sys.stderr, flush=True)
            print(f"  max_plan_hist_keep: {self.max_plan_hist_keep}", file=sys.stderr, flush=True)

    def _get_hilp_value_fn(self):
        """Lazy loader for HILP value function model."""
        # Use a non-Module attribute name to prevent PyTorch from registering it as a submodule
        if not hasattr(self, '_hilp_value_fn_instance') or self._hilp_value_fn_instance is None:
            import sys
            import os

            # Add algorithms directory to path to import cleandiffuser_ex
            algorithms_dir = os.path.join(os.path.dirname(__file__), "..")
            if algorithms_dir not in sys.path:
                sys.path.insert(0, algorithms_dir)
            from cleandiffuser_ex.hilp import HILP

            # Load HILP model
            hilp_model = HILP(
                obs_dim=self.hilp_obs_dim,
                skill_dim=self.hilp_skill_dim,
                device=self.device,
                value_hidden_dims=(512, 512, 512),
                use_layer_norm=True,
            )
            hilp_model.load(self.hilp_checkpoint_path)
            hilp_model.eval()

            # Freeze all parameters to prevent gradient updates
            for param in hilp_model.parameters():
                param.requires_grad = False

            # Store in a private attribute that won't be registered as a submodule
            object.__setattr__(self, '_hilp_value_fn_instance', hilp_model)
            print(f"[HILP] Loaded HILP value function from {self.hilp_checkpoint_path}")

        return object.__getattribute__(self, '_hilp_value_fn_instance')

    def _compute_hilp_values(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        goal: Union[np.ndarray, torch.Tensor],
        use_no_grad: bool = True,
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
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        if goal_t.ndim == 1:
            goal_t = goal_t.unsqueeze(0)

        # 3. STRICT SHAPE ASSERTION
        assert obs_t.shape == goal_t.shape, (
            f"[HILP Shape Error] obs and goal must have matching shapes. "
            f"Got obs: {obs_t.shape}, goal: {goal_t.shape}. "
            f"Broadcasting/Expansion must be handled by the caller (e.g. for guidance (T,B,D))."
        )
        assert obs_t.ndim == 2, (
            f"[HILP Shape Error] Expected 2D tensors (N, D), got {obs_t.shape}"
        )

        # 4. Padding/Cropping to self.hilp_obs_dim
        def _pad(x):
            if x.shape[-1] < self.hilp_obs_dim:
                padding = torch.zeros(
                    (*x.shape[:-1], self.hilp_obs_dim - x.shape[-1]), device=x.device
                )
                return torch.cat([x, padding], dim=-1)
            return x[..., : self.hilp_obs_dim]

        obs_t = _pad(obs_t)
        goal_t = _pad(goal_t)

        # 5. Compute values - DEBUG: Verify HILP module structure
        import sys
        has_value_attr = hasattr(hilp_value_fn, "value")
        print(f"[DEBUG _compute_hilp_values] hilp_value_fn type: {type(hilp_value_fn).__name__}, has_value: {has_value_attr}", file=sys.stderr, flush=True)

        if use_no_grad:
            with torch.no_grad():
                # Fix: Check for value() method BEFORE trying direct call
                if has_value_attr:
                    print(f"[DEBUG] Using hilp_value_fn.value() for computation", file=sys.stderr, flush=True)
                    v1, v2 = hilp_value_fn.value(obs_t, goal_t)
                    res = torch.min(v1, v2)
                else:
                    print(f"[DEBUG] Using hilp_value_fn() direct call", file=sys.stderr, flush=True)
                    v1, v2 = hilp_value_fn(obs_t, goal_t)
                    res = torch.min(v1, v2)
        else:
            if has_value_attr:
                print(f"[DEBUG] Using hilp_value_fn.value() for computation", file=sys.stderr, flush=True)
                v1, v2 = hilp_value_fn.value(obs_t, goal_t)
            else:
                print(f"[DEBUG] Using hilp_value_fn() direct call", file=sys.stderr, flush=True)
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
            raise ValueError(
                "Number of frames - 1 must be divisible by frame stack size"
            )

        nonterminals = torch.cat(
            [
                torch.ones_like(nonterminals[:, : self.frame_stack]),
                nonterminals[:, :-1],
            ],
            dim=1,
        )
        nonterminals = nonterminals.bool().permute(1, 0)  # (T, B)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()
        # masks = torch.cat([masks[:-self.frame_stack:self.jump], masks[-self.frame_stack:]], dim=0)

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(
            observations, [1, n_frames - 1], dim=1
        )  # (b t c_o)
        bundles = self._normalize_x(
            self.make_bundle(observations, actions, rewards)
        )  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = (
            0  # zero out actions and rewards after normalization
        )
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b, fs, c)
        bundles = torch.cat([init_bundle, bundles], dim=1)  # (b, fs+n_frames-1, c)
        bundles = rearrange(
            bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack
        )  # (n_tokens+1, b, fs, c)
        bundles = bundles.flatten(2, 3).contiguous()  # (n_tokens+1, b, fs*c)

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
            random_terminal = torch.randint(
                2, n_tokens + 1, (batch_size,), device=self.device
            )
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[
                :, :n_tokens
            ].bool()
            random_terminal = repeat(
                random_terminal, "b t -> (t fs) b", fs=self.frame_stack
            )
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.diffusion_model(
            xs, conditions, noise_levels=self._generate_noise_levels(xs, masks=masks)
        )

        loss = self.reweight_loss(loss, weights)

        self.log(
            "training/loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True
        )

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            o, a, r = self.split_bundle(xs_pred)
            trajectory = (
                o.detach().cpu().numpy()[:-1, :8]
            )  # last observation is dummy, sample 8
            images = make_trajectory_images(
                self.env_id, trajectory, trajectory.shape[1], None, None, False
            )
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
        self.interact(
            batch_size, conditions, namespace
        )  # interact if environment is installation

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

    def process_segment_noise_levels(
        self,
        level_array: np.ndarray,
        sequence_dividing_factor: int,
        reduction_amount: Optional[int] = None,
    ) -> np.ndarray:
        plan_tokens = len(level_array)  # T
        assert plan_tokens % sequence_dividing_factor == 0, (
            f"Plan tokens must be divisible by sequence dividing factor, but got {plan_tokens} and {sequence_dividing_factor}"
        )
        segment_size = plan_tokens // sequence_dividing_factor

        # Work with a copy
        steps = [level_array.copy()]

        work_array = level_array.copy()

        non_zero_indices = np.where(work_array > 0)[0]
        if len(non_zero_indices) == 0:
            return np.expand_dims(level_array, 0)

        start_idx = non_zero_indices[0]
        end_idx = min(start_idx + segment_size, plan_tokens)

        if self.pyramid:
            local_horizon = end_idx - start_idx
            uncertainty_scale = getattr(self, "uncertainty_scale", 1)

            initial_levels = steps[0][start_idx:end_idx]
            base_val = initial_levels[0]
            indices = np.arange(local_horizon)

            while np.any(work_array[start_idx:end_idx] > 0):
                current_step_count = len(steps)

                target_levels = (
                    base_val
                    + indices * uncertainty_scale
                    - current_step_count * reduction_amount
                )
                target_levels = np.maximum(0, target_levels).astype(work_array.dtype)

                work_array[start_idx:end_idx] = np.minimum(
                    work_array[start_idx:end_idx], target_levels
                )

                step_to_add = work_array.copy()
                steps.append(step_to_add)

        else:
            while np.any(work_array[start_idx:end_idx] > 0):
                work_array[start_idx:end_idx] = np.maximum(
                    0, work_array[start_idx:end_idx] - reduction_amount
                )

                step_to_add = work_array.copy()
                steps.append(step_to_add)

        return np.stack(steps, axis=0)  # (M, T)

    def _construct_noise_levels(
        self,
        levels: np.ndarray,
        batch_size: int,
        stabilization: int = 0,
        pad_tokens: int = 0,
        include_final_token: bool = False,
        include_init_token: bool = False,
    ) -> torch.Tensor:
        """Build noise levels for diffusion inference. (batch, n_tokens) tensor

        This function builds the full noise schedule for a single diffusion step.

        Args:
            levels: Noise level schedule for plan tokens (b, plan_tokens) shape
            batch_size: Batch size
            stabilization: Noise level for parent obs token (typically 0-2)
            pad_tokens: Number of padding tokens
            include_final_token: Whether to include final_token (bidirectional mode)
            include_init_token: Whether to prepend init_token slot (pre-built format always False).

        Returns:
            Noise levels array (t, b)
        """
        tracer = self.tracer if hasattr(self, 'tracer') else None

        components = []
        components.append(
            np.full((batch_size, 1), stabilization, dtype=np.int64)
        )  # given parent_obs additional token
        components.append(levels)  # plan tokens


        components.append(
            np.full((batch_size, pad_tokens), self.sampling_timesteps, dtype=np.int64)
        )  # padding
        components = torch.from_numpy(np.concatenate(components, axis=1)).to(
            self.device
        )

        result = rearrange(components, "b t -> t b", b=batch_size)  # (n_tokens, b)

        # Validate result shape before returning
        assert result.ndim == 2, f"result.ndim={result.ndim}, expected 2"
        assert result.shape[1] == batch_size, (
            f"result.shape[1]={result.shape[1]}, expected batch_size={batch_size}"
        )

        return result


    def _generate_bidirectional_schedule(
        self,
        start_levels: np.ndarray,
        complete_denoising: bool = False,
    ) -> np.ndarray:
        """
        Generates the N-step denoising schedule for bidirectional search.
        Returns a tensor of shape (B, Steps, T) representing the sequence of noise levels.
        """
        # start_levels shape: (B, plan_tokens)  # (b, t)

        batch_size = start_levels.shape[0]
        current_levels = start_levels.copy()
        schedule = [current_levels.copy()]

        assert self.sampling_timesteps >= self.mctd_num_denoising_steps, (
            "sampling_timesteps must be greater than or equal to mctd_num_denoising_steps"
        )
        chunk_of_sampling_timesteps_for_one_denoising = (
            self.sampling_timesteps // self.mctd_num_denoising_steps
        )

        while True:
            # Process each batch to denoise ONE segment
            to_levels_list = []
            for b in range(batch_size):
                to_levels_b = self.process_segment_noise_levels(
                    current_levels[b],
                    self.sequence_dividing_factor,
                    reduction_amount=chunk_of_sampling_timesteps_for_one_denoising,
                )  # (m, t)
                to_levels_list.append(to_levels_b)  # (b, m, t)

            # Verify that all particles in the batch have the same number of steps (M)
            # assert all(len(steps) == len(to_levels_list[0]) for steps in to_levels_list), \
            #     f"Schedules in batch have inconsistent lengths: {[len(s) for s in to_levels_list]}"

            # Determine the maximum number of steps (M) in this segment across the batch
            max_m = max(len(steps) for steps in to_levels_list)

            # Pad schedules to max_m by repeating the last step
            for b in range(batch_size):
                if len(to_levels_list[b]) < max_m:
                    padding = np.tile(
                        to_levels_list[b][-1:], (max_m - len(to_levels_list[b]), 1)
                    )
                    to_levels_list[b] = np.concatenate(
                        [to_levels_list[b], padding], axis=0
                    )

            batch_steps = np.stack(to_levels_list, axis=1)  # (M, B, T)

            # Append subsequent steps (index 0 is current_levels which is already in schedule)
            for m in range(1, batch_steps.shape[0]):
                schedule.append(batch_steps[m].copy())

            current_levels = batch_steps[-1]

            if np.all(
                current_levels == 0
            ):  # all particles(every sequence) are denoised
                break

            if not complete_denoising:  # for expansion escape (not simulation)
                break

        return np.stack(schedule, axis=0).transpose(1, 0, 2)  # (B, TotalSteps, T)

    def parallel_plan(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        horizon: int,
        conditions: Optional[Any] = None,
        guidance_scale: Optional[int] = None,
        noise_level: Optional[np.ndarray] = None,
        plans: Optional[list] = None,
        prefix_len_list: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Parallel denoising planning with diffusion guidance.

        Performs iterative denoising refinement on pre-built plans from MCTS leaf nodes,
        using guidance signals to steer trajectories toward goals while respecting constraints.

        Args:
            start: (b, obs_dim) normalized observations at root
            goal: (b, obs_dim) normalized goal observations
            horizon: scalar, planning horizon in timesteps (must be divisible by frame_stack)
            conditions: optional conditioning information
            guidance_scale: (b,) torch.Tensor of guidance scales per parallel instance
            noise_level: (b, m, plan_tokens) numpy array of noise schedules
                - b: batch size (number of parallel instances)
                - m: number of denoising steps (M from bidirectional schedule)
                - plan_tokens: horizon // frame_stack tokens to be denoised
            plans: list of b pre-built plan tensors, each shape (n_tokens, 1, fs*c)
                - n_tokens: total tokens (including padding)
                - 1: batch dimension per leaf
                - fs*c: flattened observation representation

        Returns:
            plan_hist: (m+1, plan_tokens*fs, b, c) tensor of denoising histories
                - m: number of denoising steps (M from bidirectional schedule)
                - plan_tokens*fs: horizon in frames
                - b: batch size (number of parallel instances)
                - c: observation dimension

        Shape Flow Diagram:
        ==================

        Input Plans (from _build_plan_from_leaf per node):
            plans[0]: (n_tokens, 1, fs*c)
            plans[1]: (n_tokens, 1, fs*c)
            ...
            plans[b-1]: (n_tokens, 1, fs*c)

        After concatenation (line ~1118):
            plan_with_given_tokens: (n_tokens, b, fs*c)

        Denoising Loop (lines ~1135-1191):
            Input to diffusion_model.sample_step:
                plan_with_given_tokens: (n_tokens, b, fs*c)
                from_noise_levels: (n_tokens, b)  [from _construct_noise_levels]
                to_noise_levels: (n_tokens, b)    [from _construct_noise_levels]

            Output from diffusion_model.sample_step:
                sample: (n_tokens, b, fs*c)

            Extracted plan chunks at each step:
                plan_hist[0]: (plan_tokens, b, fs*c)  [initial]
                plan_hist[1]: (plan_tokens, b, fs*c)  [after step 0->1]
                ...
                plan_hist[m]: (plan_tokens, b, fs*c)  [after step m-1->m]

        Final Processing (lines ~1179-1203):
            After stacking:
                plan_hist: (m+1, plan_tokens, b, fs*c)

            After rearrange (line ~1180-1184):
                plan_hist: (m+1, plan_tokens*fs, b, c)

        Key Shape Relationships:
            - n_tokens = plan_tokens + 1 + pad_tokens
            - plan_tokens = horizon // frame_stack
            - m = noise_level.shape[1] - 1 (number of denoising steps)
            - batch_size = len(plans) (number of parallel MCTS instances)
        """
        # start and goal are normalized tensors of shape (b, obs_dim)

        batch_size = len(plans)

        # Validate that each plan has shape (n_tokens, 1, fs*c)
        # These are pre-built from _build_plan_from_leaf with:
        #   [prefix(prefix_len) | obs_parent(1) | noisy(plan_tokens-prefix_len) | padding]
        for i, plan in enumerate(plans):
            assert plan is not None, f"plans[{i}] is None"
            assert plan.ndim == 3, f"plans[{i}].ndim={plan.ndim}, expected 3"
            assert plan.shape[1] == 1, (
                f"plans[{i}].shape[1]={plan.shape[1]}, expected 1 (batch=1)"
            )
            # FIX: plan shape is (batch, n_tokens, x_stacked_shape[0]), not (batch, n_tokens*x_stacked_shape)
            assert plan.shape[2] == self.x_stacked_shape[0], (
                f"plans[{i}].shape[2]={plan.shape[2]}, expected {self.x_stacked_shape[0]} (x_stacked_shape[0])"
            )

        if start.dim() == 2 and start.shape[0] == batch_size:
            pass
        else:
            start = torch.cat([start] * batch_size, 0)  # (b, obs_dim)

        if goal.dim() == 2 and goal.shape[0] == batch_size:
            pass
        else:
            goal = torch.cat([goal] * batch_size, 0)  # (b, obs_dim)

        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        guidance_fn = lambda x: guidance.combined_guidance(self, x, goal, horizon, guidance_scale)

        assert horizon % self.frame_stack == 0, (
            "horizon must be a multiple of frame_stack"
        )

        plan_tokens = horizon // self.frame_stack  # t (tokens per plan)

        # [INSTRUMENTATION] Ensure tracer is initialized
        if self.tracer is None:
            from utils.tracer import Tracer, set_default_tracer
            import datetime
            _now = datetime.datetime.now()
            run_id = _now.strftime("%Y-%m-%d_%H-%M-%S")
            self.tracer = Tracer(
                run_id=run_id,
                purpose="first_token_denoising_diagnosis",
                log_dir="logs_memory_debug",
                extra_meta={
                    "env_id": getattr(self, 'env_id', 'unknown'),
                    "batch_size": batch_size,
                    "frame_stack": self.frame_stack,
                }
            )
            set_default_tracer(self.tracer)
            # Note: tracer context manager should be started externally in interact()
            # For now, we rely on DEBUG_MODE environment variable for logging control

        # Plans are always pre-built (n_tokens, 1, fs*c) format from _build_plan_from_leaf.
        # Each plan in the list has shape (n_tokens, 1, fs*c) where:
        #   - n_tokens: total token capacity with padding
        #   - 1: single batch per leaf node
        #   - fs*c: flattened observation representation
        assert plans is not None and len(plans) == batch_size, (
            "plans must be a list of pre-built tensors"
        )
        assert self.n_tokens >= plan_tokens + 1, (
            f"too long horizon (n_tokens={self.n_tokens} < plan_tokens+1={plan_tokens+1})"
        )
        pad_tokens = max(0, self.n_tokens - plan_tokens - 1)  # scalar: padding tokens
        use_init_token_in_noise = False
        use_final_token_in_noise = False

        # CRITICAL: Concatenate plans from list along batch dimension
        # Input: list of b plans, each shape (n_tokens, 1, fs*c)
        # Output: (n_tokens, b, fs*c)
        # This reshapes from [plan_0:(n_tokens,1,fs*c), plan_1:(n_tokens,1,fs*c), ...]
        # to concatenated form (n_tokens, b, fs*c)
        plan_with_given_tokens = torch.cat(plans, dim=1)  # (n_tokens, b, fs*c)

        # output plan_hist.shape: (m, plan_tokens*fs, b, c)

        def extract_plan_chunk(plan_tensor, plan_tokens, prefix_len_list):
            """
            Slice window_tokens from offset.

            Args:
                plan_tensor: (n_tokens, b, fs*c) full plan with padding
                plan_tokens: number of tokens in the plan
                prefix_len_list: list of prefix lengths for each batch

            Returns:
                plan_without_parent_obs: (plan_tokens, b, fs*c) extracted plan segment
            """
            plan_with_parent_obs = plan_tensor[:plan_tokens+1].detach().clone()

            # Advanced Indexing: skip the prefix_len index for each batch
            # B = plan_with_parent_obs.shape[1]
            # t_idx = torch.arange(plan_tokens, device=device).unsqueeze(1) # (T, 1)
            # prefixes = torch.tensor(prefix_len_list, device=device)        # (B,)
            # indices = t_idx + (t_idx >= prefixes).long()                  # (T, B)
            # return plan_with_parent_obs[indices, torch.arange(B, device=plan_tensor.device)]

            plan_without_parent_obs = torch.zeros((plan_tokens, *plan_tensor.shape[1:]), device=plan_tensor.device)
            
            for i, prefix_len in enumerate(prefix_len_list):
                plan_without_parent_obs[:, i] = torch.cat([plan_with_parent_obs[:prefix_len, i], plan_with_parent_obs[prefix_len+1:, i]], dim=0)
            
            return plan_without_parent_obs

        plan_hist = [
            extract_plan_chunk(
                plan_with_given_tokens, 
                plan_tokens,
                prefix_len_list,
            )
        ] # (plan_tokens, b, fs*c)
        
        # [MEMORY DEBUG] Log initial plan_hist allocation
        if self.profiler:
            self.profiler.snapshot(
                f"parallel_plan_start_batch{batch_size}",
                phase="plan_hist_init"
            )
            if self.debug_log_level >= 2:
                import sys
                print(f"[DEBUG] parallel_plan: Created initial plan_hist (batch={batch_size})", 
                      file=sys.stderr, flush=True)

        stabilization = 0

        tracer = self.tracer if hasattr(self, 'tracer') else None

        for m in range(noise_level.shape[1] - 1):
            # noise_level shape: (b, m, plan_tokens)
            # Iterating over m denoising steps
            from_noise_levels = self._construct_noise_levels(
                noise_level[:, m],  # (b, plan_tokens) noise level at step m
                batch_size,
                stabilization,
                pad_tokens,
            )  # (n_tokens, b) noise levels for all tokens
            to_noise_levels = self._construct_noise_levels(
                noise_level[:, m + 1],  # (b, plan_tokens) noise level at step m+1
                batch_size,
                stabilization,
                pad_tokens,
            )  # (n_tokens, b) noise levels for all tokens

            sample = self.diffusion_model.sample_step(
                plan_with_given_tokens,  # (n_tokens, b, fs*c)
                conditions,
                from_noise_levels,  # (n_tokens, b)
                to_noise_levels,  # (n_tokens, b)
                guidance_fn=guidance_fn,
            )  # (n_tokens, b, fs*c)

            # (Not Necessary) Update only tokens whose noise level is actively decreasing this step.
            # This preserves denoised_prefix (level=0) and obs_parent_token (level=0).
            update_mask = (from_noise_levels > to_noise_levels).unsqueeze(
                -1
            )  # (n_tokens, b, 1) broadcast mask

            plan_with_given_tokens = torch.where(
                update_mask, sample, plan_with_given_tokens
            )  # (n_tokens, b, fs*c)

            plan_hist.append(
                extract_plan_chunk(
                    plan_with_given_tokens,
                    plan_tokens,
                    prefix_len_list,
                ) # (plan_tokens, b, fs*c)
            )

        # Stack all denoising steps
        plan_hist = torch.stack(plan_hist)  # (m+1, plan_tokens, b, fs*c)
        
        # [MEMORY OPTIMIZATION] Keep only last N histories to save memory
        if self.max_plan_hist_keep > 0 and plan_hist.shape[0] > self.max_plan_hist_keep:
            plan_hist = plan_hist[-self.max_plan_hist_keep:]
        
        # Rearrange to expand tokens into frame stacks
        plan_hist = rearrange(
            plan_hist,
            "m t b (fs c) -> m (t fs) b c",
            fs=self.frame_stack,
        )  # (m+1, plan_tokens*fs, b, c)

        # Validate plan_hist shape before returning
        # m+1: number of denoising steps (length of noise_level schedule)
        # plan_tokens*fs: horizon in frames
        # b: batch size (number of parallel instances)
        # c: observation dimension
        assert plan_hist.ndim == 4, f"plan_hist.ndim={plan_hist.ndim}, expected 4"
        assert plan_hist.shape[2] == batch_size, (
            f"plan_hist.shape[2]={plan_hist.shape[2]}, expected batch_size={batch_size}"
        )

        return plan_hist  # (m+1, plan_tokens*fs, b, c)

    def _extract_node_trajectory(self, best_node: "TreeNode") -> np.ndarray:
        """
        Recursively extract trajectory from best_node to root via sim_state.
        Returns array of shape (num_nodes, 2) with position at each node level.
        """
        trajectory = []
        current_node = best_node
        
        # ── Logging extraction process ──
        tracer = get_tracer()
        node_trace_log = []

        while current_node is not None:
            # Extract position from sim_state if available
            if current_node.sim_state is not None:
                pos = current_node.sim_state["qpos"][:2]
                trajectory.append(pos)
            elif current_node.obs_pos is not None:
                trajectory.append(current_node.obs_pos)
            
            # ── Log each node in the trace ──
            if tracer:
                node_trace_log.append({
                    "node_id": id(current_node),
                    "node_depth": current_node.depth,
                    "has_sim_state": current_node.sim_state is not None,
                    "sim_state_qpos": current_node.sim_state["qpos"][:2].tolist() if current_node.sim_state else None,
                    "obs_pos": current_node.obs_pos.tolist() if hasattr(current_node.obs_pos, 'tolist') else None,
                    "parent_id": id(current_node._parent_node) if current_node._parent_node else None,
                })

            # Move to parent
            current_node = current_node._parent_node

        # Reverse so root is first
        if trajectory:
            trajectory = np.array(trajectory[::-1])
        else:
            trajectory = np.array([])
        
        # ── Log final trajectory ──
        if tracer:
            tracer.log(
                tag="bidir_mcts._extract_node_trajectory.details",
                data={
                    "trajectory_length": len(trajectory),
                    "node_trace": node_trace_log,
                },
                depth=1,
            )

        return trajectory

    def interact(
        self,
        batch_size: int,
        conditions: Optional[Any] = None,
        namespace: str = "validation",
    ) -> None:
        try:
            import gym
            import ogbench
            from stable_baselines3.common.vec_env import DummyVecEnv
            from algorithms.diffusion_forcing.env_manager import EnvironmentManager
        except ImportError:
            print(
                "d4rl import not successful, skipping environment interaction. Check d4rl installation."
            )
            return

        print("Interacting with environment... This may take a couple minutes.")

        # [TRACER SETUP] Initialize structured memory logger
        tracer = Tracer(
            purpose="memory_efficiency_diagnosis",
            log_dir="logs_memory_debug",
            extra_meta={
                "env_id": self.env_id,
                "batch_size": batch_size,
                "frame_stack": self.frame_stack,
                "episode_len": self.episode_len,
            }
        )
        set_default_tracer(tracer)
        self.tracer = tracer  # [INSTRUMENTATION] Make tracer accessible to other methods

        with tracer:
            # [MEMORY CLEANUP] Clear caches before interaction
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            use_diffused_action = False
            agent = None  # Initialize agent to None; will be set for antmaze

            # [ENV CACHING] Check if environment is cached and batch_size matches
            env_cache_key = f"{self.env_id}_{batch_size}"
            if (
                hasattr(self, "_cached_envs_forward")
                and hasattr(self, "_cached_env_key")
                and self._cached_env_key == env_cache_key
            ):
                # Reuse cached bidirectional environments
                envs_forward = self._cached_envs_forward
                envs_backward = self._cached_envs_backward
                agent = getattr(self, "_cached_agent", None)
                envs_forward.reset()
                envs_backward.reset()
                if not (self.env_id in OGBENCH_ENVS):
                    envs_forward.seed(self.interaction_seed)
                    envs_backward.seed(self.interaction_seed)
                # For backward compatibility
                envs = envs_forward
            else:
                # Create new bidirectional environments
                if self.env_id in OGBENCH_ENVS:
                    if "pointmaze" in self.env_id:
                        env_fns = [
                            lambda: ogbench.locomaze.maze.make_maze_env(
                                "point", "maze", maze_type=self.env_id.split("-")[1]
                            )
                        ] * batch_size
                        use_diffused_action = True
                    elif "antmaze" in self.env_id:
                        env_fns = [
                            lambda: ogbench.locomaze.maze.make_maze_env(
                                "ant", "maze", maze_type=self.env_id.split("-")[1]
                            )
                        ] * batch_size
                        from dql.main_Antmaze import hyperparameters
                        from dql.agents.ql_diffusion import Diffusion_QL as Agent

                        params = hyperparameters[self.dataset]
                        
                        # Create temporary env to get dimensions
                        _temp_env = DummyVecEnv(env_fns)
                        state_dim = _temp_env.observation_space.shape[0]
                        action_dim = _temp_env.action_space.shape[0]
                        max_action = float(_temp_env.action_space.high[0])
                        _temp_env.close()
                        
                        # [DEBUG] Print actual environment dimensions
                        import sys
                        print(f"[DEBUG] OGBench Environment Dimensions:", file=sys.stderr, flush=True)
                        print(f"  - observation_space.shape: ({state_dim},)", file=sys.stderr, flush=True)
                        print(f"  - state_dim (raw): {state_dim}", file=sys.stderr, flush=True)
                        print(f"  - state_dim * 2 (passed to Agent): {state_dim * 2}", file=sys.stderr, flush=True)
                        print(f"  - action_dim: {action_dim}", file=sys.stderr, flush=True)
                        print(f"[DEBUG] DQL variant expects: state_dim=29 (from variant.json)", file=sys.stderr, flush=True)
                        
                        agent = Agent(
                            state_dim=state_dim * 2,
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
                        if (
                            self.dataset == "antmaze-medium-navigate-v0"
                            or self.dataset == "antmaze-medium-stitch-v0"
                        ):
                            dql_folder = "antmaze-medium-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                        elif (
                            self.dataset == "antmaze-large-navigate-v0"
                            or self.dataset == "antmaze-large-stitch-v0"
                        ):
                            dql_folder = "antmaze-large-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                        elif (
                            self.dataset == "antmaze-giant-navigate-v0"
                            or self.dataset == "antmaze-giant-stitch-v0"
                        ):
                            dql_folder = "antmaze-giant-navigate-v0|exp|diffusion-ql|T-5|lr_decay|ms-offline|k-1|0|2|1.0|False|cql_antmaze|0.2|4.0|10"
                        else:
                            raise ValueError(f"Dataset {self.dataset} not supported")

                        import os

                        agent.load_model(
                            os.path.join(os.getcwd(), "dql", "results", dql_folder), id=200
                        )
                else:
                    env_fns = [lambda: gym.make(self.env_id)] * batch_size
                    agent = None

                # Create bidirectional environments using EnvironmentManager
                env_manager = EnvironmentManager(
                    env_id=self.env_id,
                    batch_size=batch_size,
                    task_id=self.task_id,
                    use_random_goals=self.use_random_goals_for_interaction,
                    debug=self.debug_log_level >= 1,
                )
                envs_forward, envs_backward = env_manager.create_bidirectional_envs(env_fns)
                
                # Cache the environments and manager
                self._cached_envs_forward = envs_forward
                self._cached_envs_backward = envs_backward
                self._cached_env_manager = env_manager
                self._cached_env_key = env_cache_key
                if agent is not None:
                    self._cached_agent = agent  # Cache agent if it was created
                
                # For backward compatibility, also cache single envs reference
                envs = envs_forward  # Default to forward environment
                self._cached_envs = envs

                # [MEMORY] Log after environment creation
                log_memory_stats(tracer, "interact.envs_created", step=0)

                if self.debug_log_level >= 1:
                    import sys
                    print(
                        f"[MEM] Created and cached new bidirectional environments (batch_size={batch_size})",
                        file=sys.stderr,
                        flush=True,
                    )

            # [ENV CACHING END] Environment setup complete (cached or new)

            terminate = False
            obs_mean = self.data_mean[: self.observation_dim]
            obs_std = self.data_std[: self.observation_dim]
            obs = envs_forward.reset()
            # Randomize the goal for each environment
            if (
                self.env_id in OGBENCH_ENVS
            ):  # OGBench goal setting is already done through set_task()
                pass
            else:
                if self.use_random_goals_for_interaction:
                    for env in envs_forward.envs:
                        env.set_target()

            obs = torch.from_numpy(obs).float().to(self.device)
            start = obs.detach()
            obs_normalized = (
                (obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]
            ).detach()

            if self.env_id in OGBENCH_ENVS:  # OGBench
                goal = np.vstack(
                    [envs_forward.reset_infos[i]["goal"] for i in range(len(envs_forward.reset_infos))]
                )
            else:
                goal = np.concatenate([[env.env._target] for env in envs_forward.envs])
            goal = torch.Tensor(goal).float().to(self.device)
            goal = torch.cat([goal, torch.zeros_like(goal)], -1)
            goal = goal[:, : self.observation_dim]
            goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()
            
            # ────────────────────────────────────────────────────────────────
            # Bidirectional Environment Setup
            # ────────────────────────────────────────────────────────────────
            # For tree2 (backward direction), set up environment to start from goal
            # and plan towards start. We do this by:
            # 1. Creating a goal_sim_state (heuristic goal state for tree2)
            # 2. Will be set explicitly when tree2 needs to execute
            goal_qpos = goal.cpu().numpy()[0, :2]  # (2,) - goal coordinates
            
            # Reset envs_backward to a known state
            envs_backward.reset()
            # envs_backward will be reused in rollouts with _set_sim_state as needed

            steps = 0
            loops = 0  # Loop counter for bidirectional MCTS planning
            episode_reward = np.zeros(batch_size)
            episode_reward_if_stay = np.zeros(batch_size)
            reached = np.zeros(batch_size, dtype=bool)
            first_reach = np.zeros(batch_size)

            trajectory = []  # actual trajectory

            # run mpc with diffused actions
            planning_time = []

            # ----------------------------------------------------------------
            # Bidirectional MCTS: initialize tree1/tree2 once before MPC loop.
            # These trees are maintained across MPC steps and expanded
            # alternately within each planning call.
            # ----------------------------------------------------------------
            horizon: int = int(self.episode_len * self.horizon_scale)
            _bidir_start_np = start.cpu().numpy()[:, : self.observation_dim]  # (b, obs_dim)
            _bidir_goal_np = goal.cpu().numpy()[:, : self.observation_dim]  # (b, obs_dim)
            # Capture initial physical state if available
            initial_sim_state = self._get_sim_state(envs_forward)
            assert initial_sim_state is not None, "Failed to capture initial sim state"
            assert np.allclose(initial_sim_state["qpos"][:2], _bidir_start_np[0][:2], atol=1e-5), \
                f"Physical start position {initial_sim_state['qpos'][:2]} does not match observation start position {_bidir_start_np[0][:2]}"

            # Derive heuristic goal simulation state from initial state
            goal_sim_state = {
                "qpos": initial_sim_state["qpos"].copy(),
                "qvel": np.zeros_like(initial_sim_state["qvel"]),  # Goal is assumed static
            }
            # Replace x, y coordinates with goal coordinates
            goal_sim_state["qpos"][:2] = _bidir_goal_np[0][:2]

            bidir_tree1 = self._init_mcts_tree(
                horizon,
                tag="bidir_mcts_from_start",
                root_obs=_bidir_start_np[0],
                root_sim_state=initial_sim_state,
            )
            bidir_tree2 = self._init_mcts_tree(
                horizon,
                tag="bidir_mcts_from_goal",
                root_obs=_bidir_goal_np[0],
                root_sim_state=goal_sim_state,
            )
            
            # ── Logging initialization ──
            tracer = get_tracer()
            if tracer:
                tracer.log(
                    tag="bidir_mcts.initialization",
                    data={
                        "tree1_root_obs": _bidir_start_np[0][:2].tolist(),
                        "tree1_root_sim_state_qpos": initial_sim_state["qpos"][:2].tolist(),
                        "tree2_root_obs": _bidir_goal_np[0][:2].tolist(),
                        "tree2_root_sim_state_qpos": goal_sim_state["qpos"][:2].tolist(),
                        "tree1_id": id(bidir_tree1.root_node),
                        "tree2_id": id(bidir_tree2.root_node),
                    },
                    step=0,
                    depth=0,
                )
            
            # Flag: 0 → expand tree1 next, 1 → expand tree2 next
            expanded_tree_idx: int = 0
            # Configurable meeting threshold (Euclidean distance in unnormalized obs space)
            _meeting_delta: float = getattr(self.cfg, "meeting_delta", 2.0)

            while not terminate and loops < self.val_max_loops:
                loops += 1
                planning_start_time = time.time()

                # [EXPANSION CHECK] Early termination if both trees are fully explored
                if not bidir_tree1.root_node.is_expandable_flag and \
                   not bidir_tree2.root_node.is_expandable_flag:
                    print("[INFO] Both trees fully explored (root nodes unexpandable). Terminating planning.")
                    terminate = True
                    break

                # Generate plan (start → goal)
                # _generate_plan_between_points has been inlined here.

                # ------------------------------------------------------------------
                # Bidirectional alternating MCTS planning
                # ------------------------------------------------------------------
                _start_np = start.cpu().numpy()[:, : self.observation_dim]  # (b, obs_dim)
                _goal_np = goal.cpu().numpy()[:, : self.observation_dim]  # (b, obs_dim)

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

                # Initialize infos dicts so {**infos1, **infos2} is safe even on the first step
                
                # ── Logging before tree expansion ──
                tracer = get_tracer()
                if tracer:
                    tree_to_expand = "tree1" if expanded_tree_idx == 0 else "tree2"
                    tracer.log(
                        tag="bidir_mcts.before_expansion",
                        data={
                            "loop_count": loops,
                            "expanded_tree": tree_to_expand,
                            "expanded_tree_idx": expanded_tree_idx,
                            "t1_leaf_count": len(t1_leaf_nodes),
                            "t2_leaf_count": len(t2_leaf_nodes),
                            "t1_root_id": id(bidir_tree1.root_node),
                            "t2_root_id": id(bidir_tree2.root_node),
                        },
                        step=loops,
                        depth=0,
                    )
                    
                    # ── Verify tree instance before expansion ──
                    active_tree = bidir_tree1 if expanded_tree_idx == 0 else bidir_tree2
                    tracer.log(
                        tag="bidir_mcts.tree_instance_check",
                        data={
                            "loop_count": loops,
                            "expanded_tree_idx": expanded_tree_idx,
                            "active_tree_id": id(active_tree),
                            "active_tree_root_id": id(active_tree.root_node),
                            "active_tree_tag": active_tree.tag,
                            "bidir_tree1_id": id(bidir_tree1),
                            "bidir_tree1_root_id": id(bidir_tree1.root_node),
                            "bidir_tree2_id": id(bidir_tree2),
                            "bidir_tree2_root_id": id(bidir_tree2.root_node),
                            "matches_tree1": active_tree is bidir_tree1,
                            "matches_tree2": active_tree is bidir_tree2,
                        },
                        step=loops,
                        depth=1,
                    )

                # Alternate expansion: one single_step per MPC iteration
                active_tree, expanded_node_infos = self._run_mcts_search(
                    bidir_tree1 if expanded_tree_idx == 0 else bidir_tree2,
                    horizon,
                    conditions,
                    _start_np,
                    _goal_np,
                    opposite_leaf_nodes=t2_leaf_nodes if expanded_tree_idx == 0 else t1_leaf_nodes,
                    single_step=True,
                )
                
                # ── Logging after tree expansion ──
                if tracer:
                    tracer.log(
                        tag="bidir_mcts.after_expansion",
                        data={
                            "loop_count": loops,
                            "expanded_node_count": len(expanded_node_infos),
                            "active_tree_id": id(active_tree.root_node),
                            "active_tree_tag": active_tree.tag,
                        },
                        step=loops,
                        depth=0,
                    )

                # Per-leaf MPC rollout: update obs_pos and sim_state for newly expanded leaves

                for info in expanded_node_infos.values():
                    parent_node: "TreeNode" = info["parent_node"]
                    _child: Optional["TreeNode"] = info.get("node")  # set by expand()
                    if _child is None:
                        continue

                    # Recompute plan tensor and denoised index range from stored plan_history
                    plan_hist_last: torch.Tensor = info["plan_history"][-1][-1]  # (t*fs, c)
                    plan_unnormalized: torch.Tensor = self._unnormalize_x(
                        plan_hist_last.unsqueeze(1)
                    )  # (t*fs, 1, c)

                    seg_size: int = active_tree.plan_tokens // self.sequence_dividing_factor
                    new_denoised_start: int = parent_node.depth * seg_size * self.frame_stack
                    new_denoised_end: int = (parent_node.depth + 1) * seg_size * self.frame_stack

                    # ── Pre-rollout logging ──
                    tracer = get_tracer()
                    if tracer:
                        tracer.log(
                            tag="bidir_mcts._rollout_leaf_plan.pre_rollout",
                            data={
                                "loop_count": loops,
                                "child_node_id": id(_child),
                                "child_depth": _child.depth,
                                "parent_node_id": id(parent_node),
                                "parent_depth": parent_node.depth,
                                "parent_sim_state_qpos": parent_node.sim_state["qpos"][:2].tolist() if parent_node.sim_state else None,
                                "new_denoised_start": new_denoised_start,
                                "new_denoised_end": new_denoised_end,
                                "active_tree_id": id(active_tree.root_node),
                                "active_tree_tag": active_tree.tag,
                            },
                            depth=1,
                        )

                    _new_sim_state = self._rollout_leaf_plan(
                        leaf_plan_unnormalized=plan_unnormalized,
                        new_denoised_start_idx=new_denoised_start,
                        new_denoised_end_idx=new_denoised_end,
                        agent=agent,
                        envs=envs_forward if active_tree is bidir_tree1 else envs_backward,
                        parent_sim_state=parent_node.sim_state,
                    )
                    assert _new_sim_state is not None, "_new_sim_state is None"
                    _child.sim_state = _new_sim_state
                    _child.obs_pos = _new_sim_state["qpos"][:2]
                    
                    # ── Post-rollout logging ──
                    if tracer:
                        tracer.log(
                            tag="bidir_mcts._rollout_leaf_plan.post_rollout",
                            data={
                                "loop_count": loops,
                                "child_node_id": id(_child),
                                "child_obs_pos": _child.obs_pos.tolist() if hasattr(_child.obs_pos, 'tolist') else [float(x) for x in _child.obs_pos],
                                "child_sim_state_qpos": _child.sim_state["qpos"][:2].tolist() if _child.sim_state else None,
                                "parent_qpos": parent_node.sim_state["qpos"][:2].tolist() if parent_node.sim_state else None,
                                "displacement": (
                                    (float(_child.sim_state["qpos"][0] - parent_node.sim_state["qpos"][0]),
                                     float(_child.sim_state["qpos"][1] - parent_node.sim_state["qpos"][1]))
                                    if _child.sim_state and parent_node.sim_state else None
                                ),
                            },
                            depth=1,
                        )

                # Extract plan by selecting best leaf and combining plans
                best_info: dict = self._select_best_leaf(expanded_node_infos)

                # [GUARANTEE] With is_expandable_flag and select() filtering:
                # - Root unexpandable → terminate at loop start
                # - Root expandable → at least 1 expandable child exists
                # - select() only considers expandable children
                # - Therefore expanded_node_infos is never empty
                # - Therefore best_info is never None
                assert best_info is not None, (
                    "best_info should never be None: root is expandable, "
                    "select() must have found an expandable child"
                )
                
                best_node: "TreeNode" = best_info["node"]
                
                # ── Logging best_node selection ──
                tracer = get_tracer()
                if tracer:
                    # Trace back to root to find which tree this belongs to
                    curr = best_node
                    depth_count = 0
                    while curr._parent_node is not None:
                        depth_count += 1
                        curr = curr._parent_node
                    belongs_to_tree1 = (curr is bidir_tree1.root_node)
                    
                    tracer.log(
                        tag="bidir_mcts.best_node_selected",
                        data={
                            "loop_count": loops,
                            "best_node_id": id(best_node),
                            "best_node_depth": best_node.depth,
                            "best_node_obs_pos": best_node.obs_pos.tolist() if hasattr(best_node.obs_pos, 'tolist') else [float(x) for x in best_node.obs_pos],
                            "best_node_sim_state_qpos": best_node.sim_state["qpos"][:2].tolist() if best_node.sim_state else None,
                            "belongs_to_tree1": belongs_to_tree1,
                            "tree1_root_id": id(bidir_tree1.root_node),
                            "tree2_root_id": id(bidir_tree2.root_node),
                        },
                        step=loops,
                        depth=0,
                    )

                output_plan = self._extract_output_plan(
                    best_node,
                    plan_tokens=active_tree.plan_tokens,
                    is_tree1=(expanded_tree_idx == 0),
                )  # (T_combined*fs, 1, c)

                plan_hist = output_plan.unsqueeze(0)  # (1, T_combined*fs, 1, c)

                plan_unnormalized = self._unnormalize_x(plan_hist)[-1]  # (T_combined*fs, 1, c)

                # Flip for the next MPC step to alternate trees
                expanded_tree_idx = (expanded_tree_idx + 1) % 2

                # Visualization with both forward and reverse trajectories
                start_numpy = start.cpu().numpy()[:, :2]
                goal_numpy = goal.cpu().numpy()[:, :2]

                # Extract best_node's tree trajectory (sim_state sequence from root to leaf)
                node_trajectory = self._extract_node_trajectory(best_node)
                
                # ── Logging node_trajectory details ──
                if tracer and node_trajectory is not None and len(node_trajectory) > 0:
                    tracer.log(
                        tag="bidir_mcts.node_trajectory_extracted",
                        data={
                            "loop_count": loops,
                            "trajectory_length": len(node_trajectory),
                            "trajectory_first_pos": node_trajectory[0].tolist(),
                            "trajectory_last_pos": node_trajectory[-1].tolist(),
                            "trajectory_all_positions": node_trajectory.tolist(),
                            "start_goal_positions": {
                                "start": start_numpy[0].tolist(),
                                "goal": goal_numpy[0].tolist(),
                            },
                        },
                        step=loops,
                        depth=1,
                    )

                # Create forward trajectory image with both plan (red) and node trajectory (blue)
                plan_positions = plan_unnormalized[:, :, :2].detach().cpu().numpy()

                # Prepare trajectories dict for visualization
                trajectories_dict = {
                    'plan': plan_positions,  # Red
                    'node_path': node_trajectory if node_trajectory is not None and len(node_trajectory) > 0 else None,  # Blue
                }

                forward_image = make_trajectory_images(
                    self.env_id,
                    trajectories_dict,
                    1,
                    start_numpy,
                    goal_numpy,
                    self.plot_end_points,
                )[0]
                self.log_image(
                    f"plan/plan_at_{steps}", Image.fromarray(forward_image)
                )

                # Create reverse trajectory image (swap start and goal for visualization)
                
                planning_end_time = time.time()
                planning_time.append(planning_end_time - planning_start_time)

                

                # jumpy case (fill the gap)
                if self.jump > 1:
                    _plan = []
                    for t in range(plan_unnormalized.shape[0]):
                        for j in range(self.jump):
                            _plan.append(plan_unnormalized[t, :, :2])
                    plan_unnormalized = torch.stack(_plan)


                
                # Prepare obs for environment execution
                obs_numpy = obs.detach().cpu().numpy()

                # Use unified plan execution function
                trajectory_exec, reward_dict = self._execute_plan_in_env(
                    plan_frame_format=plan_unnormalized,
                    envs=envs,
                    agent=agent if "antmaze" in self.env_id else None,
                    use_diffused_action=use_diffused_action,
                )

                # Process returned rewards and trajectory
                reached = np.logical_or(reached, reward_dict["reached"])
                episode_reward += reward_dict["episode_reward"]
                episode_reward_if_stay += reward_dict["episode_reward_if_stay"]
                first_reach += reward_dict["first_reach"]

                # Check if episode terminated
                if (reward_dict["reached"] >= 1.0).any():
                    terminate = True

                # [MEMORY CLEANUP] Clear caches after plan execution
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Process trajectory
                if trajectory_exec is not None:
                    trajectory.extend(trajectory_exec)
                    steps += len(trajectory_exec)

                    # Update obs and obs_normalized for next planning iteration (if not terminated)
                    if not terminate and len(trajectory_exec) > 0:
                        # Get final obs from last trajectory bundle
                        final_bundle = trajectory_exec[-1]
                        # Extract obs from bundle (bundle = [obs, action, reward])
                        obs, _, _ = self.split_bundle(final_bundle)
                        obs = obs.to(self.device)
                        obs_normalized = (
                            (obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]
                        ).detach()
            self.log(f"{namespace}/planning_time", np.sum(planning_time))
            self.log(f"{namespace}/episode_reward", episode_reward.mean())
            self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
            self.log(f"{namespace}/first_reach", first_reach.mean())
            self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0) / batch_size)

            # Visualization
            # samples = min(16, batch_size)
            samples = min(32, batch_size)
            trajectory = torch.stack(trajectory)
            start = start[:, :2].cpu().numpy().tolist()
            goal = goal[:, :2].cpu().numpy().tolist()
            images = make_trajectory_images(
                self.env_id, trajectory, samples, start, goal, self.plot_end_points
            )

            for i, img in enumerate(images):
                self.log_image(
                    f"{namespace}_interaction/sample_{i}",
                    Image.fromarray(img),
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
        """
        Split bundle into [obs, action, reward?] components.

        Handles both token-format obs (stacked with frame_stack) and non-stacked obs.
        Automatically detects which format based on bundle size.
        """
        # Determine obs dimension: either self.observation_dim or self.x_stacked_shape[0]
        bundle_size = bundle.shape[-1]

        # Check if bundle contains frame-stacked obs (larger dimension)
        if bundle_size == self.x_stacked_shape[0] + self.action_dim + (1 if self.use_reward else 0):
            # Frame-stacked obs format: (batch, x_stacked_shape[0] + action_dim + reward?)
            obs_dim = self.x_stacked_shape[0]
        elif bundle_size == self.observation_dim + self.action_dim + (1 if self.use_reward else 0):
            # Non-stacked obs format: (batch, observation_dim + action_dim + reward?)
            obs_dim = self.observation_dim
        else:
            # Fallback: try to infer from bundle_size
            # Assume: bundle_size = obs_dim + action_dim + (1 if reward else 0)
            remainder = bundle_size - self.action_dim - (1 if self.use_reward else 0)
            if remainder > 0:
                obs_dim = remainder
            else:
                # Last resort: use x_stacked_shape[0]
                obs_dim = self.x_stacked_shape[0]

        if self.use_reward:
            return torch.split(bundle, [obs_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [obs_dim, self.action_dim], -1)
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

    def _generate_noise_levels(
        self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise_levels = super()._generate_noise_levels(xs, masks)
        _, batch_size, *_ = xs.shape

        # first frame is almost always known, this reflect that
        if random() < 0.5:
            noise_levels[0] = torch.randint(
                0, self.timesteps // 4, (batch_size,), device=xs.device
            )

        return noise_levels

    def visualize_node_value_plans(
        self, is_achieved_plan, search_num, values, names, plans, starts, goals, tag="mcts_plan"
    ):
        # plans: (t fs) b c

        batch_size = plans.shape[1]

        if starts.ndim == 1:
            starts = starts[None, :]
        if goals.ndim == 1:
            goals = goals[None, :]

        if starts.shape[0] == 1:
            starts = np.repeat(starts, batch_size, axis=0)
        if goals.shape[0] == 1:
            goals = np.repeat(goals, batch_size, axis=0)
        plans = self._unnormalize_x(plans)
        plan_obs, _, _ = self.split_bundle(
            plans
        )  # (t fs) b c -> [(t fs) b c1, (t fs) b c2, (t fs) b c3]
        
        plan_obs = plan_obs.detach().cpu().numpy()  # (t fs) b 2

        if plan_obs.ndim == 2:
            # (t fs) c -> (t fs) 1 c
            plan_obs = plan_obs[:, None, :]

        plan_images = make_trajectory_images(
            self.env_id,
            plan_obs,
            batch_size,
            starts,
            goals,
            self.plot_end_points,
        )
        for i in range(len(plan_images)):
            img = plan_images[i]
            self.log_image(
                # f"{tag}/{search_num+i+1}_{names[i]}_V{values[i]}",
                f"{tag}/{names[i]}/{'achieved' if is_achieved_plan[i] else 'not_achieved'}",
                Image.fromarray(img),
            )

    def calculate_values(self, plans, starts, goals):
        # plans: (sliced_tokens*fs, b, c)

        if plans.shape[1] != starts.shape[0]:  # b
            starts = starts.repeat(plans.shape[1], axis=0)  # (b, c1)
        if plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(plans.shape[1], axis=0)

        state_len = plans.shape[0]
        batch_size = plans.shape[1]
        plans = self._unnormalize_x(plans)
        obs, _, _ = self.split_bundle(
            plans
        )  # (t fs) b c -> [(t fs) b c1, (t fs) b c2, (t fs) b c3]
        obs = obs.detach().cpu().numpy()  # (t fs) b 2
        values = np.zeros(batch_size)
        infos = np.array(["NotReached"] * batch_size)  # b
        achieved_ts = np.array([None] * batch_size) # b
        for t in range(state_len):  # (t fs)
            if t == 0:
                pos_diff = np.linalg.norm(obs[t] - starts, axis=-1)  # b c1 -> b
            else:
                pos_diff = np.linalg.norm(obs[t] - obs[t - 1], axis=-1)
            infos[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = (
                "Warp"  # batch-wise indexing
            )
            values[(pos_diff > self.warp_threshold) * (infos == "NotReached")] = 0
            diff_from_goal = np.linalg.norm(obs[t] - goals, axis=-1) # b
            values[(diff_from_goal < self.meeting_delta) * (infos == "NotReached")] = (
                plans.shape[0] - t
            ) / plans.shape[0]
            achieved_ts[
                (diff_from_goal < self.meeting_delta) * (infos == "NotReached")
            ] = t
            infos[(diff_from_goal < self.meeting_delta) * (infos == "NotReached")] = (
                "Achieved"
            )

        return values, infos, achieved_ts

    def calculate_values_bidir(
        self,
        expanded_node_candidates: List[dict],
        final_best_plans: torch.Tensor, # (plan_tokens*fs, b, c)
        tree: "MCTSTreeState",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-node values for bidirectional search by pairing current plan with
        the target opposite-tree leaf node's plan.

        For each expanded candidate i:
          1. Slice plan_A from final_best_plans (current tree, depth-based length).
          2. Slice plan_B from target_node.plan_history (opposite tree, depth-based length).
          3. Flip plan_B and concatenate: [plan_A_sliced | flip(plan_B_sliced)].
          4. Delegate Warp/Achieved detection to calculate_values.

        Args:
            expanded_node_candidates: List of candidate dicts (each has 'parent_node').
            final_best_plans: Tensor of shape (T_total*fs, B, c) — fully denoised plan hists.
            tree: MCTSTreeState for the current tree (provides plan_tokens, sequence_dividing_factor).

        Returns:
            values: np.ndarray shape (B,)
            infos:  np.ndarray shape (B,), dtype str
            achieved_ts: np.ndarray shape (B,)
        """
        seg_size: int = tree.plan_tokens // self.sequence_dividing_factor
        B: int = len(expanded_node_candidates)

        values: np.ndarray = np.zeros(B)
        achieved_infos: np.ndarray = np.array(["NotReached"] * B)
        achieved_ts: np.ndarray = np.array([None] * B)

        for i, candidate in enumerate(expanded_node_candidates): # b
            parent_node: "TreeNode" = candidate["parent_node"]
            target_node: Optional["TreeNode"] = candidate["target_node"]

            assert target_node is not None, (
                f"[BiDir Value] parent_node '{parent_node.name}' has no target_node. "
                "target_node must be set by _select_dynamic_goal() before value calculation."
            )
            # FIX: target_node might be a root node of goal tree without plan_history, which is valid
            # The plan_history is not actually used in the value calculation below
            # assert target_node.plan_history, (
            #     f"[BiDir Value] target_node '{target_node.name}' has empty plan_history."
            # )

            # --- Plan A: current tree's denoised plan, sliced to parent depth --- #
            # final_best_plans: (T_total*fs, B, c) -> last denoising step for candidate i
            plan_a_full: torch.Tensor = final_best_plans[:, i]  # (T_total*fs, c)
            parent_seq_len: int = parent_node.depth * seg_size * self.frame_stack
            candid_seq_len: int = (parent_node.depth+1) * seg_size * self.frame_stack
            # Clamp candid_seq_len to actual plan length
            max_seq_len: int = min(final_best_plans.shape[0], candid_seq_len)
            plan_a_sliced: torch.Tensor = plan_a_full[parent_seq_len: max_seq_len].unsqueeze(1)  # (A_len, 1, c)

            import sys
            print(f"[DEBUG BiDir] Candidate {i}: parent_seq_len={parent_seq_len}, candid_seq_len={candid_seq_len}, max_seq_len={max_seq_len}, plan_a_sliced.shape={plan_a_sliced.shape}", file=sys.stderr, flush=True)


            # --- Delegate Warp/Achieved detection to calculate_values --- #
            # start = parent_node's physical position, goal = target_node's physical position
            start_np: np.ndarray = parent_node.obs_pos[
                None, : self.observation_dim
            ]  # (1, obs_dim)
            goal_np: np.ndarray = target_node.obs_pos[
                None, : self.observation_dim
            ]  # (1, obs_dim)
            # Always evaluate as forward (plan_A is forward, plan_B already flipped)
            _vals, _achieved_infos, _achieved_ts = self.calculate_values(
                plan_a_sliced, start_np, goal_np
            )
            values[i] = _vals[0]
            achieved_infos[i] = _achieved_infos[0]
            # Handle case where goal was not achieved (_achieved_ts[0] is None)
            if _achieved_ts[0] is not None:
                achieved_ts[i] = _achieved_ts[0] + parent_seq_len
            else:
                # Goal not reached, set to -1 or plan length
                achieved_ts[i] = -1

            import sys
            print(f"[DEBUG BiDir Values] Candidate {i}: value={_vals[0]:.4f}, info={_achieved_infos[0]}, ts={achieved_ts[i]}", file=sys.stderr, flush=True)

        return values, achieved_infos, achieved_ts

    def _init_mcts_tree(
        self,
        horizon: int,
        tag: str,
        is_tree1: bool = True,
        root_obs: Optional[np.ndarray] = None,
        root_sim_state: Optional[dict] = None,
    ) -> "MCTSTreeState":
        """
        (A function) Initialize a single MCTS tree and return its full state.

        Args:
            horizon: Planning horizon (must be divisible by frame_stack)
            tag: Tag string for tqdm progress bar labeling
            root_obs: Unnormalized root observation, shape (obs_dim,).
                      Stored in root_node.obs_pos and tree.tree_root_obs.

        Returns:
            MCTSTreeState: Fully initialized tree state ready for _run_mcts_search
        """
        plan_tokens: int = horizon // self.frame_stack  # t
        children_node_guidance_scales: list = self.mctd_guidance_scales
        max_search_num: int = self.mctd_max_search_num
        num_denoising_steps: int = self.mctd_num_denoising_steps
        skip_level_steps: int = self.mctd_skip_level_steps

        assert plan_tokens <= self.n_tokens - 1, (
            f"Plan tokens must be <= {self.n_tokens - 1}, but got {plan_tokens}"
        )
        terminal_depth: int = self.sequence_dividing_factor
        noise_level: Optional[np.ndarray] = None  # bidirectional uses dynamic schedule

        # current_levels: (1, plan_tokens) — excludes init/final tokens
        root_current_levels: np.ndarray = np.full(
            (1, plan_tokens), self.sampling_timesteps, dtype=np.int64
        )  # (1, t)

        root_node = TreeNode(
            "0",
            0,
            None,
            children_node_guidance_scales,
            [],
            terminal_depth=terminal_depth,
            virtual_visit_weight=self.virtual_visit_weight,
            current_levels=root_current_levels,
            obs_pos=root_obs,
            sim_state=root_sim_state,
        )
        root_node.set_value(0)  # Initialize the value of the root node

        pbar = tqdm(
            total=max_search_num,
            desc=f"MCTS Search ({tag})",
            leave=False,
            dynamic_ncols=True,
        )

        return MCTSTreeState(
            root_node=root_node,
            plan_tokens=plan_tokens,
            terminal_depth=terminal_depth,
            noise_level=noise_level,
            children_node_guidance_scales=children_node_guidance_scales,
            max_search_num=max_search_num,
            num_denoising_steps=num_denoising_steps,
            skip_level_steps=skip_level_steps,
            tag=tag,
            is_tree1=is_tree1,
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
    ) -> tuple[MCTSTreeState, dict[str, dict]]:
        """
        (B function) Run the MCTS search loop for a given tree state.

        When `single_step=False` (default), runs until max_search_num or time_limit.
        When `single_step=True`, executes exactly one Selection→Expansion→Simulation→
        Backpropagation→EarlyTermination cycle and returns.

        In bidirectional mode, `opposite_leaf_positions` provides the leaf positions
        from the other tree so that dynamic goal selection can be performed via HILP.

        Args:
            tree: MCTSTreeState initialized by _init_mcts_tree
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
        terminal_depth = tree.terminal_depth

        # Variable to hold expanded_node_updated_levels across the loop
        expanded_node_updated_levels: Optional[np.ndarray] = None

        # Holds the expanded node infos from the latest iteration (reset each iteration)
        expanded_node_infos: dict = {}
        
        # [MEMORY DEBUG] Log tree search start
        if self.profiler:
            self.profiler.snapshot(
                f"mcts_search_start_{tree.tag}",
                phase=f"tree_search_init({tree.tag})"
            )

        # [LOGGING] Record search start
        from utils.tracer import get_tracer
        tracer = get_tracer()
        if tracer:
            with tracer.scope("mcts_search", phase="search"):
                tracer.log(
                    tag="tree.search.start",
                    data={
                        "tree_tag": tree.tag,
                        "terminal_depth": tree.terminal_depth,
                        "max_search_num": tree.max_search_num,
                        "plan_tokens": tree.plan_tokens,
                    },
                    step=0,
                    depth=0,
                )

        while True:
            # [FINE-GRAIN LOGGING] Log loop condition check
            if tracer:
                tracer.log(
                    tag="tree.search.loop_check",
                    data={
                        "search_num": tree.search_num,
                        "p_search_num": tree.p_search_num,
                        "max_search_num": tree.max_search_num,
                        "time_limit": self.time_limit,
                    },
                    step=tree.search_num,
                    depth=1,
                )

            if self.time_limit is not None:
                if time.time() - self.start_time > self.time_limit:
                    if tracer:
                        tracer.log(
                            tag="tree.search.loop_break",
                            data={"reason": "time_limit_exceeded"},
                            step=tree.search_num,
                            depth=0,
                        )
                    break
            else:
                # if search_num >= max_search_num:
                if tree.p_search_num >= tree.max_search_num:
                    if tracer:
                        tracer.log(
                            tag="tree.search.loop_break",
                            data={"reason": "max_search_num_reached", "p_search_num": tree.p_search_num, "max_search_num": tree.max_search_num},
                            step=tree.search_num,
                            depth=0,
                        )
                    break

            ## For checking the virtual visit count
            # root_node.check_virtual_visit_count()

            # [LOGGING] Log iteration stats
            if tracer and (tree.search_num % 5 == 0):  # Log every 5 iterations
                tracer.log(
                    tag="tree.expansion.iteration",
                    data={
                        "search_num": tree.search_num,
                        "max_depth": tree.max_depth,
                        "p_search_num": tree.p_search_num,
                    },
                    step=tree.search_num,
                    depth=1,
                )

            # [MEMORY DEBUG] Periodic memory logging
            if self.profiler and (tree.search_num > 0) and (tree.search_num % self.debug_log_interval == 0):
                self.profiler.snapshot(
                    f"mcts_search_iter_{tree.search_num}_{tree.tag}",
                    phase=f"mcts_iter_{tree.search_num}"
                )

            if self.debug_log_level >= 2:
                import sys
                print(f"[DEBUG] MCTS search {tree.tag}: iteration {tree.search_num}/{tree.max_search_num}", 
                      file=sys.stderr, flush=True)

            ###############################
            # Selection
            #  When leaf parallelization is True, then the selection is done in partially parallel (the children nodes from same parent node are selected at the same time)
            #  When leaf parallelization is False, then the selection is done in fully sequential (only one node is selected at a time)

            # [FINE-GRAIN LOGGING] Log expandable nodes at iteration start
            if not self.parallel_multiple_visits:  # If parallel multiple visits is False, then we need to list all the nodes to expand
                expandable_node_names = root_node.get_expandable_node_names()
                # print(f"Expandable node names: {expandable_node_names}")

            if tracer:
                expandable_count = len(expandable_node_names) if not self.parallel_multiple_visits else "unknown"
                tracer.log(
                    tag="tree.node.expandability_check",
                    data={
                        "search_num": tree.search_num,
                        "max_depth": tree.max_depth,
                        "expandable_node_count": expandable_count,
                    },
                    step=tree.search_num,
                    depth=1,
                )

            selection_start_time = time.time()
            print("============ Selection Start ============")
            psn = self.parallel_search_num
            selected_nodes, expanded_node_candidates = [], []
            while psn > 0:
                selected_node = root_node

                # [FINE-GRAIN LOGGING] Log node traversal with child status details
                if tracer:
                    # Get child node status
                    children_info = []
                    for i, child_dict in enumerate(selected_node._children_nodes):
                        children_info.append({
                            "child_idx": i,
                            "node_exists": child_dict["node"] is not None,
                            "virtually_visited": child_dict.get("virtually_visited", False),
                            "guidance_scale": child_dict.get("guidance_scale"),
                        })

                    tracer.log(
                        tag="tree.node.traversal_start",
                        data={
                            "node_name": selected_node.name,
                            "node_depth": selected_node.depth,
                            "is_expandable": selected_node.is_expandable(consider_virtually_visited=(not self.parallel_multiple_visits)),
                            "is_expandable_without_vv": selected_node.is_expandable(consider_virtually_visited=False),
                            "is_terminal": selected_node.is_terminal(),
                            "is_selectable": selected_node.is_selectable(),
                            "visit_count": selected_node.visit_count,
                            "children_status": children_info,
                        },
                        step=tree.search_num,
                        depth=1,
                    )

                while (
                    (
                        not selected_node.is_expandable(
                            consider_virtually_visited=(
                                not self.parallel_multiple_visits
                            )
                        )
                    )
                    and (not selected_node.is_terminal())
                    and (selected_node.is_selectable())
                ):
                    selected_node = selected_node.select(
                        leaf_parallelization=self.leaf_parallelization
                    )

                    # [FINE-GRAIN LOGGING] Log selected child node
                    if tracer:
                        tracer.log(
                            tag="tree.node.selected_child",
                            data={
                                "parent_name": selected_node._parent_node.name if selected_node._parent_node else "root",
                                "child_name": selected_node.name,
                                "child_depth": selected_node.depth,
                                "is_expandable": selected_node.is_expandable(consider_virtually_visited=(not self.parallel_multiple_visits)),
                                "is_terminal": selected_node.is_terminal(),
                                "is_selectable": selected_node.is_selectable(),
                                "visit_count": selected_node.visit_count,
                            },
                            step=tree.search_num,
                            depth=1,
                        )

                # [FINE-GRAIN LOGGING] Log final selected node before expansion check
                is_term = selected_node.is_terminal()
                is_exp = selected_node.is_expandable(consider_virtually_visited=(not self.parallel_multiple_visits))
                is_sel = selected_node.is_selectable()

                if tracer:
                    tracer.log(
                        tag="tree.node.final_selection_check",
                        data={
                            "node_name": selected_node.name,
                            "node_depth": selected_node.depth,
                            "is_terminal": is_term,
                            "is_expandable": is_exp,
                            "is_selectable": is_sel,
                            "will_expand": is_exp,
                            "will_skip": is_term or (not is_sel and not is_exp),
                        },
                        step=tree.search_num,
                        depth=1,
                    )

                if is_term or (not is_sel and not is_exp):
                    psn -= (
                        1
                        if not self.leaf_parallelization
                        else len(children_node_guidance_scales)
                    )
                    if tracer:
                        tracer.log(
                            tag="tree.node.skip_reason",
                            data={
                                "node_name": selected_node.name,
                                "is_terminal": is_term,
                                "not_selectable_and_not_expandable": (not is_sel and not is_exp),
                            },
                            step=tree.search_num,
                            depth=1,
                        )
                    continue
                if self.leaf_parallelization:
                    # [FINE-GRAIN LOGGING] Log leaf parallelization expansion
                    if tracer:
                        tracer.log(
                            tag="tree.expansion.leaf_parallelization_block",
                            data={
                                "node_name": selected_node.name,
                                "num_guidance_scales": len(children_node_guidance_scales),
                            },
                            step=tree.search_num,
                            depth=1,
                        )

                    for i in range(len(children_node_guidance_scales)):
                        # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                        expanded_node_candidate = (
                            selected_node.get_expandable_candidate(
                                index=i,
                                consider_virtually_visited=(
                                    not self.parallel_multiple_visits
                                ),
                            )
                        )

                        # [FINE-GRAIN LOGGING] Log candidate obtained
                        if tracer and expanded_node_candidate is not None:
                            tracer.log(
                                tag="tree.expansion.candidate_obtained",
                                data={
                                    "candidate_name": expanded_node_candidate["name"],
                                    "candidate_depth": expanded_node_candidate["depth"],
                                    "parent_name": expanded_node_candidate["parent_node"].name,
                                    "guidance_scale": expanded_node_candidate["guidance_scale"],
                                },
                                step=tree.search_num,
                                depth=1,
                            )

                        selected_nodes.append(selected_node)
                        expanded_node_candidates.append(expanded_node_candidate)
                        if not self.parallel_multiple_visits:
                            if (
                                not expanded_node_candidate["name"]
                                in expandable_node_names
                            ):
                                raise ValueError(
                                    f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names"
                                )
                            expandable_node_names.remove(
                                expanded_node_candidate["name"]
                            )
                        # print(f"Expanded node candidate {expanded_node_candidate['name']} is selected")
                        psn -= 1
                else:
                    # [FINE-GRAIN LOGGING] Log sequential expansion
                    if tracer:
                        tracer.log(
                            tag="tree.expansion.sequential_block",
                            data={
                                "node_name": selected_node.name,
                                "node_depth": selected_node.depth,
                            },
                            step=tree.search_num,
                            depth=1,
                        )

                    # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                    expanded_node_candidate = selected_node.get_expandable_candidate(
                        index=None,
                        consider_virtually_visited=(not self.parallel_multiple_visits),
                    )

                    # [FINE-GRAIN LOGGING] Log candidate obtained
                    if tracer and expanded_node_candidate is not None:
                        tracer.log(
                            tag="tree.expansion.candidate_obtained",
                            data={
                                "candidate_name": expanded_node_candidate["name"],
                                "candidate_depth": expanded_node_candidate["depth"],
                                "parent_name": expanded_node_candidate["parent_node"].name,
                                "guidance_scale": expanded_node_candidate["guidance_scale"],
                            },
                            step=tree.search_num,
                            depth=1,
                        )

                    selected_nodes.append(selected_node)
                    expanded_node_candidates.append(expanded_node_candidate)
                    if not self.parallel_multiple_visits:
                        if not expanded_node_candidate["name"] in expandable_node_names:
                            raise ValueError(
                                f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names"
                            )
                        expandable_node_names.remove(expanded_node_candidate["name"])
                    # print(f"Expanded node candidate {expanded_node_candidate['name']} is selected")
                    psn -= 1
                if not self.parallel_multiple_visits:
                    if len(expandable_node_names) == 0:
                        print("No more expandable nodes")
                        break
            if len(selected_nodes) == 0:
                print("No more selected nodes")
                # [FINE-GRAIN LOGGING] Log why loop breaks
                if tracer:
                    tracer.log(
                        tag="tree.search.termination_reason",
                        data={
                            "reason": "no_selected_nodes",
                            "search_num": tree.search_num,
                            "max_depth": tree.max_depth,
                            "p_search_num": tree.p_search_num,
                        },
                        step=tree.search_num,
                        depth=0,
                    )
                break
            print("============ Selection End ============")
            selection_end_time = time.time()
            tree.selection_time.append(selection_end_time - selection_start_time)

            # ------------------------------------------------------------------
            # Dynamic Start & Goal Selection for each expansion candidate
            # ------------------------------------------------------------------
            # Filter out nodes with uninitialized obs_pos
            valid_candidates = []
            for info in expanded_node_candidates:
                if info["parent_node"].obs_pos is not None:
                    valid_candidates.append(info)
                elif self.debug_log_level >= 1:
                    import sys
                    print(
                        f"[WARN] Parent node '{info['parent_node'].name}' has no obs_pos. Skipping.",
                        file=sys.stderr,
                        flush=True,
                    )

            # If no valid candidates (all had None obs_pos), skip diffusion and continue
            if not valid_candidates:
                if self.debug_log_level >= 1:
                    import sys
                    print(
                        f"[WARN] No valid expansion candidates (all nodes missing obs_pos). "
                        f"Skipping expansion round.",
                        file=sys.stderr,
                        flush=True,
                    )
                break  # Exit search loop

            eff_obs_norm_list, eff_goal_norm_list = [], []
            eff_start_np_list, eff_goal_np_list = [], []

            for info in valid_candidates:
                parent_node = info["parent_node"]
                parent_obs_pos = parent_node.obs_pos

                # Start: Normalized parent position for planning context
                eff_start_np_list.append(parent_obs_pos[None, : self.observation_dim])
                obs_mean_np = self.data_mean[: self.observation_dim].cpu().numpy() if isinstance(self.data_mean, torch.Tensor) else np.array(self.data_mean[: self.observation_dim])
                obs_std_np = self.data_std[: self.observation_dim].cpu().numpy() if isinstance(self.data_std, torch.Tensor) else np.array(self.data_std[: self.observation_dim])
                p_norm = torch.tensor(
                    (parent_obs_pos[: self.observation_dim] - obs_mean_np)
                    / obs_std_np,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                eff_obs_norm_list.append(p_norm)

                # Goal: Dynamic selection from opposite tree's leaf nodes
                assert (
                    opposite_leaf_nodes is not None and len(opposite_leaf_nodes) > 0
                ), "opposite_leaf_nodes is empty"
                target_node = self._select_dynamic_goal(
                    current_leaf_obs=parent_obs_pos,
                    opposite_leaf_nodes=opposite_leaf_nodes,
                )
                info["target_node"] = (
                    target_node  # Will be propagated to child TreeNode via expand()
                )
                target_pos = target_node.obs_pos

                eff_goal_np_list.append(target_pos[None, : self.observation_dim])
                obs_mean_np = self.data_mean[: self.observation_dim].cpu().numpy() if isinstance(self.data_mean, torch.Tensor) else np.array(self.data_mean[: self.observation_dim])
                obs_std_np = self.data_std[: self.observation_dim].cpu().numpy() if isinstance(self.data_std, torch.Tensor) else np.array(self.data_std[: self.observation_dim])
                g_norm = torch.tensor(
                    (target_pos[: self.observation_dim] - obs_mean_np) / obs_std_np,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                eff_goal_norm_list.append(g_norm)

            effective_obs_normalized = torch.cat(eff_obs_norm_list, dim=0)  # (B, D)
            effective_goal_normalized = torch.cat(eff_goal_norm_list, dim=0)  # (B, D)
            effective_starts_np = np.concatenate(eff_start_np_list, axis=0)  # (B, D)
            effective_goals_np = np.concatenate(eff_goal_np_list, axis=0)  # (B, D)

            filtered_expanded_node_plan_hists = [None] * len(
                valid_candidates
            )  # the elements can be left as None is every states are at the same point
            filtered_value_estimation_plan_hists = [None] * len(
                valid_candidates
            )

            assert tree.plan_tokens % self.sequence_dividing_factor == 0, (
                f"plan_tokens {tree.plan_tokens} is not divisible by sequence_dividing_factor {self.sequence_dividing_factor}"
            )

            seg_size = tree.plan_tokens // self.sequence_dividing_factor
            
            for _ in range(
                self.num_tries_for_bad_plans
            ):  # Trick used in MCTD to resample when the generated plan is terrible (e.g., not moving plans)
                ###############################
                # Expansion
                expansion_start_time = time.time()
                print("============ Expansion Start ============")
                expanded_node_plans = []
                expanded_node_noise_levels = []
                expanded_node_guidance_scales = []

                prefix_len_list = []
                for info in expanded_node_candidates:
                    # Build pre-built plan from leaf history (n_tokens(=t), 1, fs*c)
                    initial_plan, prefix_len = self._build_plan_from_leaf(
                        parent_node=info["parent_node"],
                        plan_tokens=tree.plan_tokens,
                        segment_size=seg_size,
                    )  # (n_tokens(=t), 1, fs*c)
                    expanded_node_plans.append(initial_plan)
                    expanded_node_guidance_scales.append(info["guidance_scale"])
                    prefix_len_list.append(prefix_len)

                expanded_node_guidance_scales = torch.tensor(
                    expanded_node_guidance_scales, device=self.device
                )  # (b,)

                # Build parent_levels: (b, plan_tokens(=t)) — noise state for each candidate
                parent_levels_list = []
                for info in expanded_node_candidates:
                    parent_node = info["parent_node"]
                    if parent_node.current_levels is not None:
                        parent_levels_list.append(parent_node.current_levels)
                    else:
                        assert horizon % self.frame_stack == 0, (
                            f"horizon {horizon} is not divisible by frame_stack {self.frame_stack}"
                        )
                        _plan_tokens = horizon // self.frame_stack
                        parent_levels_list.append(
                            np.full(
                                (1, _plan_tokens),
                                self.sampling_timesteps,
                                dtype=np.int64,
                            )
                        )

                parent_levels = np.concatenate(
                    parent_levels_list, axis=0
                )  # (b, plan_tokens(=t))

                # Generate bidirectional denoising schedule
                expanded_node_noise_levels = self._generate_bidirectional_schedule(
                    parent_levels, complete_denoising=False
                )  # (b, m, plan_tokens(=t))
                expanded_node_updated_levels = expanded_node_noise_levels[
                    :, -1, :
                ]  # (b, plan_tokens(=t))

                # Expansion: input plans (n_tokens(=t), 1, fs*c), output plan_hist # (m+1, plan_tokens*fs, b, c)
                expanded_node_plan_hists = self.parallel_plan(
                    start=effective_obs_normalized,
                    goal=effective_goal_normalized,
                    horizon=horizon,
                    conditions=conditions,
                    guidance_scale=expanded_node_guidance_scales,
                    noise_level=expanded_node_noise_levels,
                    plans=expanded_node_plans,
                    prefix_len_list=prefix_len_list,
                )

                # Validate expanded_node_plan_hists shape: (m, plan_tokens*fs, B, c)
                assert expanded_node_plan_hists.ndim == 4, (
                    f"expanded_node_plan_hists.ndim={expanded_node_plan_hists.ndim}, expected 4"
                )
                assert expanded_node_plan_hists.shape[2] == len(
                    expanded_node_candidates
                ), (
                    f"expanded_node_plan_hists.shape[2]={expanded_node_plan_hists.shape[2]}, expected {len(expanded_node_candidates)}"
                )

                if self.debug:
                    print(
                        f"  [DEBUG] [{tree.root_node.name}-Search] Expansion completed for {len(expanded_node_candidates)} nodes. plan_hists shape: {expanded_node_plan_hists.shape}"
                    )

                print(f"Expanded node plan hists: {expanded_node_plan_hists.shape}")
                print("============ Expansion End ============")
                expansion_end_time = time.time()
                tree.expansion_time.append(expansion_end_time - expansion_start_time)

                ###############################
                # Simulation
                #  It includes the noise level zero-padding, finding the max denoising steps, simulation, value calculation and node allocation
                simulation_start_time = time.time()
                import sys
                print(f"[DEBUG] Starting simulation phase for {len(expanded_node_candidates)} candidates", file=sys.stderr, flush=True)
                
                def is_feasible_plan_hists(plan_hists): # plan_hists: (m, plan_tokens*fs, b, c)
                    plans = (
                        self._unnormalize_x(plan_hists[-1])[:-1]
                        .detach()
                        .cpu()
                        .numpy()
                    )  # (t fs) b c
                    diffs = np.linalg.norm(
                        plans[1:] - plans[:-1], axis=-1
                    )  # (plan_len-1, b)

                    # FIX: is_feasible size should match diffs.shape[1] (number of plans), not expanded_node_candidates
                    is_feasible = [False] * diffs.shape[1]
                    for i in range(diffs.shape[1]):
                        is_feasible[i] = np.all(
                            diffs[:, i] < self.meeting_delta
                        )
                    return is_feasible

                if not self.mcts_use_sim:
                    # Skip simulation: use HILP value directly for expansion results
                    assert prefix_len_list is not None
                    batch_size = expanded_node_plan_hists.shape[2]
                    plans_tokens = rearrange(expanded_node_plan_hists, "m (t fs) b c -> m t fs b c", fs=self.frame_stack)
                    num_tokens_to_check = seg_size
                    t_rel_idx = torch.arange(num_tokens_to_check, device=self.device).view(-1, 1) # (T_check, 1)
                    prefixes = torch.tensor(prefix_len_list, device=self.device).view(1, -1) # (1, B)
                    token_indices = t_rel_idx + prefixes # (T_check, B)
                    
                    # Prevent out of range error by clamping to [0, T-1]
                    # FIX: Use plans_tokens.shape[1] instead of undefined plan_tokens variable
                    token_indices = torch.clamp(token_indices, max=plans_tokens.shape[1] - 1)
                    
                    batch_idx = torch.arange(batch_size, device=self.device)
                    sliced_hists = plans_tokens[:, token_indices, :, batch_idx]
                    processed_hists = rearrange(sliced_hists, "m t fs b c -> m (t fs) b c")

                    import sys
                    print(f"[DEBUG] Calling is_feasible_plan_hists with shape: {processed_hists.shape}", file=sys.stderr, flush=True)
                    is_feasible = is_feasible_plan_hists(processed_hists)
                    print(f"[DEBUG] is_feasible result: {is_feasible}", file=sys.stderr, flush=True)

                    for i in range(len(expanded_node_candidates)):
                        if is_feasible[i] and filtered_expanded_node_plan_hists[i] is None:
                            filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[
                                :, :, i
                            ]
                            # Create dummy value_estimation_plan_hists using expanded_node_plan_hists
                            filtered_value_estimation_plan_hists[i] = (
                                expanded_node_plan_hists[:, :, i]
                            )

                else: # [DEPRECATED] mcts_use_sim is True
                    print("============ Simulation Start ============")
                    # Pad the noise levels - Sequential
                    simul_noiselevel_zero_padding_start = time.time()
                    value_estimation_plans, value_estimation_noise_levels = [], []
                    max_denoising_steps = 0
                    for i in range(
                        len(expanded_node_candidates)
                    ):  # find the max denoising steps
                        # expanded_node_plan_hists: (m, plan_tokens*fs, b, c)
                        # Wrap plan to (n_tokens, 1, fs*c) for value estimation.
                        _plan_t_fs = expanded_node_plan_hists[-1, :, i].unsqueeze(
                            1
                        )  # (plan_tokens*fs, 1, c)
                        _plan_tokens_val = horizon // self.frame_stack
                        _plan_rearranged = rearrange(
                            _plan_t_fs, "(t fs) b c -> t b (fs c)", fs=self.frame_stack
                        )  # (plan_tokens, 1, fs*c)
                        _sim_pad_tokens = self.n_tokens - _plan_tokens_val - 1
                        _sim_pad = torch.zeros(
                            (_sim_pad_tokens, 1, _plan_rearranged.shape[-1]),
                            device=self.device,
                        )
                        value_estimation_plans.append(
                            torch.cat([_plan_rearranged, _sim_pad], dim=0)
                        )  # (n_tokens, 1, fs*c)

                    simul_noiselevel_zero_padding_end = time.time()
                    tree.simul_noiselevel_zero_padding_time.append(
                        simul_noiselevel_zero_padding_end
                        - simul_noiselevel_zero_padding_start
                    )

                    # Simulation - Value Estimation
                    simul_value_estimation_start = time.time()

                    # Prepare expanded node's denoising state for simulation
                    simulation_initial_levels_list = []
                    for i in range(len(expanded_node_candidates)):
                        if expanded_node_updated_levels is not None:
                            simulation_initial_levels_list.append(
                                expanded_node_updated_levels[i : i + 1]
                            )
                        else:
                            simulation_initial_levels_list.append(None)

                    assert expanded_node_updated_levels is not None, (
                        "expanded_node_updated_levels must be set"
                    )
                    simulation_initial_levels = np.concatenate(
                        simulation_initial_levels_list, axis=0
                    )  # (b, plan_tokens)
                    # Generate Schedule for Simulation (Complete Denoising)
                    value_estimation_noise_levels = (
                        self._generate_bidirectional_schedule(
                            simulation_initial_levels, complete_denoising=True
                        )
                    )  # (b, m, plan_tokens)

                    # input plans: list of (n_tokens, 1, fs*c)
                    # output plan_hists: (m+1, plan_tokens*fs, b, c)
                    value_estimation_plan_hists = self.parallel_plan(
                        effective_obs_normalized,
                        effective_goal_normalized,
                        horizon,
                        conditions,
                        guidance_scale=expanded_node_guidance_scales,
                        noise_level=value_estimation_noise_levels,
                        plans=value_estimation_plans,
                        prefix_len_list=prefix_len_list,
                    )

                    # Validate value_estimation_plan_hists shape: (m, plan_tokens*fs, B, c)
                    assert value_estimation_plan_hists.ndim == 4, (
                        f"value_estimation_plan_hists.ndim={value_estimation_plan_hists.ndim}, expected 4"
                    )
                    assert value_estimation_plan_hists.shape[2] == len(
                        expanded_node_candidates
                    ), (
                        f"value_estimation_plan_hists.shape[2]={value_estimation_plan_hists.shape[2]}, expected {len(expanded_node_candidates)}"
                    )

                    simul_value_estimation_end = time.time()
                    print(
                        f"Value estimation plan hist: {value_estimation_plan_hists.shape}"
                    )

                    # check if any plan is good
                    is_feasible = is_feasible_plan_hists(value_estimation_plan_hists)
                    
                    for i in range(len(expanded_node_candidates)): # b
                        if is_feasible[i] and filtered_expanded_node_plan_hists[i] is None:
                            filtered_expanded_node_plan_hists[i] = (
                                expanded_node_plan_hists[:, :, i]
                            )  # m (t fs) b c -> m (t fs) c
                            filtered_value_estimation_plan_hists[i] = (
                                value_estimation_plan_hists[:, :, i]
                            )
                    

                if None in filtered_expanded_node_plan_hists:
                    print("any diffs[:, i] > self.meeting_delta, resampling")
                    simulation_end_time = time.time()
                    tree.simulation_time.append(
                        simulation_end_time - simulation_start_time
                    )
                    continue
                else:
                    break
                

            # ----------------------SIM (DDIM) LOOP END----------------------------------------

            for i in range(len(filtered_expanded_node_plan_hists)):
                if filtered_expanded_node_plan_hists[i] is None:
                    filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[
                        :, :, i
                    ]
                    filtered_value_estimation_plan_hists[i] = (
                        value_estimation_plan_hists[:, :, i] if self.mcts_use_sim else expanded_node_plan_hists[:, :, i]
                    )
            expanded_node_plan_hists = torch.stack(
                filtered_expanded_node_plan_hists, dim=2
            )  # m (t fs) 'B' c
            value_estimation_plan_hists = torch.stack(
                filtered_value_estimation_plan_hists, dim=2
            )  # m (t fs) 'B' c

            # Value Calculation
            simul_value_calculation_start = time.time()
            achieved_indices = []
            final_best_plans = value_estimation_plan_hists[-1] # (plan_tokens*fs, b, c)

            import sys
            print(f"[DEBUG] Starting calculate_values_bidir with {len(expanded_node_candidates)} candidates", file=sys.stderr, flush=True)
            values, achieved_infos, achieved_ts = self.calculate_values_bidir(
                expanded_node_candidates, final_best_plans, tree
            )
            print(f"[DEBUG] calculate_values_bidir completed", file=sys.stderr, flush=True)
            for i in range(len(achieved_infos)):  # B
                achieved_info = achieved_infos[i]
                achieved_t = achieved_ts[i]
                if achieved_info == "Achieved":
                    tree.achieved = True
                    achieved_indices.append(i)

            print(f"Value Calculation: {values}, {achieved_infos}, {achieved_ts}")
            simul_value_calculation_end = time.time()

            # Node Allocation
            simul_node_allocation_start = time.time()
            selected_nodes_for_expansion = {}
            expanded_node_infos = {}
            for i in range(len(expanded_node_candidates)):  # B
                name = expanded_node_candidates[i]["name"]
                if name not in expanded_node_infos:
                    selected_nodes_for_expansion[name] = selected_nodes[i]
                    expanded_node_infos[name] = expanded_node_candidates[i]
                    expanded_node_infos[name]["plan_history"].append([])
                    # Note: is_tree1 is a property of MCTSTreeState (tree), not individual nodes
                value = values[i]
                plan_hist = expanded_node_plan_hists[:, :, i]  # m (t fs) c
                value_estimation_plan = value_estimation_plan_hists[-1, :, i]

                # Store updated denoising state for child node
                if expanded_node_updated_levels is not None:
                    updated_level = expanded_node_updated_levels[
                        i : i + 1
                    ]  # Shape: (1, plan_tokens)
                else:
                    updated_level = None

                if expanded_node_infos[name]["value"] is None:
                    expanded_node_infos[name]["value"] = value
                    expanded_node_infos[name]["value_estimation_plan"] = (
                        value_estimation_plan
                    )
                    expanded_node_infos[name]["plan_history"][-1] = (
                        plan_hist  # d m (t fs) c
                    )
                    expanded_node_infos[name]["current_levels"] = updated_level
                else:
                    if value > expanded_node_infos[name]["value"]:
                        expanded_node_infos[name]["value"] = value
                        expanded_node_infos[name]["value_estimation_plan"] = (
                            value_estimation_plan
                        )
                        expanded_node_infos[name]["plan_history"][-1] = plan_hist
                        expanded_node_infos[name]["current_levels"] = updated_level

            for name in selected_nodes_for_expansion:
                child_node = selected_nodes_for_expansion[name].expand(
                    **expanded_node_infos[name]
                )
                expanded_node_infos[name]["node"] = child_node

            simul_node_allocation_end = time.time()
            tree.simul_node_allocation_time.append(
                simul_node_allocation_end - simul_node_allocation_start
            )

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

            tree.search_num += 1
            tree.p_search_num += len(expanded_node_candidates)
            tree.pbar.update(len(expanded_node_candidates))
            tree.max_depth = max(
                tree.max_depth,
                max([info["depth"] for info in expanded_node_candidates]),
            )

            is_early_termination = tree.achieved

            import sys
            print(f"[DEBUG Early Term] is_early_termination={is_early_termination}, viz_plans={self.viz_plans}", file=sys.stderr, flush=True)

            if self.viz_plans:
                depths = [info["depth"] for info in expanded_node_candidates]
                terminal_depth_indices = [
                    i
                    for i, info in enumerate(expanded_node_candidates)
                    if info["depth"] == terminal_depth
                ]

                if is_early_termination:
                    visualize_indices = list(
                        set(terminal_depth_indices) | set(achieved_indices)
                    )
                else: 
                    visualize_indices = terminal_depth_indices

                visualize_indices = sorted(visualize_indices)

                # print(f"[DEBUG] viz_plans=True at search_num={tree.search_num}")
                # print(f"[DEBUG] terminal_depth={terminal_depth}")
                # print(f"[DEBUG] expanded_node_candidates depths={depths}")
                # print(f"[DEBUG] terminal_indices count={len(terminal_indices)}")

                if len(visualize_indices) > 0:
                    terminal_values = values[visualize_indices]
                    terminal_names = [
                        expanded_node_candidates[i]["name"] for i in visualize_indices
                    ]
                    terminal_expanded_plans = expanded_node_plan_hists[
                        -1, :, visualize_indices
                    ]  # m (t fs) b c
                    is_achieved_plan = [True if i in achieved_indices else False for i in visualize_indices]

                    # For goal tree visualization, flip the plans so they appear in start→goal direction
                    if "from_goal" in tree.tag:
                        terminal_expanded_plans = torch.flip(terminal_expanded_plans, [0])

                    self.visualize_node_value_plans(
                        is_achieved_plan,
                        tree.search_num,
                        terminal_values,
                        terminal_names,
                        terminal_expanded_plans,
                        start,
                        goal,
                        tag=tree.tag,
                    )

               

            print("============ Early Termination End ============")
            early_termination_end_time = time.time()
            tree.early_termination_time.append(
                early_termination_end_time - early_termination_start_time
            )

            # [FINE-GRAIN LOGGING] Log iteration completion and tree state
            if tracer:
                tracer.log(
                    tag="tree.iteration.complete",
                    data={
                        "search_num": tree.search_num,
                        "p_search_num": tree.p_search_num,
                        "max_depth": tree.max_depth,
                        "is_early_termination": is_early_termination,
                        "single_step": single_step,
                    },
                    step=tree.search_num,
                    depth=0,
                )

            if is_early_termination:
                if tracer:
                    tracer.log(
                        tag="tree.search.loop_break",
                        data={"reason": "early_termination"},
                        step=tree.search_num,
                        depth=0,
                    )
                break

            # ------------------------------------------------------------------
            # single_step mode: exit after 1 iteration (expanded_node_infos already set)
            # ------------------------------------------------------------------
            if single_step:
                if tracer:
                    tracer.log(
                        tag="tree.search.loop_break",
                        data={"reason": "single_step_mode"},
                        step=tree.search_num,
                        depth=0,
                    )
                break

        tree.pbar.close()

        # [LOGGING] Record search completion and tree stats
        from utils.tracer import get_tracer
        tracer = get_tracer()
        if tracer:
            terminal_depth_reached = tree.max_depth >= tree.terminal_depth
            tracer.log(
                tag="tree.search.complete",
                data={
                    "tree_tag": tree.tag,
                    "final_search_num": tree.search_num,
                    "final_max_depth": tree.max_depth,
                    "terminal_depth": tree.terminal_depth,
                    "terminal_depth_reached": terminal_depth_reached,
                    "total_nodes_expanded": tree.search_num,
                },
                step=tree.search_num,
                depth=0,
            )

        return tree, expanded_node_infos

    # =========================================================================
    # Helper functions for bidirectional alternating MCTS
    # =========================================================================

    def _build_plan_from_leaf(
        self,
        parent_node: "TreeNode",
        plan_tokens: int,
        segment_size: int,
    ) -> Tuple[torch.Tensor, int]:
        # Assembles a diffusion sequence: [prior trajectory | current obs | random noise | padding]
        """Construct the full plan_with_given_tokens for a new leaf node expansion.

        Returns a tensor of shape (n_tokens, 1, fs*c) with layout:
            [denoised_prefix(prefix_len) | obs_parent_token(1) | noisy_chunk | padding]
        When denoised_prefix is empty (root depth=0):
            [obs_parent_token(1) | noisy_chunk | padding]

        This output is ready to be passed directly to parallel_plan (pre-built format).
        """
        # Structural guarantee: plan_history is ALWAYS a list (never None)
        # This is guaranteed by TreeNode.__init__ and get_expandable_candidate
        assert isinstance(parent_node.plan_history, list), \
            f"plan_history must be list, got {type(parent_node.plan_history)}"

        if self.debug:
            print(
                f"    [DEBUG] Building initial plan from leaf. Parent: {parent_node.name}, Depth: {parent_node.depth}, History Segments: {len(parent_node.plan_history)}"
            )

        # Build obs_parent_token: the parent node's current observation, tokenised.
        parent_obs_pos = parent_node.obs_pos  # Raw (unnormalized) world coordinate

        # Normalize parent observation for diffusion model input
        obs_mean_np = self.data_mean[: self.observation_dim].cpu().numpy() if isinstance(self.data_mean, torch.Tensor) else np.array(self.data_mean[: self.observation_dim])
        obs_std_np = self.data_std[: self.observation_dim].cpu().numpy() if isinstance(self.data_std, torch.Tensor) else np.array(self.data_std[: self.observation_dim])
        parent_obs_normalized = (parent_obs_pos[: self.observation_dim] - obs_mean_np) / obs_std_np

        parent_obs_tensor = torch.tensor(
            parent_obs_normalized, dtype=torch.float32, device=self.device
        )
        obs_parent_token_raw = self.pad_init(
            parent_obs_tensor.unsqueeze(0)
        )  # (fs, 1, c) - normalized
        obs_parent_token = rearrange(
            obs_parent_token_raw, "fs b c -> 1 b (fs c)"
        )  # (1, 1, fs*c) - normalized

        # DEBUG: Check shapes
        import sys
        print(f"[DEBUG _build_plan_from_leaf] obs_parent_token_raw.shape={obs_parent_token_raw.shape}, obs_parent_token.shape={obs_parent_token.shape}", file=sys.stderr, flush=True)
        print(f"[DEBUG _build_plan_from_leaf] self.x_stacked_shape={self.x_stacked_shape}, self.frame_stack={self.frame_stack}", file=sys.stderr, flush=True)
        print(f"[DEBUG _build_plan_from_leaf] Expected obs_parent_token.shape=(1, 1, {self.frame_stack * self.x_stacked_shape[-1]})", file=sys.stderr, flush=True)

        prefix_len = 0 # initial value

        # --- Build denoised prefix from parent's plan_history ---
        # Structural guarantee: plan_history is ALWAYS a list (initialized to [] in TreeNode.__init__)
        # - Root nodes: plan_history = [] (empty, evaluates to False)
        # - Non-root nodes: plan_history has accumulated plan segments (non-empty, evaluates to True)
        # This condition checks: "if parent is NOT a root node" to use accumulated prefix
        if parent_node.plan_history:
            # plan_history stores plans in canonical (forward) order via flip_plan_for_insert_hist.
            latest_plan_canonical = parent_node.plan_history[-1][-1]  # (plan_tokens*fs, c)
            prefix_len_frames = parent_node.depth * segment_size * self.frame_stack
            full_prefix_canonical = latest_plan_canonical[:prefix_len_frames].unsqueeze(
                1
            )  # (prefix_len*fs, 1, c)

            # plan_history already contains normalized data from diffusion model output.
            # Do NOT normalize again - plan_history is already in normalized space.
            full_prefix = full_prefix_canonical  # (prefix_len*fs, 1, c) - already normalized
            denoised_prefix = rearrange(
                full_prefix, "(t fs) b c -> t b (fs c)", fs=self.frame_stack
            )  # (prefix_len, 1, fs*c)
            prefix_len = denoised_prefix.shape[0]

        else:
            denoised_prefix = None
            prefix_len = 0
        
        # Layout within plan_tokens_with_parent_obs: [prefix(prefix_len) | obs_parent(1) | noisy(plan_tokens-prefix_len)] Totally, plan_tokens + 1.
        noisy_total = plan_tokens - prefix_len
        assert noisy_total >= 0, f"Noisy total must be non-negative: {noisy_total}"

        # DEBUG
        import sys
        print(f"[DEBUG _build_plan_from_leaf] plan_tokens={plan_tokens}, prefix_len={prefix_len}, noisy_total={noisy_total}", file=sys.stderr, flush=True)

        batch_size = obs_parent_token.shape[1]  # always 1 per leaf
        noisy_parts = torch.randn(
            (noisy_total, batch_size, *self.x_stacked_shape),
            device=self.device,
        )
        noisy_parts = torch.clamp(
            noisy_parts, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise
        )

        # DEBUG: Check shapes before concatenation
        import sys
        print(f"[DEBUG _build_plan_from_leaf] noisy_parts.shape={noisy_parts.shape}", file=sys.stderr, flush=True)
        if denoised_prefix is not None:
            print(f"[DEBUG _build_plan_from_leaf] denoised_prefix.shape={denoised_prefix.shape}", file=sys.stderr, flush=True)

        # Assemble plan_tokens-length chunk: [prefix | obs_parent | noisy]
        if denoised_prefix is not None:
            plan_chunk_with_parent_obs = torch.cat(
                [denoised_prefix, obs_parent_token, noisy_parts],
                dim=0,  # (plan_tokens+1, 1, fs*c)
            )
        else:
            plan_chunk_with_parent_obs = torch.cat(
                [obs_parent_token, noisy_parts], dim=0
            )  # (plan_tokens+1, 1, fs*c)

        # [INSTRUMENTATION] Log plan assembly - HYPOTHESIS TESTING
        tracer = self.tracer if hasattr(self, 'tracer') else None
        if tracer:
            with tracer.scope("plan_assembly", phase="inference", depth=0):
                obs_parent_actual_pos = prefix_len if denoised_prefix is not None else 0

                # H1: Check if obs_parent and first noisy token have different properties
                first_token = plan_chunk_with_parent_obs[0:1]
                obs_parent_token_in_chunk = plan_chunk_with_parent_obs[obs_parent_actual_pos:obs_parent_actual_pos+1]
                first_noisy_token = plan_chunk_with_parent_obs[obs_parent_actual_pos+1:obs_parent_actual_pos+2] if obs_parent_actual_pos+1 < plan_chunk_with_parent_obs.shape[0] else None

                tracer.log(
                    tag="plan.assembly.structure",
                    data={
                        "prefix_len": prefix_len,
                        "obs_parent_pos": obs_parent_actual_pos,
                        "noisy_start_pos": obs_parent_actual_pos + 1,
                        "total_plan_tokens": plan_tokens + 1,
                        "plan_chunk_shape": list(plan_chunk_with_parent_obs.shape),
                        "parent_depth": parent_node.depth,
                        "segment_size": segment_size,
                    },
                    depth=0,
                )

                # H1 Testing: Log obs_parent properties
                tracer.log_tensor_stats(
                    tag="plan.segment_boundary.obs_parent",
                    tensor=obs_parent_token_in_chunk,
                    depth=1,
                    label=f"obs_parent_at_pos_{obs_parent_actual_pos}_depth_{parent_node.depth}",
                )

                # H2 Testing: Log first noisy token
                if first_noisy_token is not None:
                    tracer.log_tensor_stats(
                        tag="plan.segment_boundary.first_noisy",
                        tensor=first_noisy_token,
                        depth=1,
                        label=f"first_noisy_at_pos_{obs_parent_actual_pos+1}_depth_{parent_node.depth}",
                    )

                # H3 Testing: Log first token overall (will become segment start)
                tracer.log_tensor_stats(
                    tag="plan.segment_boundary.first_token",
                    tensor=first_token,
                    depth=1,
                    label=f"first_token_pos_0_depth_{parent_node.depth}",
                )

        assert plan_chunk_with_parent_obs.shape[0] == plan_tokens+1, (
            f"Plan chunk length mismatch: {plan_chunk_with_parent_obs.shape[0]} != {plan_tokens+1}"
        )

        # Append zero-padding to reach n_tokens.
        # padding formula should be: pad_tokens = n_tokens - (plan_tokens+1)
        plan_chunk_len = plan_chunk_with_parent_obs.shape[0]
        pad_tokens = self.n_tokens - plan_chunk_len

        # DEBUG: Verify padding calculation
        import sys
        print(f"[DEBUG _build_plan_from_leaf] plan_chunk.shape[0]={plan_chunk_len}, n_tokens={self.n_tokens}, pad_tokens={pad_tokens}", file=sys.stderr, flush=True)

        assert pad_tokens >= 0, f"pad_tokens must be non-negative: {pad_tokens}"
        pad = torch.zeros(
            (pad_tokens, batch_size, *self.x_stacked_shape), device=self.device
        )

        result = torch.cat([plan_chunk_with_parent_obs, pad], dim=0)  # (n_tokens, 1, fs*c)

        # DEBUG: Final result shape
        print(f"[DEBUG _build_plan_from_leaf] final result.shape[0]={result.shape[0]}, expected={self.n_tokens}", file=sys.stderr, flush=True)

        # Validate result shape before returning
        assert result.shape[0] == self.n_tokens, (
            f"result.shape[0]={result.shape[0]}, expected n_tokens={self.n_tokens}"
        )
        assert result.shape[1] == 1, f"result.shape[1]={result.shape[1]}, expected 1"
        # FIX: x_stacked_shape[0] is already the stacked dimension (frame_stack * original_dim)
        # so we check against x_stacked_shape[0], not frame_stack * x_stacked_shape[-1]
        assert result.shape[2] == self.x_stacked_shape[0], (
            f"result.shape[2]={result.shape[2]}, expected x_stacked_shape[0]={self.x_stacked_shape[0]}"
        )

        return result, prefix_len

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
        targets = np.stack([n.obs_pos for n in opposite_leaf_nodes])  # (N, D)
        obs_expanded = np.tile(current_leaf_obs, (targets.shape[0], 1))  # (N, D)
        values = self._compute_hilp_values(obs_expanded, targets, use_no_grad=True)

        best_idx = torch.argmax(values).item()
        best_value = values[best_idx].item()
        best_node = opposite_leaf_nodes[best_idx]

        if self.debug:
            print(
                f"      [DEBUG] Dynamic Goal Selection: Evaluated {len(opposite_leaf_nodes)} candidates. Best Value: {best_value:.4f}"
            )
        return best_node

    def _execute_plan_in_env(
        self,
        plan_frame_format: torch.Tensor,  # (T*fs, 1, c) - frame format
        envs: Any,
        agent: Optional[Any] = None,
        use_diffused_action: bool = False,
        parent_sim_state: Optional[dict] = None,
    ) -> tuple[List[torch.Tensor], dict]:
        """
        Execute a plan segment in environment with unified action computation.

        Handles both antmaze and pointmaze with:
        - Sub-goal interval updates for antmaze
        - PID controller for pointmaze
        - Trajectory and reward collection
        - Optional parent state injection for state stitching

        Args:
            plan_frame_format: Plan in frame format, shape (T*fs, 1, c)
            envs: Vectorized environment
            agent: RL agent for antmaze (None for pointmaze)
            use_diffused_action: If True, use action directly from diffusion
            parent_sim_state: Optional sim state (qpos, qvel) to restore before execution

        Returns:
            (trajectory_list, reward_dict)
            - trajectory_list: List of trajectory bundles
            - reward_dict: Dict with keys 'reached', 'episode_reward', etc.
        """
        trajectory = []


        self._set_sim_state(envs, parent_sim_state)

        # Get the full observation from environment (qpos + qvel concatenation for antmaze)
        # For DQL agent compatibility, we need the full obs [qpos, qvel], not just qpos[:2]
        current_sim_state = self._get_sim_state(envs)
        qpos = current_sim_state["qpos"]  # shape: (15,)
        qvel = current_sim_state["qvel"]  # shape: (14,)
        # Concatenate qpos and qvel to form full observation
        obs_flat = np.concatenate([qpos, qvel], axis=0)  # shape: (29,)
        obs_numpy = obs_flat[np.newaxis, :]  # shape: (1, 29)
    
      

        batch_size = plan_frame_format.shape[1]
        reached = np.zeros(batch_size, dtype=bool)
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        first_reach = np.zeros(batch_size)

        # Initialize sub_goal for antmaze
        plan_slice_np = None
        sub_goal_pos = None  # 2D position
        sub_goal_sim_state = None  # Full 29D sim_state
        sub_goal_step = None
        if "antmaze" in self.env_id:
            plan_slice_np = plan_frame_format[:, 0, :].detach().cpu().numpy()  # (T*fs, c)
            sub_goal_idx = min(self.sub_goal_interval, plan_frame_format.shape[0] - 1)
            sub_goal_pos = plan_slice_np[sub_goal_idx, :2]
            sub_goal_step = sub_goal_idx
            # Initialize sub_goal_sim_state as current state (will be updated during rollout)
            sub_goal_sim_state = current_sim_state.copy() if current_sim_state is not None else None
            sub_goal_sim_state["qpos"][:2] = sub_goal_pos

        # Execute plan: iterate, ensuring at least open_loop_horizon steps
        # MODIFIED: Use current_sim_state to generate plan_frame instead of cycling through stored plan
        plan_prev_frame = None
        loop_cnt = 0
        
        
        while loop_cnt < self.open_loop_horizon:
            # Generate plan_frame from current_sim_state 
            # Extract position from the actual executed state, not from pre-computed plan
            
            # current_sim_state = self._get_sim_state(envs)
            # plan_frame = torch.from_numpy(current_sim_state["qpos"][:2]).float().unsqueeze(0)  # (1, 2)
            plan_frame = torch.from_numpy(obs_numpy[:, :2]).float()  # (1, 2)

            # Update sub_goal for antmaze (with interval logic)
            if "antmaze" in self.env_id:
                if np.linalg.norm(plan_frame - sub_goal_pos[0]) < 1.0:
                    if sub_goal_step < plan_frame_format.shape[0] - self.sub_goal_interval:
                        sub_goal_step += self.sub_goal_interval
                        sub_goal_pos = plan_slice_np[sub_goal_step, :2]
                        # Update sub_goal_sim_state position
                        sub_goal_sim_state["qpos"][:2] = sub_goal_pos
                    else:
                        sub_goal_pos = plan_slice_np[-1, :2]
                        sub_goal_sim_state["qpos"][:2] = sub_goal_pos

            # Compute action using unified function
            action = self._compute_action_from_plan(
                obs_numpy=obs_numpy,
                plan_frame=plan_frame,
                plan_prev_frame=plan_prev_frame,
                agent=agent,
                sub_goal_pos=sub_goal_pos,
                sub_goal_sim_state=sub_goal_sim_state,
                current_sim_state=current_sim_state,
                use_diffused_action=use_diffused_action,
            )

            # Execute action in environment
            action_np = action.detach().cpu().numpy()
            
            # ── Pre-step state logging ──
            pre_step_sim_state = self._get_sim_state(envs)
            pre_step_qpos = pre_step_sim_state["qpos"][:2].copy() if pre_step_sim_state else None
            
            obs_numpy, reward, done, _ = envs.step(np.nan_to_num(action_np))

            # Ensure obs_numpy is 2D: (batch_size, obs_dim)
            if obs_numpy.ndim == 1:
                obs_numpy = obs_numpy[None, :]  # Add batch dimension
            
            # ── Post-step state logging ──
            post_step_sim_state = self._get_sim_state(envs)
            post_step_qpos = post_step_sim_state["qpos"][:2].copy() if post_step_sim_state else None
            obs_qpos_from_step = obs_numpy[0, :2].copy()  # Extract qpos from step return
            
            # Log state difference
            from utils.tracer import get_tracer
            tracer = get_tracer()
            if tracer:
                pre_post_qpos_diff = np.linalg.norm(post_step_qpos - pre_step_qpos) if pre_step_qpos is not None and post_step_qpos is not None else None
                obs_vs_get_attr_diff = np.linalg.norm(obs_qpos_from_step - post_step_qpos) if post_step_qpos is not None else None
                
                tracer.log(
                    tag="bidir_mcts._execute_plan_in_env.step_state_comparison",
                    data={
                        "loop_cnt": loop_cnt,
                        "pre_step_qpos": pre_step_qpos.tolist() if pre_step_qpos is not None else None,
                        "post_step_qpos": post_step_qpos.tolist() if post_step_qpos is not None else None,
                        "obs_qpos_from_step": obs_qpos_from_step.tolist(),
                        "pre_post_qpos_diff": float(pre_post_qpos_diff) if pre_post_qpos_diff is not None else None,
                        "obs_vs_get_attr_diff": float(obs_vs_get_attr_diff) if obs_vs_get_attr_diff is not None else None,
                        "reward": float(reward[0]) if reward is not None else None,
                        "done": bool(done[0]) if done is not None else None,
                    },
                    depth=1,
                )

            # Track rewards
            reached = np.logical_or(reached, reward >= 1.0)
            episode_reward += reward
            episode_reward_if_stay += np.where(~reached, reward, 1)
            first_reach += ~reached

            # Collect trajectory
            obs_torch = torch.from_numpy(obs_numpy).float()
            bundle = self.make_bundle(obs_torch, action, reward[..., None])
            trajectory.append(bundle)

            # Update prev_frame for velocity calculation
            plan_prev_frame = plan_frame

            # Increment counter
            loop_cnt += 1

            # Check for episode termination
            if done.any():
                break

        # Return results
        reward_dict = {
            "reached": reached,
            "episode_reward": episode_reward,
            "episode_reward_if_stay": episode_reward_if_stay,
            "first_reach": first_reach,
        }

        return trajectory, reward_dict

    def _compute_action_from_plan(
        self,
        obs_numpy: np.ndarray,  # (1, obs_size)
        plan_frame: torch.Tensor,  # (1, c)
        plan_prev_frame: Optional[torch.Tensor],  # (1, c) or None
        agent: Optional[Any] = None,
        sub_goal_pos: Optional[np.ndarray] = None,  # (2,) for antmaze - position only
        sub_goal_sim_state: Optional[dict] = None,  # Full sim_state for sub_goal
        current_sim_state: Optional[dict] = None,  # Full sim_state for current state
        use_diffused_action: bool = False,
    ) -> torch.Tensor:
        """
        Compute action for a single timestep given current observation and plan frame.

        Unified function supporting both antmaze and pointmaze:
        - antmaze: Uses agent.sample_action with current state (29D = qpos+qvel) + goal (2D position)
        - pointmaze: Uses PID-like controller

        Args:
            obs_numpy: Current observation, shape (1, obs_size)
            plan_frame: Plan frame at current timestep, shape (1, c)
            plan_prev_frame: Plan frame at previous timestep (for velocity calculation), shape (1, c) or None
            agent: RL agent for antmaze (None for pointmaze)
            sub_goal_pos: Goal position for antmaze, shape (2,) - only for reference
            sub_goal_sim_state: Full sim_state dict for sub_goal (with qpos, qvel)
            current_sim_state: Full sim_state dict for current state (with qpos, qvel)
            use_diffused_action: If True, use action directly from diffusion output

        Returns:
            action: (1, action_dim) - clipped to [-1, 1]
        """
        if use_diffused_action:
            # Direct action from diffusion model
            _, action, _ = self.split_bundle(plan_frame)
            return action

        if "antmaze" in self.env_id:
            # AntMaze: agent-based action
            # Agent expects: state (29D = qpos+qvel) + goal (2D position)
            # NOT the concatenation of [current, goal]
            assert agent is not None, "agent must be provided for antmaze"
            assert sub_goal_pos is not None, "sub_goal_pos must be provided for antmaze"
            assert current_sim_state is not None, "current_sim_state must be provided for antmaze"
            
            # Extract current state: qpos + qvel (29D total)
            current_qpos = current_sim_state["qpos"]  # (dof,)
            current_qvel = current_sim_state["qvel"]  # (dof,)
            state_29d = np.concatenate([current_qpos, current_qvel], axis=0)  # (29,)
            state_input = state_29d[np.newaxis, :]  # (1, 29)
            
            # Log via tracer
            from utils.tracer import get_tracer
            tracer = get_tracer()
            
            if tracer is not None:
                with tracer.scope("dql_agent_input", phase="validation"):
                    tracer.log(
                        tag="agent.input_dimensions",
                        data={
                            "qpos_shape": list(current_qpos.shape),
                            "qvel_shape": list(current_qvel.shape),
                            "state_input_shape": list(state_input.shape),
                            "sub_goal_pos_shape": list(sub_goal_pos.shape),
                            "expected_state_dim": 29,
                            "expected_goal_dim": 2,
                        },
                        depth=0,
                    )
            
            # Pass 29D state and 2D goal position to agent
            action = agent.sample_action(state_input, sub_goal_pos)
            return torch.from_numpy(action).float().reshape(1, -1)
        else:
            # PointMaze: PID-like controller
            obs_t = torch.from_numpy(obs_numpy).float()

            if plan_prev_frame is None:
                # First step: compute velocity from obs to plan
                plan_vel = plan_frame[:, :2] - obs_t[:, :2]
            else:
                # Subsequent steps: compute velocity from previous plan frame
                plan_vel = plan_frame[:, :2] - plan_prev_frame[:, :2]

            action = 12.5 * (plan_frame[:, :2] - obs_t[:, :2]) + 1.2 * (
                plan_vel - obs_t[:, 2:4]
            )
            return torch.clip(action, -1, 1)

    def _rollout_leaf_plan(
        self,
        leaf_plan_unnormalized: torch.Tensor,
        new_denoised_start_idx: int,
        new_denoised_end_idx: int,
        agent: Any,
        envs: Any,
        parent_sim_state: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Execute a freshly denoised plan segment in the actual environment.
        Restores the parent's physical state before stepping to ensure consistency.

        Args:
            leaf_plan_unnormalized: Fully assembled plan tensor, shape (T*fs, 1, c) unnormalized, where T is tokens and fs is frame_stack.
            new_denoised_start_idx: Start frame index (T*fs) of the freshly denoised chunk.
            new_denoised_end_idx: End frame index (exclusive, T*fs) of the freshly denoised chunk.
            agent: RL agent (used for antmaze sub-goal following).
            envs: Vectorized environment.
            parent_sim_state: Physical state (qpos/qvel) of the parent node to restore.

        Returns:
            final_sim_state: dictionary containing reached qpos/qvel (None if extraction failed).
        """
        # Restore parent's physical state before simulation
        assert parent_sim_state is not None, (
            "Parent sim state must be provided for rollout"
        )

        self._set_sim_state(envs, parent_sim_state)

        plan_slice = leaf_plan_unnormalized[
            new_denoised_start_idx:new_denoised_end_idx
        ]  # (chunk_t*fs, 1, c) in frame format

        if plan_slice.shape[0] == 0:
            return self._get_sim_state(envs)

        # [LOGGING] 1) Verify that current sim state matches parent_sim_state after restoration
        current_sim_state = self._get_sim_state(envs)
        assert current_sim_state is not None, "Failed to get current sim state"
        
        parent_qpos = parent_sim_state["qpos"][:2]
        current_qpos = current_sim_state["qpos"][:2]
        
        qpos_diff = np.linalg.norm(current_qpos - parent_qpos)
        assert qpos_diff < 1e-5, (
            f"After _set_sim_state, current qpos {current_qpos} does not match "
            f"parent_sim_state qpos {parent_qpos}. Diff: {qpos_diff}"
        )

        # [LOGGING] 2) Log continuity between parent_sim_state and plan_slice[0]
        from utils.tracer import get_tracer
        tracer = get_tracer()
        
        # plan_slice is already unnormalized, so use it directly
        plan_slice_first_qpos = plan_slice[0, 0, :2].detach().cpu().numpy()
        
        first_frame_diff = np.linalg.norm(plan_slice_first_qpos - current_qpos)
        
        if tracer:
            tracer.log(
                tag="rollout.plan_slice_continuity",
                data={
                    "parent_qpos": current_qpos.tolist(),
                    "plan_slice_first_qpos": plan_slice_first_qpos.tolist(),
                    "first_frame_diff": float(first_frame_diff),
                    "plan_slice_shape": str(plan_slice.shape),
                    "start_idx": new_denoised_start_idx,
                    "end_idx": new_denoised_end_idx,
                },
                depth=1,
            )

        # Execute plan with parent state injection for continuous state stitching
        # This restores parent's complete sim state before rolling out the new plan
        trajectory, _ = self._execute_plan_in_env(
            plan_frame_format=plan_slice,
            envs=envs,
            agent=agent if "antmaze" in self.env_id else None,
            use_diffused_action=False,
            parent_sim_state=parent_sim_state,  # Pass complete state for restoration
        )
        # Extract all obs from trajectory frames
        # trajectory is a list of bundles, not a tensor
        trajectory_bundle_list = []
        for trajectory_bundle in trajectory:
            obs_t, _, _ = self.split_bundle(trajectory_bundle)
            trajectory_bundle_list.append(obs_t)
        
        # Stack all obs: (T, batch, obs_dim) -> (T, obs_dim) when batch=1 and cat along dim=0
        trajectory_all_obs = torch.cat(trajectory_bundle_list, dim=0)  # (T, obs_dim)
        # Extract x,y positions: if batch dimension exists, extract it
        if trajectory_all_obs.dim() == 3:
            # Shape is (T, batch, obs_dim)
            trajectory_obs_positions = trajectory_all_obs[:, 0, :2].detach().cpu().numpy()  # (T, 2)
        else:
            # Shape is (T, obs_dim) - batch was singleton and got removed
            trajectory_obs_positions = trajectory_all_obs[:, :2].detach().cpu().numpy()  # (T, 2)
        
        # Also get final obs for state update
        obs, _, _ = self.split_bundle(trajectory[-1])

        final_sim_state = self._get_sim_state(envs)  # dummy sim_state
        final_sim_state["qpos"][:2] = obs[0, :2]

        # [LOGGING] 3) Capture reached physical state and log continuity with plan_slice[-1]
        
        # plan_slice is already unnormalized, so use it directly
        plan_slice_last_qpos = plan_slice[-1, 0, :2].detach().cpu().numpy()
        final_qpos = final_sim_state["qpos"][:2]
        
        last_frame_diff = np.linalg.norm(final_qpos - plan_slice_last_qpos)
        
        # Extract plan positions for logging
        plan_positions_np = plan_slice[:, 0, :2].detach().cpu().numpy()
        
        if tracer:
            tracer.log(
                tag="rollout.final_state_continuity",
                data={
                    "plan_slice_last_qpos": plan_slice_last_qpos.tolist(),
                    "final_sim_state_qpos": final_qpos.tolist(),
                    "last_frame_diff": float(last_frame_diff),
                    "plan_slice_shape": str(plan_slice.shape),
                    "trajectory_length": len(trajectory),
                    "trajectory_obs_positions_length": len(trajectory_obs_positions),
                    "trajectory_obs_positions": trajectory_obs_positions.tolist(),
                    "plan_positions": plan_positions_np.tolist(),
                    "trajectory_obs_first": trajectory_obs_positions[0].tolist(),
                    "trajectory_obs_last": trajectory_obs_positions[-1].tolist(),
                },
                depth=1,
            )

        return final_sim_state

    def _select_best_leaf(
        self,
        expanded_node_infos: dict[str, dict],
    ) -> dict:
        """
        Select the best expanded node info from an expanded_node_infos dict.

        Selects the candidate with the highest 'value' field.
        The returned dict contains a 'node' key with the actual child TreeNode.

        Args:
            expanded_node_infos: Dict[name -> info_dict] as returned by _run_mcts_search.

        Returns:
            The info dict with the highest value, or None if expanded_node_infos is empty.
        """
        if not expanded_node_infos:
            # No nodes were expanded (e.g., tree reached terminal_depth with no selectable nodes)
            # Return None to signal that no valid plan was found
            return None

        # ── Detailed validation logging ──
        tracer = get_tracer()
        if tracer:
            evaluation_log = []
            for node_name, info in expanded_node_infos.items():
                node = info.get("node")
                parent_node = info.get("parent_node")
                value = info.get("value")
                plan_history = info.get("plan_history", [])
                
                if node and parent_node:
                    evaluation_log.append({
                        "node_name": node_name,
                        "node_id": id(node),
                        "node_depth": node.depth,
                        "parent_id": id(parent_node),
                        "parent_depth": parent_node.depth,
                        "value": float(value) if value is not None else None,
                        "obs_pos": node.obs_pos.tolist() if hasattr(node.obs_pos, 'tolist') else [float(x) for x in node.obs_pos],
                        "sim_state_qpos": node.sim_state["qpos"][:2].tolist() if node.sim_state else None,
                        "plan_history_length": len(plan_history),
                    })
            
            tracer.log(
                tag="bidir_mcts._select_best_leaf.evaluation",
                data={
                    "total_candidates": len(expanded_node_infos),
                    "candidates": evaluation_log,
                },
                depth=1,
            )

        best_info = max(
            expanded_node_infos.values(),
            key=lambda info: info["value"]
            if info.get("value") is not None
            else float("-inf"),
        )
        
        # ── Log selected best ──
        if tracer:
            best_node = best_info.get("node")
            best_value = best_info.get("value")
            best_parent = best_info.get("parent_node")
            
            tracer.log(
                tag="bidir_mcts._select_best_leaf.selected",
                data={
                    "best_node_id": id(best_node),
                    "best_node_depth": best_node.depth,
                    "best_node_obs_pos": best_node.obs_pos.tolist() if hasattr(best_node.obs_pos, 'tolist') else [float(x) for x in best_node.obs_pos],
                    "best_node_sim_state_qpos": best_node.sim_state["qpos"][:2].tolist() if best_node.sim_state else None,
                    "best_parent_id": id(best_parent),
                    "best_parent_depth": best_parent.depth if best_parent else None,
                    "best_value": float(best_value) if best_value is not None else None,
                },
                depth=1,
            )
        
        return best_info

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
            output_plan: Tensor of shape (T_combined*fs, 1, c), where T = combined path length.
        """
        seg_size: int = plan_tokens // self.sequence_dividing_factor

        # --- Plan A: forward tree leaf ---
        # Structural guarantee: best_node is always an expanded node (from _select_best_leaf)
        # Expanded nodes always have non-empty plan_history (set during expansion in _run_mcts_search)
        # Root node can never be best_node because it is not expanded (see _select_best_leaf logic)
        assert len(best_node.plan_history) > 0, \
            f"best_node.plan_history must be non-empty for expanded nodes, but got {best_node.plan_history}"
        assert len(best_node.plan_history[-1]) > 0, \
            f"best_node.plan_history[-1] must be non-empty, but got {best_node.plan_history[-1]}"

        plan_a_full: torch.Tensor = best_node.plan_history[-1][-1]  # (T_total*fs, c)
        a_len: int = best_node.depth * seg_size * self.frame_stack
        t1_segments: torch.Tensor = plan_a_full[:a_len]  # (A_len, c)

        # [DIAGNOSTIC] Log plan slicing details
        if self.debug_log_level >= 2:
            import sys
            print(f"\n[DEBUG _extract_output_plan] Plan A slicing:", file=sys.stderr, flush=True)
            print(f"  best_node.depth: {best_node.depth}", file=sys.stderr, flush=True)
            print(f"  seg_size: {seg_size} (plan_tokens={plan_tokens}, seq_div={self.sequence_dividing_factor})", file=sys.stderr, flush=True)
            print(f"  frame_stack: {self.frame_stack}", file=sys.stderr, flush=True)
            print(f"  a_len calculated: {a_len}", file=sys.stderr, flush=True)
            print(f"  plan_a_full shape: {plan_a_full.shape}", file=sys.stderr, flush=True)
            print(f"  t1_segments shape after slicing: {t1_segments.shape}", file=sys.stderr, flush=True)
            print(f"  First few positions of plan_a_full: {plan_a_full[:5, :2]}", file=sys.stderr, flush=True)
            print(f"  First few positions of t1_segments: {t1_segments[:5, :2]}", file=sys.stderr, flush=True)

        # --- Bidirectional search: target_node handling ---
        # Structural guarantee in bidirectional MCTS (always active in interact()):
        # - best_node.target_node is ALWAYS set (opposite tree leaf selected at line 2341)
        # - best_node.target_node is NEVER None (guarded by assertion at line 2338-2340)
        #
        # However, target_node.plan_history can be empty in early iterations:
        # - Iteration 1: Tree1 active, Tree2 not expanded → Tree2.leaves may include root
        # - If root node selected as target_node → plan_history=[] (root has empty plan_history)
        # - Fallback to plan_A only (opposite tree hasn't contributed yet)
        # - After iteration 2+: Tree2 is expanded → target_node.plan_history is non-empty
        assert best_node.target_node is not None, \
            "target_node must be set in bidirectional MCTS (opposite tree leaf must be available)"
        if len(best_node.target_node.plan_history) == 0:
            # --- Early iteration or missing opposite tree: use plan_A only ---
            combined = t1_segments
        else:
            # --- Bidirectional: flip plan_B and concat ---
            plan_b_full: torch.Tensor = best_node.target_node.plan_history[-1][-1]  # (T_total*fs, c)
            b_len: int = best_node.target_node.depth * seg_size * self.frame_stack
            t2_flipped: torch.Tensor = torch.flip(
                plan_b_full[:b_len], [0]
            )  # (B_len, c)

            if self.debug:
                print(
                    f"[DEBUG] [Extract Plan] A_len={a_len}, B_len={b_len}, "
                    f"Combined={a_len + b_len}"
                )

            combined = torch.cat([t1_segments, t2_flipped], dim=0)  # (A_len+B_len, c)
            # H5 Testing: Log segment boundary details for concatenation
            tracer = self.tracer if hasattr(self, 'tracer') else None
            if tracer:
                with tracer.scope("plan_extraction", phase="inference", depth=0):
                    tracer.log(
                        tag="plan.extract.segment_concatenation",
                        data={
                            "t1_a_len": int(a_len),
                            "t2_b_len": int(b_len),
                            "combined_len": int(a_len + b_len),
                            "plan_tokens": plan_tokens,
                            "segment_size": int(plan_tokens // self.sequence_dividing_factor),
                            "best_node_depth": best_node.depth,
                            "target_node_depth": best_node.target_node.depth if best_node.target_node is not None else None,
                        },
                        depth=0,
                    )
                    
                    # Log first few tokens of each segment at boundary
                    tracer.log_tensor_stats(
                        tag="plan.extract.t1_last_tokens",
                        tensor=t1_segments[-3:] if t1_segments.shape[0] > 3 else t1_segments,
                        depth=1,
                        label=f"t1_last_3_tokens_len_{a_len}",
                    )
                    
                    tracer.log_tensor_stats(
                        tag="plan.extract.t2_first_tokens",
                        tensor=t2_flipped[:3] if t2_flipped.shape[0] > 3 else t2_flipped,
                        depth=1,
                        label=f"t2_first_3_tokens_after_flip_len_{b_len}",
                    )
                    
                    tracer.log_tensor_stats(
                        tag="plan.extract.combined_boundary",
                        tensor=combined[a_len-2:a_len+3] if combined.shape[0] > a_len+2 else combined[max(0, a_len-2):],
                        depth=1,
                        label=f"boundary_at_a_len_{a_len}_total_{a_len+b_len}",
                    )


        if not is_tree1:
            combined = torch.flip(combined, [0])  # (A_len+B_len, c)

        output = combined.unsqueeze(1)  # (A_len+B_len, 1, c)

        # Validate output shape before returning
        assert output.ndim == 3, f"output.ndim={output.ndim}, expected 3"
        assert output.shape[1] == 1, f"output.shape[1]={output.shape[1]}, expected 1"

        return output

    
    def _print_memory_report(self) -> None:
        """Print memory usage report if profiler is enabled."""
        if self.profiler:
            import sys
            report = self.profiler.report()
            print(report, file=sys.stderr, flush=True)

    def _get_sim_state(self, envs: Any) -> Optional[dict]:
        """Extract current qpos/qvel from envs (DummyVecEnv)."""
        try:
            # get_attr returns a list of attributes for each env in the vector
            # We assume batch size 1 for SimState restoration as per requirements
            data = envs.get_attr("data")
            if data and len(data) > 0:
                return {"qpos": data[0].qpos.copy(), "qvel": data[0].qvel.copy()}
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

# ============================================================================
# MEMORY MONITORING HELPER FUNCTIONS
# ============================================================================

def log_memory_stats(tracer, tag_prefix: str, step: Optional[int] = None):
    """Log current GPU and system memory statistics."""
    if tracer is None:
        return
    
    try:
        import torch
        import psutil
        
        data = {}
        
        # GPU memory stats
        if torch.cuda.is_available():
            data["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
            data["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
            data["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1e6
        
        # System memory stats
        try:
            mem = psutil.virtual_memory()
            data["system_memory_used_pct"] = mem.percent
            data["system_memory_used_gb"] = mem.used / 1e9
            data["system_memory_available_gb"] = mem.available / 1e9
        except:
            pass
        
        tracer.log(
            f"{tag_prefix}.memory_stats",
            data,
            step=step,
            depth=1
        )
    except Exception as e:
        pass

