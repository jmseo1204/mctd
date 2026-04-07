from typing import Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import time
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
import wandb
from PIL import Image, ImageDraw

from .df_base import DiffusionForcingBase
from utils.logging_utils import (
    make_trajectory_images,
    make_trajectory_videos,
    get_random_start_goal,
    make_convergence_animation,
    make_mpc_animation,
    get_maze_grid,
    is_grid_env,
    _sample_frame_indices,
)
from utils.tracer import Tracer, set_default_tracer, get_tracer
from .tree_node import TreeNode
from . import guidance
from .hilp_loader import HILPMemoizedWrapper, get_hilp_fn
from .env_executor import PlanExecutorMixin
from .plan_postproc import PlanPostprocMixin
from .kde_estimator import KDEEstimatorMixin
from .noise_schedule import NoiseScheduleMixin
from .plan_viz import PlanVizMixin

# Module-level process start time for lifecycle elapsed-time logging.
# All [LIFECYCLE] prints compute elapsed seconds from this anchor so you can
# read a single Docker-log stream and see exactly when each phase started.
_PROC_T0: float = time.time()

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
    replanning_time: List = field(default_factory=list)
    backprop_time: List = field(default_factory=list)
    early_termination_time: List = field(default_factory=list)
    replan_noiselevel_zero_padding_time: List = field(default_factory=list)
    replan_diffusion_time: List = field(default_factory=list)
    replan_value_calculation_time: List = field(default_factory=list)
    replan_node_allocation_time: List = field(default_factory=list)

    def get_all_nodes(self) -> List["TreeNode"]:
        """Return every node in the tree (BFS traversal), including the root."""
        nodes: List["TreeNode"] = []
        stack = [self.root_node]
        while stack:
            n = stack.pop()
            nodes.append(n)
            for c in n._children_nodes:
                if c["node"] is not None:
                    stack.append(c["node"])
        return nodes


class DiffusionForcingPlanning(KDEEstimatorMixin, NoiseScheduleMixin, PlanVizMixin, PlanExecutorMixin, PlanPostprocMixin, DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        # [INSTRUMENTATION] Initialize tracer (will be set later in interact() or on-demand in parallel_plan())
        self.tracer = None
        self.guidance_tracer = None  # Set to validation_anal tracer in interact()
        
        self.env_id = cfg.env_id
        self.dataset = cfg.dataset
        self.action_dim = len(cfg.action_mean)
        # obs_dim_indices: which indices of the dataset observation vector to use.
        # Defaults to all dims (backward compat). Set from training_hparams at eval time.
        _obs_idx = cfg.get("obs_dim_indices", None)
        self.obs_dim_indices: list = (
            list(_obs_idx) if _obs_idx is not None
            else list(range(len(cfg.observation_mean)))
        )
        # obs_bundle_indices: indices of obs dims within the model-space bundle [obs|action|reward].
        # Model obs is always laid out at positions 0..n_obs-1 in the bundle (Method A).
        self.obs_bundle_indices: list = list(range(len(self.obs_dim_indices)))
        # pos_dim_indices: which obs vector indices are spatial (x,y) position.
        # Physical qpos[:2] is always world (x,y) in AntMaze/PointMaze — kept separate.
        _pos_idx = cfg.get("pos_dim_indices", None)
        self.pos_dim_indices: list = list(_pos_idx) if _pos_idx is not None else [0, 1]
        self.use_reward = cfg.use_reward
        _n_obs = len(self.obs_dim_indices)
        self.unstacked_dim = _n_obs + self.action_dim + int(self.use_reward)
        # non_obs_bundle_indices: action and reward positions within the bundle (for zeroing).
        self.non_obs_bundle_indices: list = list(range(_n_obs, self.unstacked_dim))
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len

        # Manually initialize frame_stack as requested to solve dependency order
        self.frame_stack = cfg.frame_stack
        assert self.episode_len % self.frame_stack == 0, (
            "Episode length must be divisible by frame stack size"
        )
        self.n_tokens = self.episode_len // self.frame_stack
        self.valid_episode_len_multiple = cfg.get("valid_episode_len_multiple", 1)

        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.observation_mean = np.array(cfg.observation_mean)[self.obs_dim_indices]
        self.observation_std = np.array(cfg.observation_std)[self.obs_dim_indices]
        self.action_mean = np.array(cfg.action_mean[: self.action_dim])
        self.action_std = np.array(cfg.action_std[: self.action_dim])
        self.open_loop_horizon = cfg.get("open_loop_horizon", None)
        self.padding_mode = cfg.padding_mode
        self.interaction_seed = cfg.get("interaction_seed", None)
        self.use_random_goals_for_interaction = cfg.get("use_random_goals_for_interaction", False)
        self.task_id = cfg.get("task_id", None)
        # task_ids: list of task IDs to evaluate sequentially in one process.
        # Set by run_jobs.py when multiple same-config jobs are batched together.
        raw_ids = cfg.get("task_ids", None)
        self.task_ids = list(raw_ids) if raw_ids is not None else None  # Optional[List[int]]
        self.dql_model = cfg.get("dql_model", None)
        self.val_max_loops = cfg.get("val_max_loops", None)
        _scales = cfg.get("mctd_guidance_scales", [0.0])
        self.mctd_guidance_scales = list(_scales) if _scales is not None else [0.0]
        self.mctd_max_search_num = cfg.get("mctd_max_search_num", None)
        _rpl = cfg.get("replanning_target_level")
        try:
            _sampling_ts = cfg.diffusion.sampling_timesteps
        except Exception:
            _sampling_ts = None
        self.replanning_target_level = _rpl if _rpl is not None else ((_sampling_ts // 3) if _sampling_ts is not None else 0)
        self.mctd_skip_level_steps = cfg.get("mctd_skip_level_steps", None)
        self.jump = cfg.jump
        self.time_limit = cfg.get("time_limit", None)
        self.parallel_search_node = cfg.get("parallel_search_node", 1)
        self.parallel_search_num = self.parallel_search_node * len(self.mctd_guidance_scales)
        self.virtual_visit_weight = cfg.get("virtual_visit_weight", 1.0)
        self.warp_threshold = cfg.get("warp_threshold", 3.0) * self.jump
        self.leaf_parallelization = cfg.get("leaf_parallelization", False)
        if self.leaf_parallelization:
            _N = len(self.mctd_guidance_scales)
            assert self.parallel_search_num % _N == 0, (
                f"With leaf_parallelization=True, parallel_search_num "
                f"({self.parallel_search_num}) must be a multiple of "
                f"len(mctd_guidance_scales) ({_N})."
            )
        self.parallel_multiple_visits = cfg.get("parallel_multiple_visits", False)
        self.num_tries_for_bad_plans = cfg.get("num_tries_for_bad_plans", None)
        self.sub_goal_interval = cfg.get("sub_goal_interval", None)
        self.sub_goal_blend_steps = cfg.get("sub_goal_blend_steps", 1)
        self.viz_final_plans = cfg.get("viz_final_plans", False)
        self.meeting_delta = cfg.get("meeting_delta", 0.5)
        self.plan_feasibility_delta = cfg.get("plan_feasibility_delta", 100.0)
        self.diverge_threshold = cfg.get("diverge_threshold", 2.0)
        self.min_progress_threshold = cfg.get("min_progress_threshold", 0.0)
        self.max_child_resets = cfg.get("max_child_resets", 3)
        self.particle_guidance_scale = cfg.get("particle_guidance_scale", 0.0)
        self.use_TD_metric_as_dist = cfg.get("use_TD_metric_as_dist", False)
        self.debug_memory_profile = cfg.get("debug_memory_profile", False)
        self.profiler_snapshot_frames = cfg.get("profiler_snapshot_frames", cfg.get("max_plan_hist_keep", 20))  # Number of denoising frames kept for video
        self.sequence_dividing_factor = cfg.get("sequence_dividing_factor", None)
        self.horizon_scale = cfg.get("horizon_scale", None)
        self.noise_level_building_way = cfg.get("noise_level_building_way", "pyramid")

        # HILP value function guidance
        hilp_path = cfg.get("hilp_checkpoint_path", "td_models/hilp_ckpt_latest.pt")
        # Resolve path relative to repo root if relative
        import os
        if not os.path.isabs(hilp_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            hilp_path = os.path.join(repo_root, hilp_path)
        self.hilp_checkpoint_path = hilp_path
        self.hilp_obs_dim = cfg.get("hilp_obs_dim", 29)    # used only for legacy .pt checkpoints
        self.hilp_skill_dim = cfg.get("hilp_skill_dim", 256)  # used only for legacy .pt checkpoints
        # HILP value function instance will be loaded lazily and stored in _hilp_value_fn_instance
        # We don't initialize it here to prevent PyTorch from registering it as a submodule
        self.anchor_guidance_scale_ratio = cfg.get("anchor_guidance_scale_ratio", 1.0)
        self.rdf_guidance_scale = cfg.get("rdf_guidance_scale", 2.0)
        self.rmse_guidance_scale = cfg.get("rmse_guidance_scale", 1.0)
        self.use_score_func_with_TD = cfg.get("use_score_func_with_TD", True)
        self.TD_thres_for_far_target = cfg.get("TD_thres_for_far_target", None)
        self.kde_sigma = cfg.get("kde_sigma", 0.3)
        self.kde_grad_thres_sigma_coeff = cfg.get("kde_grad_thres_sigma_coeff", 0.3)
        self.kde_lam = cfg.get("kde_lam", 2.0)
        self.regularize_goal_guidance = cfg.get("regularize_goal_guidance", False)
        self.kde_sample_ratio = cfg.get("kde_sample_ratio", 0.1)
        self._kde_save_dir = os.path.expanduser(cfg.get("kde_save_dir", "~/.ogbench/data"))
        self.mcts_use_replan = cfg.get("mcts_use_replan", False)
        self.viz_replanning: bool = cfg.get("viz_replanning", True)
        self.use_uncertainty_as_value: bool = cfg.get("use_uncertainty_as_value", False)
        self.viz_uncertain_next_subplan_last_obs: bool = cfg.get("viz_uncertain_next_subplan_last_obs", False)
        self.fast_sampling_multiple: int = cfg.get("fast_sampling_multiple", 5)
        self.fast_sampling_steps: int = cfg.get("fast_sampling_steps", 10)
        self.use_rollout: bool = cfg.get("use_rollout", False)
        self.use_dynamic_obs_padding: bool = cfg.get("use_dynamic_obs_padding", True)

        super().__init__(cfg)
        self.plot_end_points = cfg.get("plot_start_goal", False)
        
        self.frame_sampling_way: str = cfg.get("frame_sampling_way", "linear")
        self.validation_video_max_frames: int = int(cfg.get("validation_video_max_frames", 200))
        self.validation_video_path_stride: int = int(cfg.get("validation_video_path_stride", 4))
        self.validation_video_fps: int = int(cfg.get("validation_video_fps", 8))
        self.viz_subplan_denoising: bool = bool(cfg.get("viz_subplan_denoising", False))
        self.viz_agent_rollout: bool = bool(cfg.get("viz_agent_rollout", False))
        self.viz_mujoco_renderer: bool = bool(cfg.get("viz_mujoco_renderer", False))
        self.viz_compare_expanded_to_value: bool = bool(cfg.get("viz_compare_expanded_to_value", False))

        # Initialize memory profiler for debugging
        if self.debug_memory_profile:
            from utils.memory_profiler import init_profiler
            self.profiler = init_profiler(self.device)
        else:
            self.profiler = None

        # KDE is only needed during planning (interact/eval), not training.
        # Loaded lazily on first interact() call.

        print(f"[LIFECYCLE +{time.time()-_PROC_T0:.1f}s] algo.__init__ complete", flush=True)

        # Initialize timing tracer now so all pre-interact lifecycle events are captured
        # in timestamp_anal_*.jsonl (not just in the Docker stdout log).
        from pathlib import Path as _Path_ti
        _ti_ts, _ti_model_id, _ti_job_name = "", "unknown", ""
        try:
            from hydra.core.hydra_config import HydraConfig as _HC_ti
            _hc_ti = _HC_ti.get()
            _out_ti = _Path_ti(_hc_ti.runtime.output_dir)
            _ti_ts = _out_ti.parts[-2].replace("-", "") + "_" + _out_ti.parts[-1].replace("-", "")
            for _ov_ti in list(_hc_ti.overrides.task):
                _kv_ti = _ov_ti.lstrip("+~")
                if _kv_ti.startswith("load="):
                    _raw_id = _kv_ti.split("=", 1)[1]
                    _ti_model_id = _Path_ti(_raw_id).parent.name if "/" in _raw_id else _raw_id
                elif _kv_ti.startswith("resume=") and _ti_model_id == "unknown":
                    _raw_id = _kv_ti.split("=", 1)[1]
                    _ti_model_id = _Path_ti(_raw_id).parent.name if "/" in _raw_id else _raw_id
                elif _kv_ti.startswith("name="):
                    _ti_job_name = _kv_ti.split("=", 1)[1]
        except Exception:
            import datetime as _dt_ti
            _ti_ts = _dt_ti.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tracer = Tracer(
            run_id=f"timestamp_anal_{_ti_ts}_{_ti_model_id}",
            purpose="plan_following_diagnosis",
            log_dir="logs",
            extra_meta={
                "env_id": self.env_id,
                "frame_stack": self.frame_stack,
                "episode_len": self.episode_len,
                "job_name": _ti_job_name,
            },
            debug_mode=self.cfg.get("DEBUG", True),
        )
        self.tracer.__enter__()  # Open file now (before interact() is called)
        set_default_tracer(self.tracer)
        self.tracer.log("lifecycle.algo_init_complete", {
            "elapsed_s": round(time.time() - _PROC_T0, 2),
        }, step=0, depth=0)

    def _tlog(self, tag: str, data: dict, depth: int = 0, source: str = "") -> None:
        """Safe timing tracer log: routes to timestamp_anal_*.jsonl."""
        if self.tracer is not None:
            self.tracer.log(tag, data, depth=depth, source=source or "df_planning.py")

    def _glog(self, tag: str, data: dict, depth: int = 0, source: str = "") -> None:
        """Safe guidance tracer log: routes to validation_anal_*.jsonl."""
        if self.guidance_tracer is not None:
            self.guidance_tracer.log(tag, data, depth=depth, source=source or "df_planning.py")

    def _get_hilp_value_fn(self):
        """Lazy loader for HILP value function model (or memoized wrapper).

        Delegates to hilp_loader.get_hilp_fn(), passing maze bounds derived
        from env_id (grid envs) or data_mean/std (continuous envs).
        """
        if not hasattr(self, '_hilp_value_fn_instance') or self._hilp_value_fn_instance is None:
            use_memo = bool(self.cfg.get("use_hilp_memoization", False))
            G = int(self.cfg.get("hilp_memoization_grid_size", 100))

            # Compute maze bounds
            if is_grid_env(self.env_id):
                maze_grid = get_maze_grid(self.env_id)
                H = len(maze_grid); W = len(maze_grid[0])
                x_min, x_max = (0.5 - 1) * 4, (H + 0.5 - 1) * 4
                y_min, y_max = (0.5 - 1) * 4, (W + 0.5 - 1) * 4
            else:
                _dm = (self.data_mean.cpu().numpy() if isinstance(self.data_mean, torch.Tensor)
                       else np.array(self.data_mean))
                _ds = (self.data_std.cpu().numpy() if isinstance(self.data_std, torch.Tensor)
                       else np.array(self.data_std))
                obs_mean_np = _dm[self.pos_dim_indices]
                obs_std_np  = _ds[self.pos_dim_indices]
                x_min, x_max = obs_mean_np[0] - 3 * obs_std_np[0], obs_mean_np[0] + 3 * obs_std_np[0]
                y_min, y_max = obs_mean_np[1] - 3 * obs_std_np[1], obs_mean_np[1] + 3 * obs_std_np[1]

            instance = get_hilp_fn(
                checkpoint_path=self.hilp_checkpoint_path,
                device=self.device,
                use_memoization=use_memo,
                hilp_obs_dim=self.hilp_obs_dim,
                hilp_skill_dim=self.hilp_skill_dim,
                grid_size=G,
                x_min=float(x_min), x_max=float(x_max),
                y_min=float(y_min), y_max=float(y_max),
                ref_obs=getattr(self, "_hilp_ref_obs", None),
            )

            _hilp_load_elapsed = time.time() - _PROC_T0
            print(f"[LIFECYCLE +{_hilp_load_elapsed:.1f}s] HILP loaded"
                  f"  memo={'yes' if use_memo else 'no'}"
                  f"  path={self.hilp_checkpoint_path}", flush=True)
            if self.tracer:
                self.tracer.log(
                    tag="lifecycle.hilp_loaded",
                    data={"elapsed_s": round(_hilp_load_elapsed, 2),
                          "path": self.hilp_checkpoint_path,
                          "memoized": use_memo},
                    step=0, depth=0,
                )

            object.__setattr__(self, '_hilp_value_fn_instance', instance)

        return object.__getattribute__(self, '_hilp_value_fn_instance')

    def _pad_obs_to_hilp_dim(self, obs_np: np.ndarray) -> np.ndarray:
        """Pad or crop observations to self.hilp_obs_dim.

        Non-position dims are filled from _hilp_ref_obs (a real env joint-state reference)
        so HILP receives in-distribution inputs rather than zeros.

        Args:
            obs_np: (N, D) float32 array.

        Returns:
            (N, hilp_obs_dim) float32 array.
        """
        obs_np = np.atleast_2d(obs_np).astype(np.float32)
        D = obs_np.shape[-1]
        if D == self.hilp_obs_dim:
            return obs_np
        _hilp_ref = getattr(self, '_hilp_ref_obs', None)
        if D < self.hilp_obs_dim:
            if _hilp_ref is not None:
                out = np.broadcast_to(_hilp_ref[None], (obs_np.shape[0], self.hilp_obs_dim)).copy()
                out[:, :D] = obs_np
                return out
            pad = np.zeros((obs_np.shape[0], self.hilp_obs_dim - D), dtype=np.float32)
            return np.concatenate([obs_np, pad], axis=-1)
        return obs_np[:, : self.hilp_obs_dim]

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

        # 4. Padding/Cropping to self.hilp_obs_dim via shared helper.
        dev = obs_t.device
        obs_t  = torch.from_numpy(self._pad_obs_to_hilp_dim(obs_t.cpu().numpy())).to(dev)
        goal_t = torch.from_numpy(self._pad_obs_to_hilp_dim(goal_t.cpu().numpy())).to(dev)

        # 5. Compute values
        has_value_attr = hasattr(hilp_value_fn, "value")

        if use_no_grad:
            with torch.no_grad():
                # Fix: Check for value() method BEFORE trying direct call
                if has_value_attr:
                    v1, v2 = hilp_value_fn.value(obs_t, goal_t)
                    res = torch.min(v1, v2)
                else:
                    v1, v2 = hilp_value_fn(obs_t, goal_t)
                    res = torch.min(v1, v2)
        else:
            if has_value_attr:
                v1, v2 = hilp_value_fn.value(obs_t, goal_t)
            else:
                v1, v2 = hilp_value_fn(obs_t, goal_t)
            res = torch.min(v1, v2)

        return res

    def _compute_state_temporal_dist_np(
        self,
        obs_a_np: np.ndarray,  # (N, D) float32 — unnormalized observations
        obs_b_np: np.ndarray,  # (N, D) float32 — unnormalized observations (same shape)
    ) -> np.ndarray:           # (N,) float32 — temporal distance (positive, smaller = temporally closer)
        """Compute HILP-based temporal distance: -V(obs_a, obs_b).

        Only called when use_TD_metric_as_dist=True. Smaller values indicate
        temporally closer states (mirrors the meaning of small L2 distance).
        """
        v = self._compute_hilp_values(obs_a_np, obs_b_np)  # (N,) tensor, negative values
        return (-v).cpu().numpy().astype(np.float32) / 7

    def _compute_distance(
        self,
        obs1: np.ndarray,  # (N, obs_dim) — observations for TD metric
        obs2: np.ndarray,  # (N, obs_dim) — observations for TD metric
        pos1: np.ndarray,  # (N, pos_dim) — positions for Euclidean distance
        pos2: np.ndarray,  # (N, pos_dim) — positions for Euclidean distance
    ) -> np.ndarray:       # (N,) — distance array
        """Unified distance computation based on use_TD_metric_as_dist flag.
        
        Encapsulates the choice between TD metric and Euclidean distance.
        Caller does not need to check use_TD_metric_as_dist — this function
        handles the branching internally.
        
        Args:
            obs1: First observation array (N, obs_dim) — used if use_TD_metric_as_dist=True
            obs2: Second observation array (N, obs_dim) — used if use_TD_metric_as_dist=True
            pos1: First position array (N, pos_dim) — used if use_TD_metric_as_dist=False
            pos2: Second position array (N, pos_dim) — used if use_TD_metric_as_dist=False
            
        Returns:
            distances: (N,) array of distances
        """
        if self.use_TD_metric_as_dist:
            return self._compute_state_temporal_dist_np(obs1, obs2)
        else:
            return np.linalg.norm(pos1 - pos2, axis=1)

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

        observations = observations[..., self.obs_dim_indices]
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
        init_bundle[:, self.non_obs_bundle_indices] = (
            0  # zero out actions and rewards after normalization
        )
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b, fs, c)
        bundles = torch.cat([init_bundle, bundles], dim=1)  # (b, fs+n_frames-1, c)
        bundles = rearrange(
            bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack
        )  # (n_tokens, b, fs, c)  where n_tokens = n_seg+1 = (n_frames-1)//fs + 1
        bundles = bundles.flatten(2, 3).contiguous()  # (n_tokens, b, fs*c)

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = None
        # bundles = bundles[::self.jump]
        return bundles, conditions, masks

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Override obs_dim_indices from checkpoint if its length conflicts with config."""
        parent = super()
        if hasattr(parent, "on_load_checkpoint"):
            parent.on_load_checkpoint(checkpoint)
        sd = checkpoint.get("state_dict", {})
        dm = sd.get("data_mean")
        if dm is not None:
            ckpt_obs_dim = int(dm.shape[0]) - self.action_dim - int(self.use_reward)
            if ckpt_obs_dim != len(self.obs_bundle_indices):
                import warnings
                warnings.warn(
                    f"[DFPlanning] Config obs_dim_indices length ({len(self.obs_bundle_indices)}) does not match "
                    f"checkpoint obs_dim ({ckpt_obs_dim}). Overriding obs_dim_indices with range({ckpt_obs_dim}).",
                    stacklevel=2,
                )
                self.obs_dim_indices = list(range(ckpt_obs_dim))
                self.obs_bundle_indices = list(range(ckpt_obs_dim))
                self.non_obs_bundle_indices = list(range(ckpt_obs_dim, ckpt_obs_dim + self.action_dim + int(self.use_reward)))
                self.unstacked_dim = ckpt_obs_dim + self.action_dim + int(self.use_reward)

    def training_step(self, batch, batch_idx):
        _step = self.trainer.global_step if self.trainer else 0

        # Memory snapshot: record steps 0-5, dump at step 5
        if _step == 0 and torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(max_entries=100000)
        elif _step == 5 and torch.cuda.is_available():
            import os
            _snap_dir = os.path.join(os.path.dirname(__file__), "../../logs")
            os.makedirs(_snap_dir, exist_ok=True)
            _snap_path = os.path.join(_snap_dir, "memory_snapshot.pickle")
            torch.cuda.memory._dump_snapshot(_snap_path)
            torch.cuda.memory._record_memory_history(enabled=None)

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

        # Per-dimension loss logging (before reduction)
        # loss shape: (n_tokens, B, fs*C), C = obs_dim + action_dim  [n_tokens = xs.shape[0]]
        with torch.no_grad():
            _loss_4d = rearrange(loss.detach(), "t b (fs c) -> t b fs c", fs=self.frame_stack)
            _w = rearrange(weights.float(), "(t fs) b -> t b fs 1", fs=self.frame_stack)
            _w_sum = _w.sum(dim=(0, 1, 2)).clamp(min=1e-8)
            _loss_per_dim = (_loss_4d * _w).sum(dim=(0, 1, 2)) / _w_sum  # (C,)

            _important_dims = {
                0: "obs/pos_x", 1: "obs/pos_y",
                3: "obs/quat_w", 4: "obs/quat_x", 5: "obs/quat_y", 6: "obs/quat_z",
            }
            for _dim_idx, _dim_name in _important_dims.items():
                if _dim_idx < _loss_per_dim.shape[0]:
                    self.log(f"training/loss_dim/{_dim_name}", _loss_per_dim[_dim_idx],
                             on_step=True, on_epoch=False, sync_dist=True)

            _per_dim_for_jsonl = _loss_per_dim.cpu().tolist()

        loss = self.reweight_loss(loss, weights)

        # Write all-dim losses to local JSONL (wandb-independent)
        _step = self.trainer.global_step if self.trainer else 0
        _epoch = self.trainer.current_epoch if self.trainer else 0
        self._dim_loss_logger.log(
            step=_step,
            epoch=_epoch,
            total_loss=float(loss.item()),
            per_dim_loss=_per_dim_for_jsonl,
        )

        self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log(
            "training/loss_epoch", loss, on_step=False, on_epoch=True, sync_dist=True
        )

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization: generate a fresh plan of valid_episode_len tokens via DDIM
        # (valid_episode_len = episode_len * valid_episode_len_multiple)
        if self.global_step % 10000 == 0:
            valid_n_tokens = int(self.n_tokens * self.valid_episode_len_multiple)
            _viz_batch = 8
            _n_viz_steps = 20
            with torch.no_grad():
                _x_t = torch.randn(
                    valid_n_tokens, _viz_batch, *self.x_stacked_shape, device=self.device
                )
                _x_t = torch.clamp(_x_t, -self.clip_noise, self.clip_noise)
                _sched = np.round(
                    np.linspace(self.sampling_timesteps, 0, _n_viz_steps + 1)
                ).astype(np.int64)
                for _m in range(_n_viz_steps):
                    _from_nl = torch.full(
                        (valid_n_tokens, _viz_batch), int(_sched[_m]),
                        dtype=torch.long, device=self.device,
                    )
                    _to_nl = torch.full(
                        (valid_n_tokens, _viz_batch), int(_sched[_m + 1]),
                        dtype=torch.long, device=self.device,
                    )
                    _x_t = self.diffusion_model.sample_step(_x_t, None, _from_nl, _to_nl)
            _xs_viz = self._unstack_and_unnormalize(_x_t)[self.frame_stack - 1:]
            _o_viz, _, _ = self.split_bundle(_xs_viz)
            trajectory = _o_viz.detach().cpu().numpy()[:-1]  # remove dummy last obs
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

        task_ids = self.task_ids if self.task_ids is not None else [self.task_id]
        for tid in task_ids:
            if tid != self.task_id:
                self.task_id = tid
                # Invalidate env cache so the new task gets freshly configured envs
                if hasattr(self, "_cached_env_key"):
                    del self._cached_env_key
            task_ns = f"{namespace}/task{tid}" if self.task_ids is not None else namespace
            self.interact(batch_size, conditions, task_ns)

    
    @staticmethod
    def _linspace_schedule(start: int, end: int, n_steps: int) -> np.ndarray:
        """Interpolate from start to end in n_steps, rounded and deduplicated.

        Mirrors the linspace subsampling used in diffusion.sample_step
        (timesteps → sampling_timesteps), applied here to map
        sampling_timesteps → _num_denoising_steps.
        Returns an array of length ≤ n_steps+1 with strictly monotone values.
        """
        raw = np.round(np.linspace(start, end, n_steps + 1)).astype(int)
        mask = np.concatenate([[True], np.diff(raw) != 0])
        return raw[mask]

    def process_segment_noise_levels(
        self,
        level_array: np.ndarray,
        sequence_dividing_factor: int,
        is_replanning = False,
        num_denoising_steps_override: Optional[int] = None,
    ) -> np.ndarray:
        plan_tokens = len(level_array)  # T
        assert plan_tokens % sequence_dividing_factor == 0, (
            f"Plan tokens must be divisible by sequence dividing factor, but got {plan_tokens} and {sequence_dividing_factor}"
        )
        segment_size = plan_tokens // sequence_dividing_factor

        _num_denoising_steps = num_denoising_steps_override if num_denoising_steps_override is not None else self.sampling_timesteps

        # Work with a copy
        steps = [level_array.copy()]

        work_array = level_array.copy()

        non_zero_indices = np.where(work_array > 0)[0]
        if len(non_zero_indices) == 0:
            if not is_replanning:
                return np.expand_dims(level_array, 0)
            # is_replanning + all zeros: entire plan is denoised, add noise to all tokens
            start_idx = plan_tokens
        else:
            start_idx = non_zero_indices[0]
        end_idx = min(start_idx + segment_size, plan_tokens)

        if is_replanning:
            # Replan the already-denoised prefix [:start_idx]:
            # 1) increase noise uniformly from 0 -> target_level  (up phase)
            # 2) decrease noise uniformly back to 0               (down phase)
            # Each phase uses half of _num_denoising_steps, mirroring how
            # sample_step subsamples timesteps → sampling_timesteps via linspace.
            target_level = self.replanning_target_level
            assert target_level <= self.sampling_timesteps, "replanning_target_level must be less than or equal to sampling_timesteps"

            half_n = max(_num_denoising_steps // 2, 1)
            up_schedule = self._linspace_schedule(0, target_level, half_n)
            down_schedule = self._linspace_schedule(target_level, 0, half_n)
            for level in up_schedule[1:]:
                work_array[:start_idx] = level
                steps.append(work_array.copy())
            for level in down_schedule[1:]:
                work_array[:start_idx] = level
                steps.append(work_array.copy())

        elif self.noise_level_building_way == 'causal':
            local_horizon = end_idx - start_idx
            uncertainty_scale = getattr(self, "uncertainty_scale", 1)

            initial_levels = steps[0][start_idx:end_idx]
            base_val = initial_levels[0]
            indices = np.arange(local_horizon)
            # Total reduction needed to drive the noisiest token (rightmost) to 0
            max_level = int(base_val + (local_horizon - 1) * uncertainty_scale)

            # Linspace over cumulative reductions: 0 → max_level in _num_denoising_steps
            reduction_schedule = self._linspace_schedule(0, max_level, _num_denoising_steps)
            for r in reduction_schedule[1:]:
                target_levels = np.maximum(
                    0, base_val + indices * uncertainty_scale - r
                ).astype(work_array.dtype)
                work_array[start_idx:end_idx] = np.minimum(
                    work_array[start_idx:end_idx], target_levels
                )
                steps.append(work_array.copy())

        elif self.noise_level_building_way == 'smooth' and start_idx > 0:
            # Phase 1: bilateral ramp — grow the denoised prefix toward B (left ramp)
            # while reducing the new segment head toward 0 (right ramp), so the
            # boundary jump level[B]-level[B-1] drops from N to ≤1.
            #
            # Left-ramp tokens have *increasing* noise levels (0 → k).  ddim_sample_step
            # would normally produce NaN for these via sqrt(negative) in the sigma term.
            # This is now handled transparently inside ddim_sample_step via forward_step
            # (q_sample), so sample_step is safe to call for all tokens.
            B = start_idx
            S = segment_size
            N_val = int(work_array[B])
            window_start = max(0, B - S)
            window_end = min(plan_tokens, B + S)

            for k in range(1, S + 1):
                # Left ramp: prefix tail grows from 0 → k
                left_start = max(window_start, B - k)
                left_len = B - left_start
                if left_len > 0:
                    ramp_start_val = k - left_len + 1
                    work_array[left_start:B] = np.arange(ramp_start_val, ramp_start_val + left_len)

                # Right ramp: new-segment head drops from N → N-k
                right_end = min(window_end, B + k)
                right_len = right_end - B
                if right_len > 0:
                    work_array[B:right_end] = np.maximum(
                        0, np.arange(N_val - k, N_val - k + right_len)
                    )

                steps.append(work_array.copy())

                # Early stop when boundary is smooth
                if work_array[B] - work_array[B - 1] <= 1:
                    break

            # Phase 2: linspace-based uniform subtraction over extended window.
            # Use delta steps so tokens with different post-Phase-1 levels all
            # reach 0 proportionally (same approach as sample_step subsampling).
            remaining_level = int(work_array[window_start:window_end].max())
            if remaining_level > 0:
                phase2_schedule = self._linspace_schedule(remaining_level, 0, _num_denoising_steps)
                prev = remaining_level
                for target in phase2_schedule[1:]:
                    delta = prev - target
                    work_array[window_start:window_end] = np.maximum(
                        0, work_array[window_start:window_end] - delta
                    )
                    steps.append(work_array.copy())
                    prev = target

        else:
            # Normal uniform denoising (also handles 'smooth' at start_idx==0).
            # Use delta steps to preserve relative differences across tokens.
            start_val = int(work_array[start_idx:end_idx].max())
            schedule = self._linspace_schedule(start_val, 0, _num_denoising_steps)
            prev = start_val
            for target in schedule[1:]:
                delta = prev - target
                work_array[start_idx:end_idx] = np.maximum(
                    0, work_array[start_idx:end_idx] - delta
                )
                steps.append(work_array.copy())
                prev = target

        return np.stack(steps, axis=0)  # (M, T)

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
        particle_guidance_scale: float = 0.0,
        group_ids: Optional[list] = None,
        call_type: str = "expansion",
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
            guidance_fn = None
        else:
            _pgs = particle_guidance_scale  # capture for lambda closure
            _gids = group_ids               # capture for lambda closure

            # smooth 모드: 과거 segment도 forward pass를 거치므로 guidance를
            # 현재 denoising 중인 segment tail 하나에만 적용해야 한다.
            # prefix_len_list[i] = i번째 candidate의 이미 denoised된 prefix token 수
            #   → active segment tail (frame-space) = frame_stack + (prefix_len + seg_size) * frame_stack - 1
            _active_tails = None
            if (getattr(self, 'noise_level_building_way', 'causal') == 'smooth'
                    and prefix_len_list is not None):
                _seg_t = horizon // self.frame_stack // self.sequence_dividing_factor
                _active_tails = [
                    self.frame_stack * (1 + pl + _seg_t) - 1
                    for pl in prefix_len_list
                ]

            _pat = _active_tails  # capture for lambda closure
            guidance_fn = lambda x: guidance.combined_guidance(
                self, x, goal, horizon, guidance_scale,
                particle_guidance_scale=_pgs, group_ids=_gids,
                active_tail_per_batch=_pat,
            )

        # [TIMING] Wrap guidance_fn to measure per-step cost and capture last loss values
        _guidance_call_ms: list = []
        _last_guidance_losses: dict = {}  # populated with scalar loss values from last call
        if guidance_scale is not None:
            _raw_gfn = guidance_fn
            def guidance_fn(x, _fn=_raw_gfn):
                _gt0 = time.time()
                r = _fn(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _guidance_call_ms.append((time.time() - _gt0) * 1000)
                if isinstance(r, dict):
                    _last_guidance_losses.clear()
                    for _k, _v in r.items():
                        try:
                            _last_guidance_losses[_k] = float(_v.sum().detach().item())
                        except Exception:
                            pass
                return r

        assert horizon % self.frame_stack == 0, (
            "horizon must be a multiple of frame_stack"
        )

        plan_tokens = horizon // self.frame_stack  # t (tokens per plan)

        # [INSTRUMENTATION] Ensure tracer is initialized (fallback: called outside interact())
        if self.tracer is None:
            from utils.tracer import Tracer, set_default_tracer
            self.tracer = Tracer(
                run_id="validation_run",
                purpose="plan_following_diagnosis",
                log_dir="logs",
                extra_meta={
                    "env_id": getattr(self, 'env_id', 'unknown'),
                    "batch_size": batch_size,
                    "frame_stack": self.frame_stack,
                },
                debug_mode=self.cfg.get("DEBUG", True),
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
        pad_tokens = self.n_tokens - plan_tokens - 1  # scalar: padding tokens (must match _build_plan_from_leaf)

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
                if not self.use_dynamic_obs_padding:
                    prefix_len = 0
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

        stabilization = 0
        _gpu_step_times = []

        # --- Denoising-step hook: capture prior score & guidance grads per step ---
        # After processing, _step_captures is stored in self._parallel_plan_step_captures
        # so callers (_run_mcts_search) can retrieve per-candidate captures for video.
        _step_captures = []
        _sc_plan_tokens = plan_tokens
        _sc_prefix_len_list = prefix_len_list
        _sc_fs = self.frame_stack

        def _capture_hook(data):
            """Extract plan-space pred_noise & guidance grads, move to CPU."""
            pn = data['prior_pred_noise']          # (n_tokens, B, fs*c) GPU
            pn_chunk = extract_plan_chunk(pn, _sc_plan_tokens, _sc_prefix_len_list)
            pn_exp = rearrange(pn_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs)
            gg_exp = {}
            for k, v in data['guidance_grads'].items():
                v_chunk = extract_plan_chunk(v.detach(), _sc_plan_tokens, _sc_prefix_len_list)
                gg_exp[k] = rearrange(v_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs).cpu()
            # Clean-space grads: ∂V/∂x̂_0 (no Jacobian — matches crimson HILP grad field direction)
            gg_clean_exp = {}
            for k, v in data.get('guidance_grads_clean', {}).items():
                v_chunk = extract_plan_chunk(v.detach(), _sc_plan_tokens, _sc_prefix_len_list)
                gg_clean_exp[k] = rearrange(v_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs).cpu()
            # pred_x_start: x̂_0 denoised estimate at this step (n_tokens, B, fs*c)
            pxs = data.get('pred_x_start')
            if pxs is not None:
                pxs_chunk = extract_plan_chunk(pxs.detach(), _sc_plan_tokens, _sc_prefix_len_list)
                pxs_exp = rearrange(pxs_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs).cpu()
            else:
                pxs_exp = None
            # Capture effective DDIM scale: sqrt(1 - alpha_t) for each plan token.
            # curr_noise_level is an INTEGER timestep index (0..timesteps-1), NOT a float.
            # Look up alphas_cumprod[t] to get the true alpha, then sqrt(1-alpha) ≈ c coefficient.
            # High noise (early denoising, large t) → alpha small → sqrt(1-alpha) ≈ 1 (large arrows)
            # Low noise (late denoising, small t) → alpha → 1 → sqrt(1-alpha) ≈ 0 (small arrows)
            nl = data.get('curr_noise_level')  # (n_tokens, B) integer GPU tensor
            if nl is not None:
                nl_chunk = extract_plan_chunk(nl.unsqueeze(-1), _sc_plan_tokens, _sc_prefix_len_list)  # (plan_tokens, B, 1)
                nl_int = nl_chunk.squeeze(-1).detach().long()  # (plan_tokens, B) integer timestep indices
                _n_alphas = self.diffusion_model.alphas_cumprod.shape[0]
                nl_int_clamped = nl_int.clamp(0, _n_alphas - 1)
                alpha_t = self.diffusion_model.alphas_cumprod[nl_int_clamped]  # (plan_tokens, B) float
                nl_cpu = (1.0 - alpha_t).sqrt().cpu()  # (plan_tokens, B) float in [0, 1]
            else:
                nl_cpu = None
            _step_captures.append({
                'prior_pred_noise': pn_exp.detach().cpu(),  # (plan_tokens*fs, B, c)
                'guidance_grads': gg_exp,                   # ∂V/∂x_t (through J_θ)
                'guidance_grads_clean': gg_clean_exp,       # ∂V/∂x̂_0 (clean space)
                'pred_x_start_pos': pxs_exp,               # x̂_0 positions (plan_tokens*fs, B, c) or None
                'noise_level': nl_cpu,  # (plan_tokens, B) float sqrt(1-alpha) or None
            })

        self.diffusion_model._step_hook = _capture_hook

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

            _gpu_t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sample = self.diffusion_model.sample_step(
                plan_with_given_tokens,  # (n_tokens, b, fs*c)
                conditions,
                from_noise_levels,  # (n_tokens, b)
                to_noise_levels,  # (n_tokens, b)
                guidance_fn=guidance_fn,
            )  # (n_tokens, b, fs*c)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _gpu_step_times.append(time.time() - _gpu_t0)

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

        # Clear the denoising-step hook before any further processing
        self.diffusion_model._step_hook = None

        # Stack all denoising steps
        plan_hist = torch.stack(plan_hist)  # (m+1, plan_tokens, b, fs*c)

        # Prepend a null capture so indices align with plan_hist (plan_hist[0] = initial noise,
        # capture[0] = null; capture[m] = step that produced plan_hist[m]).
        _step_captures.insert(0, {'prior_pred_noise': None, 'guidance_grads': {}, 'guidance_grads_clean': {}, 'pred_x_start_pos': None})

        # [MEMORY OPTIMIZATION] Keep N frames spanning full denoising process
        _n_keep = self.profiler_snapshot_frames
        _M_total = plan_hist.shape[0]
        # Default: no subsampling — each frame m corresponds to original step m
        self._parallel_plan_step_indices = np.arange(_M_total)
        self._parallel_plan_step_total = _M_total
        if _n_keep > 0 and plan_hist.shape[0] > _n_keep:
            _M = plan_hist.shape[0]
            if self.frame_sampling_way == 'quadratic':
                # Dense toward end-of-denoising: idx = (1-(1-t)^2) * (M-1), t = k/(N-1)
                _t = np.linspace(0.0, 1.0, _n_keep)
                _t_mapped = 1.0 - (1.0 - _t) ** 2  # quadratic: dense near t=1 (end of denoising)
                _idx_np = np.round(_t_mapped * (_M - 1)).astype(int)
                _idx = torch.from_numpy(_idx_np).long()
            else:
                # linear: evenly strided
                _idx = torch.linspace(0, _M - 1, _n_keep).long()
            plan_hist = plan_hist[_idx]
            # Apply same subsampling to step captures
            _step_captures = [_step_captures[i] for i in _idx.tolist()]
            # Store which original denoising steps were kept, for video frame labels
            self._parallel_plan_step_indices = _idx.numpy()
        
        # Rearrange to expand tokens into frame stacks
        plan_hist = rearrange(
            plan_hist,
            "m t b (fs c) -> m (t fs) b c",
            fs=self.frame_stack,
        )  # (m+1, plan_tokens*fs, b, c)

        # Store step captures for retrieval by callers (e.g. _run_mcts_search → video loop).
        # Each entry: {'prior_pred_noise': (plan_tokens*fs, B, c) or None, 'guidance_grads': dict}
        self._parallel_plan_step_captures = _step_captures

        # [TIMING] Log guidance_fn cost within denoising loop
        if _guidance_call_ms:
            _n_g = len(_guidance_call_ms)
            _g_total = sum(_guidance_call_ms)
            _gpu_total = sum(_gpu_step_times) * 1000
            self._tlog("timing.guidance_fn_in_denoising", {
                "n_calls": _n_g,
                "total_ms": round(_g_total, 1),
                "mean_ms": round(_g_total / _n_g, 1),
                "max_ms": round(max(_guidance_call_ms), 1),
                "guidance_pct_of_gpu": round(_g_total / (_gpu_total + 1e-6) * 100, 1),
            }, depth=1)

        # [TIMING] Log GPU sample_step timing per parallel_plan call
        if _gpu_step_times:
            _n = len(_gpu_step_times)
            _total_gpu_ms = sum(_gpu_step_times) * 1000
            _mean_gpu_ms = _total_gpu_ms / _n
            self._tlog("timing.gpu_sample_step", {
                "call_type": call_type,
                "batch_size": batch_size,
                "n_denoising_steps": _n,
                "total_gpu_ms": round(_total_gpu_ms, 1),
                "mean_per_step_ms": round(_mean_gpu_ms, 1),
            }, depth=1)

        # Validate plan_hist shape before returning
        # m+1: number of denoising steps (length of noise_level schedule)
        # plan_tokens*fs: horizon in frames
        # b: batch size (number of parallel instances)
        # c: observation dimension
        assert plan_hist.ndim == 4, f"plan_hist.ndim={plan_hist.ndim}, expected 4"
        assert plan_hist.shape[2] == batch_size, (
            f"plan_hist.shape[2]={plan_hist.shape[2]}, expected batch_size={batch_size}"
        )

        # Store last guidance losses so callers (_run_mcts_search) can log them via _glog.
        self._last_guidance_losses = _last_guidance_losses

        return plan_hist  # (m+1, plan_tokens*fs, b, c)

    def interact(
        self,
        batch_size: int,
        conditions: Optional[Any] = None,
        namespace: str = "validation",
    ) -> None:
        # Lazy KDE load: skip during training, load once on first eval/interact call.
        if self.kde_lam > 0.0 and not hasattr(self, "_kde_data_xy_cache"):
            self._load_kde_data_xy()

        try:
            import gym
            import ogbench
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print(
                "d4rl import not successful, skipping environment interaction. Check d4rl installation."
            )
            return

        _interact_t0 = time.time()
        print(f"[LIFECYCLE +{_interact_t0-_PROC_T0:.1f}s] interact() start", flush=True)

        # Capture validation_anal tracer (set by exp_base.py before calling interact()).
        # Do NOT override the default — keep it pointing to validation_anal so _glog can use it.
        self.guidance_tracer = get_tracer()  # validation_anal_*.jsonl during validation, else None
        tracer = self.tracer  # timestamp_anal_*.jsonl for timing/lifecycle logs

        with tracer:
            # [LIFECYCLE] Log pre-interact phases into jsonl so latency_analysis.sh
            # can show the full job time breakdown (not just MCTS compute time).
            tracer.log(
                tag="lifecycle.interact_start",
                data={
                    "proc_t0_unix": round(_PROC_T0, 3),
                    "pre_interact_elapsed_s": round(_interact_t0 - _PROC_T0, 2),
                    "batch_size": batch_size,
                },
                step=0, depth=0,
            )

            # [MEMORY CLEANUP] Clear caches before interaction
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            use_diffused_action = False
            agent = None  # Initialize agent to None; will be set for antmaze

            # [ENV CACHING] Check if environment is cached
            env_cache_key = f"{self.env_id}_{self.task_id}"
            if (
                hasattr(self, "_cached_envs")
                and hasattr(self, "_cached_env_key")
                and self._cached_env_key == env_cache_key
            ):
                # Reuse cached single environment
                envs = self._cached_envs
                agent = getattr(self, "_cached_agent", None)
                envs.reset()
                if not (self.env_id in OGBENCH_ENVS):
                    envs.seed(self.interaction_seed)
            else:
                # Create single environment
                # Use higher resolution when mujoco renderer is enabled
                _render_size = 480 if (self.viz_agent_rollout and self.viz_mujoco_renderer) else 200
                if self.env_id in OGBENCH_ENVS:
                    if "pointmaze" in self.env_id:
                        _maze_type = self.env_id.split("-")[1]
                        env_fn = lambda: ogbench.locomaze.maze.make_maze_env(
                            "point", "maze", maze_type=_maze_type,
                            width=_render_size, height=_render_size,
                        )
                        use_diffused_action = True
                    elif "antmaze" in self.env_id:
                        _maze_type = self.env_id.split("-")[1]
                        env_fn = lambda: ogbench.locomaze.maze.make_maze_env(
                            "ant", "maze", maze_type=_maze_type,
                            width=_render_size, height=_render_size,
                        )
                        from dql.main_Antmaze import hyperparameters
                        from dql.agents.ql_diffusion import Diffusion_QL as Agent

                        params = hyperparameters[self.dataset]

                        # Create temporary env to get dimensions
                        _temp_env = DummyVecEnv([env_fn])
                        state_dim = _temp_env.observation_space.shape[0]
                        action_dim = _temp_env.action_space.shape[0]
                        max_action = float(_temp_env.action_space.high[0])
                        _temp_env.close()

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
                        print(f"[INIT] DQL agent loaded: {dql_folder}", flush=True)
                else:
                    env_fn = lambda: gym.make(self.env_id)
                    agent = None

                # Create single DummyVecEnv and set task
                envs = DummyVecEnv([env_fn])
                if self.env_id in OGBENCH_ENVS:
                    envs.envs[0].set_task(self.task_id)
                elif self.use_random_goals_for_interaction:
                    envs.envs[0].set_target()
                print(f"[INIT] Environment created: {self.env_id} x1 (task_id={self.task_id})", flush=True)

                # Cache environment and agent
                self._cached_envs = envs
                self._cached_env_key = env_cache_key
                if agent is not None:
                    self._cached_agent = agent

                # [MUJOCO RENDERER] Create overhead renderer for viz_mujoco_renderer
                if self.viz_agent_rollout and self.viz_mujoco_renderer:
                    try:
                        import mujoco as _mujoco
                        _inner = envs.envs[0]
                        _fb_w = getattr(_inner, "width", 200)
                        _fb_h = getattr(_inner, "height", 200)
                        _renderer = _mujoco.Renderer(_inner.model, _fb_h, _fb_w)
                        _cam = _mujoco.MjvCamera()
                        _cam.type = _mujoco.mjtCamera.mjCAMERA_FREE
                        _cam.azimuth = 0.0
                        _cam.elevation = -90.0
                        if "giant" in self.env_id:
                            _cam.lookat[:] = [26.0, 18.0, 0.0]
                            _cam.distance = 70.0
                        else:  # large (default)
                            _cam.lookat[:] = [20.0, 14.0, 0.0]
                            _cam.distance = 55.0
                        self._cached_mujoco_renderer = _renderer
                        self._cached_overhead_camera = _cam
                        print(f"[INIT] MuJoCo overhead renderer created ({_fb_w}x{_fb_h})", flush=True)
                    except Exception as _re:
                        print(f"[WARN] MuJoCo renderer init failed: {_re}", flush=True)
                        self._cached_mujoco_renderer = None
                        self._cached_overhead_camera = None

            # [ENV CACHING END] Environment setup complete (cached or new)

            terminate = False
            # Method B: use stored observation_mean/std directly (no data_mean slicing)
            obs_mean = torch.tensor(self.observation_mean, dtype=torch.float32, device=self.device)
            obs_std = torch.tensor(self.observation_std, dtype=torch.float32, device=self.device)
            obs = envs.reset()
            # Randomize the goal for each environment
            if (
                self.env_id in OGBENCH_ENVS
            ):  # OGBench goal setting is already done through set_task()
                pass
            else:
                if self.use_random_goals_for_interaction:
                    for env in envs.envs:
                        env.set_target()

            obs = torch.from_numpy(obs).float().to(self.device)
            start = obs.detach()
            obs_normalized = (
                (obs[:, self.obs_dim_indices] - obs_mean[None]) / obs_std[None]
            ).detach()

            if self.env_id in OGBENCH_ENVS:  # OGBench
                goal = np.vstack(
                    [envs.reset_infos[i]["goal"] for i in range(len(envs.reset_infos))]
                )
            else:
                goal = np.concatenate([[env.env._target] for env in envs.envs])
            goal = torch.Tensor(goal).float().to(self.device)
            goal = torch.cat([goal, torch.zeros_like(goal)], -1)
            goal = goal[:, self.obs_dim_indices]  # select obs dims from raw env goal
            goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()

            steps = 0
            loops = 0  # Loop counter for bidirectional MCTS planning
            episode_reward = np.zeros(batch_size)
            episode_reward_if_stay = np.zeros(batch_size)
            reached = np.zeros(batch_size, dtype=bool)
            first_reach = np.zeros(batch_size)

            trajectory = []  # actual trajectory
            validation_rollout_agent_history: List[np.ndarray] = []
            validation_rollout_subgoal_history: List[np.ndarray] = []
            validation_pp_plan_history: List[np.ndarray] = []  # per-step pp_plan for video overlay
            validation_mujoco_frame_history: List[np.ndarray] = []  # per-step mujoco render frames

            # run mpc with diffused actions
            planning_time = []

            # ----------------------------------------------------------------
            # Bidirectional MCTS: initialize tree1/tree2 once before MPC loop.
            # These trees are maintained across MPC steps and expanded
            # alternately within each planning call.
            # ----------------------------------------------------------------
            horizon: int = int(self.episode_len * self.horizon_scale)
            _bidir_start_np = start.cpu().numpy()[:, self.obs_dim_indices]  # (b, obs_dim)
            _bidir_goal_np = goal.cpu().numpy()[:, self.obs_dim_indices]  # (b, obs_dim)
            # Capture initial physical state (always, regardless of use_rollout)
            initial_sim_state = self._get_sim_state(envs)
            assert initial_sim_state is not None, "Failed to capture initial sim state"
            assert np.allclose(initial_sim_state["qpos"][:2], _bidir_start_np[0][self.pos_dim_indices], atol=1e-5), \
                f"Physical start position {initial_sim_state['qpos'][:2]} does not match observation start position {_bidir_start_np[0][self.pos_dim_indices]}"

            # Build full reference observation (real joint state) for HILP.
            # HILP was trained on 29D obs (qpos+qvel); padding with zeros produces OOD inputs.
            # Storing this lets _compute_hilp_values use real non-position dims as a base.
            _ref_full = np.concatenate(
                [initial_sim_state["qpos"], initial_sim_state["qvel"]]
            )[: self.hilp_obs_dim].astype(np.float32)
            self._hilp_ref_obs = _ref_full  # used in _compute_hilp_values._pad

            # Derive heuristic goal simulation state from initial state
            goal_sim_state = {
                "qpos": initial_sim_state["qpos"].copy(),
                "qvel": np.zeros_like(initial_sim_state["qvel"]),  # Goal is assumed static
            }
            # Replace x, y coordinates with goal coordinates
            goal_sim_state["qpos"][:2] = _bidir_goal_np[0][self.pos_dim_indices]

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
                is_tree1=False,
             )
            
            # Flag: 0 → expand tree1 next, 1 → expand tree2 next
            expanded_tree_idx: int = 0
            # Configurable meeting threshold (Euclidean distance in unnormalized obs space)
            _meeting_delta: float = getattr(self.cfg, "meeting_delta", 2.0)

            _start_pos = _bidir_start_np[0][self.pos_dim_indices].tolist()
            _goal_pos  = _bidir_goal_np[0][self.pos_dim_indices].tolist()
            print(f"\n[MCTD] Episode start | start={[round(x,1) for x in _start_pos]} → goal={[round(x,1) for x in _goal_pos]} | horizon={horizon} | max_loops={self.val_max_loops}", flush=True)

            # Single-shot planning state
            is_meeting: bool = False
            best_node: Optional["TreeNode"] = None
            last_is_tree1: bool = True
            active_tree = bidir_tree1
            planning_start_time: float = time.time()

            while not terminate and loops < self.val_max_loops and not is_meeting:
                loops += 1

                # [EXPANSION CHECK] Early termination if both trees are fully explored
                if not bidir_tree1.root_node.is_expandable_flag and \
                   not bidir_tree2.root_node.is_expandable_flag:
                    terminate = True
                    break

                # Generate plan (start → goal)
                # _generate_plan_between_points has been inlined here.

                # ------------------------------------------------------------------
                # Bidirectional alternating MCTS planning
                # ------------------------------------------------------------------
                _start_np = start.cpu().numpy()[:, self.obs_dim_indices]  # (b, n_obs) — select from raw env obs
                _goal_np = goal.cpu().numpy()  # (b, n_obs) — already indexed by obs_dim_indices

                # Initialize infos dicts so {**infos1, **infos2} is safe even on the first step

                # Alternate expansion: one single_step per MPC iteration
                active_tree, expanded_node_infos = self._run_mcts_search(
                    bidir_tree1 if expanded_tree_idx == 0 else bidir_tree2,
                    bidir_tree2 if expanded_tree_idx == 0 else bidir_tree1,
                    horizon,
                    conditions,
                    _start_np,
                    _goal_np,
                    single_step=True,
                )
                
                # Per-leaf MPC rollout: update obs and sim_state for newly expanded leaves

                if self.use_rollout:
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
                        new_denoised_start: int = self._get_prefix_len_frames_from_depth(parent_node.depth, seg_size)
                        new_denoised_end: int = self._get_prefix_len_frames_from_depth(parent_node.depth + 1, seg_size)

                        _new_sim_state = self._rollout_leaf_plan(
                            leaf_plan_unnormalized=plan_unnormalized,
                            new_denoised_start_idx=new_denoised_start,
                            new_denoised_end_idx=new_denoised_end,
                            agent=agent,
                            envs=envs,
                            parent_sim_state=parent_node.sim_state,
                            is_backward=(active_tree is bidir_tree2),
                        )
                        assert _new_sim_state is not None, "_new_sim_state is None"
                        _child.sim_state = _new_sim_state
                        _child.obs = np.concatenate([_new_sim_state["qpos"], _new_sim_state["qvel"]])[self.obs_dim_indices]
                
                else:
                    # Derive obs from plan_history without physical simulation
                    seg_size: int = active_tree.plan_tokens // self.sequence_dividing_factor
                    for info in expanded_node_infos.values():
                        parent_node: "TreeNode" = info["parent_node"]
                        _child: Optional["TreeNode"] = info.get("node")
                        if _child is None:
                            continue

                        plan_hist_last: torch.Tensor = info["plan_history"][-1][-1]  # (t*fs, c)
                        _child.obs = self._extract_obs_at_boundary(
                            plan_hist_last.unsqueeze(1),  # (t*fs, 1, c)
                            depth=parent_node.depth + 1,
                            seg_size=seg_size,
                        )[0]  # (observation_dim,)

                        # Create new sim_state: copy parent's structure and update qpos[:2] with last valid position
                        _child.sim_state = {}
                        for k, v in parent_node.sim_state.items():
                            if isinstance(v, np.ndarray):
                                _child.sim_state[k] = v.copy()
                            else:
                                _child.sim_state[k] = v
                        # Update qpos[:2] with position from pos_dim_indices
                        _child.sim_state['qpos'][:2] = _child.obs[self.pos_dim_indices]


                # --- Per-expansion denoising video logging ---
                # Runs before _select_best_leaf so videos are logged even if the
                # episode terminates after this loop's plan execution.
                _v_start_np = start.cpu().numpy()[:, self.pos_dim_indices]
                _v_goal_np = goal.cpu().numpy()[:, self.pos_dim_indices]
                _v_hilp_fn = getattr(self, '_hilp_value_fn_instance', None)

                # Heatmap & grad-field are computed per-candidate based on the target (green star),
                # not the episode goal.  Cache by target_node name to avoid redundant computation.
                _v_tgt_vis_cache: dict = {}  # target_node_name → (heatmap, grad_field)

                # Per-candidate step captures from the expansion parallel_plan call.
                _v_sc_by_name: dict = getattr(self, '_expansion_step_captures_by_name', {})
                # obs_std for converting normalized pred_noise to world-space arrow scale
                _v_obs_std_np = (
                    self.data_std.cpu().numpy() if isinstance(self.data_std, torch.Tensor)
                    else np.array(self.data_std)
                )[self.pos_dim_indices]

                _viz_subplan_expand_ms = 0.0
                _viz_subplan_replan_ms = 0.0
                _viz_subplan_unc_ms = 0.0
                _viz_subplan_n = 0
                for _vname, _vinfo in (expanded_node_infos.items() if self.viz_subplan_denoising else []):
                    _viz_subplan_n += 1
                    # Always log expanded stage denoising
                    _viz_t0 = time.time()
                    self._log_candidate_plan_video(
                        _vname, _vinfo, active_tree,
                        _v_start_np, _v_goal_np,
                        _v_hilp_fn,
                        _v_tgt_vis_cache,
                        _v_sc_by_name,
                        _v_obs_std_np,
                        loops,
                        log_prefix="expanded",
                        plan_hist_override=_vinfo.get("expanded_plan_hist_frame"),
                    )
                    _viz_subplan_expand_ms += (time.time() - _viz_t0) * 1000
                    # Additionally log replanned stage denoising if mcts_use_replan is enabled
                    if self.mcts_use_replan and self.viz_replanning and _vinfo.get("replanned_plan_hist_frame") is not None:
                        _viz_t0 = time.time()
                        self._log_candidate_plan_video(
                            _vname, _vinfo, active_tree,
                            _v_start_np, _v_goal_np,
                            _v_hilp_fn,
                            _v_tgt_vis_cache,
                            _v_sc_by_name,
                            _v_obs_std_np,
                            loops,
                            log_prefix="replanned",
                            plan_hist_override=_vinfo.get("replanned_plan_hist_frame"),
                        )
                        _viz_subplan_replan_ms += (time.time() - _viz_t0) * 1000
                    # Log uncertainty estimate video: K*G fast-sampled sub_plans overlaid,
                    # each rendered as red gradient with alpha-scaled green endpoint dots.
                    # Skip if node is at terminal_depth (no further sub-plan denoising possible).
                    if (
                        self.viz_uncertain_next_subplan_last_obs
                        and _vinfo.get("uncertainty_plan_hist_frame") is not None
                        and _vinfo.get("depth", 0) < active_tree.terminal_depth
                    ):
                        _viz_t0 = time.time()
                        self._log_candidate_plan_video(
                            _vname, _vinfo, active_tree,
                            _v_start_np, _v_goal_np,
                            _v_hilp_fn,
                            _v_tgt_vis_cache,
                            _v_sc_by_name,
                            _v_obs_std_np,
                            loops,
                            log_prefix="uncertainty_estimate",
                            plan_hist_override=_vinfo["uncertainty_plan_hist_frame"],
                            is_uncertainty_viz=True,
                        )
                        _viz_subplan_unc_ms += (time.time() - _viz_t0) * 1000
                if _viz_subplan_n > 0 or self.viz_subplan_denoising:
                    self._tlog("timing.viz_subplan_denoising", {
                        "n_candidates": _viz_subplan_n,
                        "expand_ms": round(_viz_subplan_expand_ms, 1),
                        "replan_ms": round(_viz_subplan_replan_ms, 1),
                        "uncertainty_ms": round(_viz_subplan_unc_ms, 1),
                        "total_ms": round(_viz_subplan_expand_ms + _viz_subplan_replan_ms + _viz_subplan_unc_ms, 1),
                    }, depth=1)

                # Extract plan by selecting best leaf and combining plans
                best_info: dict = self._select_best_leaf(expanded_node_infos)

                # expanded_node_infos can be empty if all candidates were killed by
                # endpoint deduplication. In that case, skip plan extraction and continue.
                if best_info is None:
                    expanded_tree_idx = (expanded_tree_idx + 1) % 2
                    continue
                
                best_node = best_info["node"]

                # Update meeting condition: trees exhausted or FWD/BWD plans close enough
                _trees_exhausted = (
                    not bidir_tree1.root_node.is_expandable_flag and
                    not bidir_tree2.root_node.is_expandable_flag
                )
                _gap = self._compute_plan_gap(best_node, active_tree.plan_tokens, is_tree1=(expanded_tree_idx == 0))
                is_meeting = _trees_exhausted or (_gap is not None and _gap < self.meeting_delta)

                last_is_tree1 = (expanded_tree_idx == 0)
                # Alternate trees for next iteration
                expanded_tree_idx = (expanded_tree_idx + 1) % 2

            # Single-shot plan extraction and environment execution (after MCTS search completes)
            if best_node is not None:
                output_plan = self._extract_output_plan(
                    best_node,
                    plan_tokens=active_tree.plan_tokens,
                    is_tree1=last_is_tree1,
                    goal_normalized=goal_normalized,
                )  # (T_combined*fs+goal_pad, 1, c)

                plan_unnormalized = self._unnormalize_x(output_plan.unsqueeze(0))[-1]  # (T_combined*fs, 1, c)

                # Visualization with both forward and reverse trajectories
                start_numpy = start.cpu().numpy()[:, self.pos_dim_indices]
                goal_numpy = goal.cpu().numpy()[:, self.pos_dim_indices]

                # [only for viz] Extract best_node's tree trajectory (sim_state sequence from root to leaf)
                node_trajectory = self._extract_node_trajectory(best_node)

                # Create forward trajectory image with both plan (red) and node trajectory (blue)

                # Extract best_node's target_node obs (single green point)
                best_node_target_pos = None
                if best_node.target_node is None:
                    pass
                elif best_node.target_node.obs is None:
                    pass
                else:
                    best_node_target_pos = best_node.target_node.obs  # (obs_dim,) world coords

                # Compute HILP value heatmap if model is already loaded
                hilp_heatmap = None
                _hm_t0 = time.time()
                if hasattr(self, '_hilp_value_fn_instance') and self._hilp_value_fn_instance is not None:
                    try:
                        hilp_heatmap = self._compute_hilp_heatmap(best_node_target_pos)
                    except Exception as _hm_err:
                        pass
                _hm_ms = (time.time() - _hm_t0) * 1000

                # Compute HILP gradient field for arrow overlay
                hilp_grad_field = None
                _gf_t0 = time.time()
                if best_node_target_pos is not None:
                    try:
                        hilp_grad_field = self._compute_guidance_grad_fields(best_node_target_pos)
                    except Exception as _gf_err:
                        pass
                _gf_ms = (time.time() - _gf_t0) * 1000

                planning_end_time = time.time()
                self._tlog("timing.hilp_viz", {
                    "heatmap_ms": round(_hm_ms, 1),
                    "grad_field_ms": round(_gf_ms, 1),
                    "total_ms": round(_hm_ms + _gf_ms, 1),
                }, depth=0)
                _planning_loop_ms = (planning_end_time - planning_start_time) * 1000
                planning_time.append(planning_end_time - planning_start_time)

                obs_numpy = obs.detach().cpu().numpy()

                _t1_expansions = bidir_tree1.p_search_num
                _t2_expansions = bidir_tree2.p_search_num
                print(f"[MCTD] Search complete | loops={loops} | tree1={_t1_expansions}/{self.mctd_max_search_num} tree2={_t2_expansions}/{self.mctd_max_search_num} | plan_frames={plan_unnormalized.shape[0]} | node={best_node.name}", flush=True)

                # Reorder plan frames by proximity to resolve FWD-BWD spatial gap
                _reorder_t0 = time.time()
                plan_unnormalized = self._reorder_plan_by_proximity(plan_unnormalized)
                _reorder_ms = (time.time() - _reorder_t0) * 1000

                # Visualize postprocessed plan
                _ppviz_t0 = time.time()
                _pp_plan_np = plan_unnormalized[:, :, self.pos_dim_indices].detach().cpu().numpy()  # (K, 1, pos_dim)
                _pp_images = make_trajectory_images(
                    self.env_id, _pp_plan_np, 1, start_numpy.tolist(), goal_numpy.tolist(), self.plot_end_points
                )
                for _pp_i, _pp_img in enumerate(_pp_images):
                    self.log_image(
                        f"{namespace}_interaction/postprocessed_plan",
                        Image.fromarray(_pp_img),
                    )
                _ppviz_ms = (time.time() - _ppviz_t0) * 1000
                self._tlog("timing.plan_postproc", {
                    "n_frames": int(plan_unnormalized.shape[0]),
                    "reorder_ms": round(_reorder_ms, 1),
                    "pre_exec_viz_ms": round(_ppviz_ms, 1),
                    "total_ms": round(_reorder_ms + _ppviz_ms, 1),
                }, depth=0)

                # Use unified plan execution function
                _exec_start_time = time.time()
                trajectory_exec, reward_dict, rollout_viz = self._execute_plan_in_env(
                    plan_frame_format=plan_unnormalized,
                    envs=envs,
                    agent=agent if "antmaze" in self.env_id else None,
                    use_diffused_action=use_diffused_action,
                )
                _execution_loop_ms = (time.time() - _exec_start_time) * 1000
                self._tlog("timing.mpc_loop", {
                    "loop": loops,
                    "planning_ms": round(_planning_loop_ms, 1),
                    "execution_ms": round(_execution_loop_ms, 1),
                    "total_ms": round(_planning_loop_ms + _execution_loop_ms, 1),
                }, depth=0)

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
                    if rollout_viz["agent_positions"].size > 0:
                        validation_rollout_agent_history.append(rollout_viz["agent_positions"])
                        validation_rollout_subgoal_history.append(rollout_viz["subgoal_positions"])
                        n_steps_this_loop = rollout_viz["agent_positions"].shape[0]
                        for _ in range(n_steps_this_loop):
                            validation_pp_plan_history.append(_pp_plan_np)
                    _mj_frames = rollout_viz.get("mujoco_frames")
                    if _mj_frames is not None and len(_mj_frames) > 0:
                        validation_mujoco_frame_history.append(_mj_frames)

            # Close tree pbars (kept open during single_step MPC loop)
            bidir_tree1.pbar.close()
            bidir_tree2.pbar.close()
            print(f"[MCTD] Episode done | loops={loops} | steps={steps} | reached={bool(reached.any())} | reward={float(episode_reward.mean()):.3f}", flush=True)

            self.log(f"{namespace}/task_id", float(self.task_id))
            self.log(f"{namespace}/planning_time", np.sum(planning_time))
            self.log(f"{namespace}/episode_reward", episode_reward.mean())
            self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
            self.log(f"{namespace}/first_reach", first_reach.mean())
            self.log(f"{namespace}/success_rate", sum(episode_reward >= 1.0) / batch_size)

            # Visualization
            _post_exec_t0 = time.time()
            _post_exec_traj_image_ms = 0.0
            _post_exec_rollout_video_ms = 0.0
            _post_exec_mujoco_video_ms = 0.0
            if len(trajectory) > 0:
                samples = 1 # min(32, batch_size)
                trajectory = torch.stack(trajectory)
                start = start[:, self.pos_dim_indices].cpu().numpy().tolist()
                goal = goal[:, self.pos_dim_indices].cpu().numpy().tolist()
                rollout_agent_history = validation_rollout_agent_history
                rollout_subgoal_history = validation_rollout_subgoal_history

                _timg_t0 = time.time()
                images = make_trajectory_images(
                    self.env_id, trajectory[:, -samples:], samples, start, goal, self.plot_end_points
                )
                for i, img in enumerate(images):
                    self.log_image(
                        f"{namespace}_interaction/sample_{i}",
                        Image.fromarray(img),
                    )
                _post_exec_traj_image_ms = (time.time() - _timg_t0) * 1000

                if rollout_agent_history and self.viz_agent_rollout:
                    _rvid_t0 = time.time()
                    rollout_agent_np = np.concatenate(rollout_agent_history, axis=0)[:, None, :]
                    rollout_subgoal_np = np.concatenate(rollout_subgoal_history, axis=0)[:, None, :]
                    pp_plan_per_frame = validation_pp_plan_history if validation_pp_plan_history else None
                    videos = make_trajectory_videos(
                        self.env_id,
                        {"plan": trajectory[:, -samples:].detach().cpu().numpy()},
                        samples,
                        start,
                        goal,
                        rollout_agent_np,
                        rollout_subgoal_np,
                        self.plot_end_points,
                        max_frames=self.validation_video_max_frames,
                        path_stride=self.validation_video_path_stride,
                        postprocessed_plan_per_frame=pp_plan_per_frame,
                    )
                    for i, video in enumerate(videos):
                        if video.shape[0] > 0:
                            _video_step = self.get_safe_wandb_step()
                            print(
                                f"[wandb-video-debug] key={namespace}_interaction/sample_{i}_video "
                                f"shape={video.shape} dtype={video.dtype} fps={self.validation_video_fps} "
                                f"step={_video_step}",
                                flush=True,
                            )
                            self.log_video(
                                f"{namespace}_interaction/sample_{i}_video",
                                video,
                                fps=self.validation_video_fps,
                                step=_video_step,
                            )
                    _post_exec_rollout_video_ms = (time.time() - _rvid_t0) * 1000

                # [MUJOCO RENDER VIDEO] Log overhead MuJoCo render video
                if (
                    self.viz_agent_rollout
                    and self.viz_mujoco_renderer
                    and validation_mujoco_frame_history
                ):
                    _mj_t0 = time.time()
                    _mj_all = np.concatenate(validation_mujoco_frame_history, axis=0)  # (T, H, W, 3)
                    _mj_n = _mj_all.shape[0]
                    _mj_indices = _sample_frame_indices(_mj_n, self.validation_video_max_frames)
                    _mj_sampled = _mj_all[_mj_indices]  # (N, H, W, 3)
                    _mj_video = _mj_sampled.transpose(0, 3, 1, 2)  # (N, C, H, W)
                    _mj_step = self.get_safe_wandb_step()
                    print(
                        f"[wandb-video-debug] key={namespace}_interaction/mujoco_render "
                        f"shape={_mj_video.shape} dtype={_mj_video.dtype} fps={self.validation_video_fps} "
                        f"step={_mj_step}",
                        flush=True,
                    )
                    self.log_video(
                        f"{namespace}_interaction/mujoco_render",
                        _mj_video,
                        fps=self.validation_video_fps,
                        step=_mj_step,
                    )
                    _post_exec_mujoco_video_ms = (time.time() - _mj_t0) * 1000

            _post_exec_ms = (time.time() - _post_exec_t0) * 1000
            self._tlog("timing.post_exec", {
                "total_ms": round(_post_exec_ms, 1),
                "n_steps": steps,
                "traj_image_ms": round(_post_exec_traj_image_ms, 1),
                "rollout_video_ms": round(_post_exec_rollout_video_ms, 1),
                "mujoco_video_ms": round(_post_exec_mujoco_video_ms, 1),
            }, depth=0)

            _interact_elapsed = time.time() - _interact_t0
            _interact_done_elapsed = time.time() - _PROC_T0
            print(f"[LIFECYCLE +{_interact_done_elapsed:.1f}s] interact() complete  ({_interact_elapsed:.1f}s elapsed)", flush=True)
            tracer.log(
                tag="lifecycle.interact_complete",
                data={
                    "interact_elapsed_s": round(_interact_elapsed, 2),
                    "total_elapsed_s": round(_interact_done_elapsed, 2),
                },
                step=0, depth=0,
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
        # Determine obs split size by checking bundle format.
        bundle_size = bundle.shape[-1]
        _n_obs = len(self.obs_bundle_indices)

        # Check if bundle contains frame-stacked obs (larger dimension)
        if bundle_size == self.x_stacked_shape[0] + self.action_dim + (1 if self.use_reward else 0):
            # Frame-stacked obs format
            obs_split = self.x_stacked_shape[0]
        elif bundle_size == _n_obs + self.action_dim + (1 if self.use_reward else 0):
            # Non-stacked obs format
            obs_split = _n_obs
        else:
            # Fallback: infer from bundle_size
            remainder = bundle_size - self.action_dim - (1 if self.use_reward else 0)
            obs_split = remainder if remainder > 0 else self.x_stacked_shape[0]

        if self.use_reward:
            return torch.split(bundle, [obs_split, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [obs_split, self.action_dim], -1)
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
            obs = torch.zeros(batch_shape + (len(self.obs_bundle_indices),)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)

    def visualize_node_value_plans(
        self, is_achieved_plan, search_num, values, names, plans, starts, goals, tag="mcts_plan",
        denoised_lens=None,
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

        # Mask tokens beyond denoised length with NaN so they are invisible in plots
        if denoised_lens is not None:
            for i, used_len in enumerate(denoised_lens):
                if used_len < plan_obs.shape[0]:
                    plan_obs[used_len:, i, :] = np.nan

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

    def visualize_expanded_vs_value_plans(self, is_achieved_plan, names, expanded_plans, value_plans, starts, goals):
        # expanded_plans: (t fs) b c  — plans from expanded_node_plan_hists[-1]
        # value_plans:    (t fs) b c  — plans from replanned_plan_hists[-1]
        batch_size = expanded_plans.shape[1]

        if starts.ndim == 1:
            starts = starts[None, :]
        if goals.ndim == 1:
            goals = goals[None, :]
        if starts.shape[0] == 1:
            starts = np.repeat(starts, batch_size, axis=0)
        if goals.shape[0] == 1:
            goals = np.repeat(goals, batch_size, axis=0)

        def _to_obs_np(plans):
            plans_unnorm = self._unnormalize_x(plans)
            obs, _, _ = self.split_bundle(plans_unnorm)
            obs_np = obs.detach().cpu().numpy()
            if obs_np.ndim == 2:
                obs_np = obs_np[:, None, :]
            return obs_np

        expanded_obs = _to_obs_np(expanded_plans)
        value_obs = _to_obs_np(value_plans)

        expanded_images = make_trajectory_images(
            self.env_id, expanded_obs, batch_size, starts, goals, self.plot_end_points
        )
        value_images = make_trajectory_images(
            self.env_id, value_obs, batch_size, starts, goals, self.plot_end_points
        )

        for i in range(batch_size):
            self.log_image(f"test/expanded_{names[i]}", Image.fromarray(expanded_images[i]))
            self.log_image(f"test/value_{names[i]}", Image.fromarray(value_images[i]))

    def calculate_values(self, sub_plans, starts, goals):
        # sub_plans: (sliced_tokens*fs, b, c)

        if sub_plans.shape[1] != starts.shape[0]:  # b
            starts = starts.repeat(sub_plans.shape[1], axis=0)  # (b, c1)
        if sub_plans.shape[1] != goals.shape[0]:
            goals = goals.repeat(sub_plans.shape[1], axis=0)

        state_len = sub_plans.shape[0]
        batch_size = sub_plans.shape[1]
        sub_plans = self._unnormalize_x(sub_plans)
        obs, _, _ = self.split_bundle(
            sub_plans
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
                sub_plans.shape[0] - t
            ) / sub_plans.shape[0]
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
            parent_seq_len: int = self._get_prefix_len_frames_from_depth(parent_node.depth, seg_size)
            candid_seq_len: int = self._get_prefix_len_frames_from_depth(parent_node.depth + 1, seg_size)
            # Clamp candid_seq_len to actual plan length
            max_seq_len: int = min(final_best_plans.shape[0], candid_seq_len)
            plan_a_sliced: torch.Tensor = plan_a_full[parent_seq_len: max_seq_len].unsqueeze(1)  # (A_len, 1, c)

            # --- Delegate Warp/Achieved detection to calculate_values --- #
            # start = parent_node's physical position, goal = target_node's last valid frame from plan_hist
            start_np: np.ndarray = parent_node.obs[None]  # (1, n_obs) — obs already indexed by obs_dim_indices
            goal_np: np.ndarray = target_node.obs[None]   # (1, n_obs) — unnormalized world coords
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


        return values, achieved_infos, achieved_ts

    def _check_achieved_bidir(
        self,
        expanded_node_candidates: List[dict],
        final_best_plans: torch.Tensor,  # (plan_tokens*fs, B, c)
        tree: "MCTSTreeState",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lightweight Achieved detection without Warp check.

        Used exclusively when use_uncertainty_as_value=True so that value is determined
        solely by uncertainty while goal-reaching is still recorded separately.

        Checks whether any frame in the candidate's active segment falls within
        meeting_delta of the target node (same slice logic as calculate_values_bidir).

        Returns:
            achieved_infos: np.ndarray shape (B,), values 'Achieved' or 'NotReached'
            achieved_ts: np.ndarray shape (B,), frame index of first goal touch or -1
        """
        seg_size: int = tree.plan_tokens // self.sequence_dividing_factor
        B: int = len(expanded_node_candidates)
        achieved_infos: np.ndarray = np.array(["NotReached"] * B)
        achieved_ts: np.ndarray = np.full(B, -1, dtype=object)

        plan_unnorm = self._unnormalize_x(final_best_plans)  # (T*fs, B, c)
        obs_raw, _, _ = self.split_bundle(plan_unnorm)       # (T*fs, B, obs_dim)
        obs_np = obs_raw.detach().cpu().numpy()              # (T*fs, B, obs_dim)

        for i, candidate in enumerate(expanded_node_candidates):
            parent_node: "TreeNode" = candidate["parent_node"]
            target_node: Optional["TreeNode"] = candidate["target_node"]
            if target_node is None:
                continue
            goal_np: np.ndarray = target_node.obs  # (obs_dim,)

            parent_seq_len: int = self._get_prefix_len_frames_from_depth(parent_node.depth, seg_size)
            candid_seq_len: int = self._get_prefix_len_frames_from_depth(parent_node.depth + 1, seg_size)
            max_seq_len: int = min(final_best_plans.shape[0], candid_seq_len)

            for t in range(parent_seq_len, max_seq_len):
                if np.linalg.norm(obs_np[t, i] - goal_np) < self.meeting_delta:
                    achieved_infos[i] = "Achieved"
                    achieved_ts[i] = t
                    break

        return achieved_infos, achieved_ts

    def _compute_node_uncertainty(
        self,
        curr_obs: np.ndarray,  # (obs_dim,) — unnormalized current observation
        target_node: "TreeNode",
        tail_obs: np.ndarray,  # (G*K, obs_dim) — unnormalized tail observations
        gamma: float = 0.995,
        eps: float = 1e-8,
        eps_progress: float = 0.01,
    ) -> dict:
        """Compute uncertainty of a tree node from the spread of fast-sampled tail states.

        Theory: HILP converges to V(s,g) = -||phi(s)-phi(g)|| = -(1-γ^{d*})/(1-γ)
        where d*(s,g) is the true temporal distance and γ is the discount factor.

        Therefore the true temporal distance is recovered as:
            emb_dist = ||phi(s) - phi(g)||
            d*(s,g)  = log(1 - emb_dist*(1-γ)) / log(γ)

        Sigma and g_hat are computed in HILP embedding space (not temporal-distance space).
        T_curr, T_tail, and Delta_bar are in temporal-distance space (converted from emb_dist).

        NOTE: Requires hilp_fn to expose get_phi(torch.Tensor) → torch.Tensor.
        Currently only HILPJax and HILPMemoizedWrapper implement this interface.
        HILP (PyTorch) uses a different get_phi signature and is NOT supported.

        Args:
            curr_obs: Current node observation (obs_dim,) — the node being evaluated.
            target_node: Goal tree node (source of goal_obs).
            tail_obs: (G*K, obs_dim) unnormalized observations at the tail of each unc sample.
            gamma: Discount factor used during HILP training (default 0.995).
            eps: Small float for numerical stability.
            eps_progress: Minimum Delta_bar as a fraction of T_curr (prevents M_rem blow-up).

        Returns:
            dict with keys: U, ln_K, sigma_parallel, sigma_perp,
                            T_curr, T_tail, Delta_bar, M_rem
        """
        import math

        hilp_fn = self._get_hilp_value_fn()
        goal_obs: np.ndarray = target_node.obs  # (obs_dim,)
        K: int = tail_obs.shape[0]

        all_obs = np.concatenate(
            [curr_obs[None], goal_obs[None], tail_obs], axis=0
        )  # (2+K, obs_dim)
        all_obs_padded = self._pad_obs_to_hilp_dim(all_obs)

        obs_t = torch.from_numpy(all_obs_padded).float().to(self.device)
        with torch.no_grad():
            Z_all = hilp_fn.get_phi(obs_t).cpu().numpy()  # (2+K, D)

        z_curr = Z_all[0]   # (D,)
        z_goal = Z_all[1]   # (D,)
        Z      = Z_all[2:]  # (K, D)

        # --- Degenerate: already at goal in embedding space ---
        emb_dist_curr = float(np.linalg.norm(z_goal - z_curr))
        if emb_dist_curr < eps:
            return {
                'U': 0.0, 'ln_K': 0.0,
                'sigma_parallel': 0.0, 'sigma_perp': 0.0,
                'T_curr': 0.0, 'T_tail': 0.0, 'Delta_bar': 0.0, 'M_rem': 0.0,
            }

        # --- Embedding-distance → temporal distance conversion ---
        # HILP converges to ||phi(s)-phi(g)|| = (1-γ^{d*})/(1-γ)
        # → d*(s,g) = log(1 - emb_dist*(1-γ)) / log(γ)
        def _emb_to_td(emb_d: np.ndarray) -> np.ndarray:
            val = 1.0 - np.asarray(emb_d, dtype=np.float64) * (1.0 - gamma)
            val = np.clip(val, eps, 1.0 - eps)
            return (np.log(val) / np.log(gamma)).astype(np.float64)

        # --- Sigma in embedding space (step 2-4 from spec §3.2) ---
        z_bar = Z.mean(axis=0)                    # (D,)
        dZ    = Z - z_bar                          # (K, D)
        N_cov = max(K - 1, 1)                      # unbiased when K > 1
        Sigma = (dZ.T @ dZ) / N_cov               # (D, D)

        g_hat = (z_goal - z_curr) / emb_dist_curr  # (D,) unit vector in embedding space

        sigma_parallel_sq = float(g_hat @ Sigma @ g_hat)
        sigma_parallel_sq = max(sigma_parallel_sq, eps)
        sigma_perp_sq     = max(float(np.trace(Sigma)) - sigma_parallel_sq, eps)

        # --- Per-step entropy ln_K (spec §2.3) ---
        ln_K = (
            math.log(2.0 * math.pi * math.e)
            + 0.5 * math.log(sigma_parallel_sq)
            + 0.5 * math.log(sigma_perp_sq)
        )

        # --- Temporal distances (converted from embedding L2) ---
        T_curr = float(_emb_to_td(emb_dist_curr))

        emb_dists_tail = np.linalg.norm(z_goal[None] - Z, axis=-1)  # (K,)
        T_i    = _emb_to_td(emb_dists_tail)                          # (K,)
        T_tail = float(T_i.mean())

        Delta_bar = T_curr - T_tail
        Delta_bar = max(Delta_bar, T_curr * eps_progress)  # floor at eps_progress fraction

        M_rem = T_tail / Delta_bar
        # Cap M_rem: when Delta_bar hits the eps_progress floor (T_tail ≈ T_curr → near-zero
        # progress), M_rem blows up to ~1/eps_progress (~100).  Hard-cap at 20 for display
        # sanity; the eps_progress floor already handles the U floor implicitly.
        M_rem = min(M_rem, 20.0)
        # U     = ln_K * (1.0 + M_rem)
        U     = ln_K * T_curr

        return {
            'U':               U,
            'ln_K':            ln_K,
            'sigma_parallel':  math.sqrt(sigma_parallel_sq),
            'sigma_perp':      math.sqrt(sigma_perp_sq),
            'T_curr':          T_curr,
            'T_tail':          T_tail,
            'Delta_bar':       Delta_bar,
            'M_rem':           M_rem,
        }

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
                      Stored in root_node.obs and tree.tree_root_obs.

        Returns:
            MCTSTreeState: Fully initialized tree state ready for _run_mcts_search
        """
        plan_tokens: int = horizon // self.frame_stack  # t
        children_node_guidance_scales: list = self.mctd_guidance_scales
        max_search_num: int = self.mctd_max_search_num
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
            obs=root_obs,
            sim_state=root_sim_state,
        )
        root_node.set_value(0)  # Initialize the value of the root node

        pbar = tqdm(
            total=max_search_num,
            desc=f"MCTS ({tag})",
            leave=True,
            dynamic_ncols=True,
        )

        return MCTSTreeState(
            root_node=root_node,
            plan_tokens=plan_tokens,
            terminal_depth=terminal_depth,
            noise_level=noise_level,
            children_node_guidance_scales=children_node_guidance_scales,
            max_search_num=max_search_num,
            skip_level_steps=skip_level_steps,
            tag=tag,
            is_tree1=is_tree1,
            pbar=pbar,
            tree_root_obs=root_obs,
        )

    def _run_mcts_search(
        self,
        tree: MCTSTreeState,
        opposite_tree: MCTSTreeState,
        horizon: int,
        conditions: Optional[Any],
        start: np.ndarray,
        goal: np.ndarray,
        single_step: bool = False,
    ) -> tuple[MCTSTreeState, dict[str, dict]]:
        """
        (B function) Run the MCTS search loop for a given tree state.

        When `single_step=False` (default), runs until max_search_num or time_limit.
        When `single_step=True`, executes exactly one Selection→Expansion→Simulation→
        Backpropagation→EarlyTermination cycle and returns.

        In bidirectional mode, `opposite_tree` provides all nodes from the other tree
        so that dynamic goal selection can be performed via HILP across all candidates.

        Args:
            tree: MCTSTreeState initialized by _init_mcts_tree
            opposite_tree: MCTSTreeState of the other (opposite) tree; all its nodes
                           are used as candidates for dynamic goal selection.
            horizon: Planning horizon
            conditions: Planning conditions
            start: Raw (unnormalized) start observation, shape (1, obs_dim)
            goal: Raw (unnormalized) goal observation, shape (1, obs_dim)
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
                      'target_node': TreeNode|None,# dynamically selected opposite-tree node (bidir only)
                      ... (other TreeNode constructor fields)
                    }
                  Empty dict when no expansion occurred (e.g. all candidates already expanded).
        """
        # Unpack frequently used tree fields for readability
        root_node = tree.root_node
        children_node_guidance_scales = tree.children_node_guidance_scales
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
        self._glog("tree.search.start", {
            "tree_tag": tree.tag,
            "terminal_depth": tree.terminal_depth,
            "max_search_num": tree.max_search_num,
            "plan_tokens": tree.plan_tokens,
        }, depth=0)

        while True:
            if self.time_limit is not None:
                if time.time() - self.start_time > self.time_limit:
                    break
            else:
                # if search_num >= max_search_num:
                if tree.p_search_num >= tree.max_search_num:
                    break

            ## For checking the virtual visit count
            # root_node.check_virtual_visit_count()
            # [MEMORY DEBUG] Periodic memory logging
            if self.profiler and (tree.search_num > 0) and (tree.search_num % 10 == 0):
                self.profiler.snapshot(
                    f"mcts_search_iter_{tree.search_num}_{tree.tag}",
                    phase=f"mcts_iter_{tree.search_num}"
                )

            ###############################
            # Selection
            #  When leaf parallelization is True, then the selection is done in partially parallel (the children nodes from same parent node are selected at the same time)
            #  When leaf parallelization is False, then the selection is done in fully sequential (only one node is selected at a time)

            selection_start_time = time.time()
            psn = self.parallel_search_num
            selected_nodes, expanded_node_candidates = [], []
            while psn > 0:
                selected_node = root_node

                _selection_exhausted = False
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
                    try:
                        selected_node = selected_node.select(
                            leaf_parallelization=self.leaf_parallelization,
                            max_child_resets=self.max_child_resets,
                        )
                    except ValueError:
                        # Node's subtree is fully exhausted (all children permanently killed
                        # and reset budget spent). Skip this psn slot gracefully.
                        _selection_exhausted = True
                        break

                # Recompute expandable node names after traversal: select() may call
                # reset_dead_children() which wipes children and makes previously
                # non-listed slots expandable again, causing stale-list mismatches.
                if not self.parallel_multiple_visits:
                    expandable_node_names = root_node.get_expandable_node_names()

                if _selection_exhausted:
                    psn -= (
                        1
                        if not self.leaf_parallelization
                        else len(children_node_guidance_scales)
                    )
                    continue

                is_term = selected_node.is_terminal()
                is_exp = selected_node.is_expandable(consider_virtually_visited=(not self.parallel_multiple_visits))
                is_sel = selected_node.is_selectable()

                if is_term or (not is_sel and not is_exp):
                    psn -= (
                        1
                        if not self.leaf_parallelization
                        else len(children_node_guidance_scales)
                    )
                    continue
                if self.leaf_parallelization:
                    for i in range(len(children_node_guidance_scales)):
                        # Skip slots that already have a child node (created in a previous
                        # iteration) or are already virtually visited — both mean the slot
                        # is occupied / scheduled for this round and should not be re-added.
                        # Also skip permanently_dead slots (dedup-killed): get_expandable_candidate
                        # with explicit index bypasses the permanently_dead check, so we guard here.
                        child_slot = selected_node._children_nodes[i]
                        if child_slot['node'] is not None:
                            continue
                        if child_slot['permanently_dead']:
                            continue
                        if (not self.parallel_multiple_visits) and child_slot['virtually_visited']:
                            continue

                        # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                        expanded_node_candidate = (
                            selected_node.get_expandable_candidate(
                                index=i,
                                consider_virtually_visited=(
                                    not self.parallel_multiple_visits
                                ),
                            )
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
                        psn -= 1
                else:
                    # when multiple visits is False, then we need to consider the virtually visited nodes to visit only once
                    expanded_node_candidate = selected_node.get_expandable_candidate(
                        index=None,
                        consider_virtually_visited=(not self.parallel_multiple_visits),
                    )

                    selected_nodes.append(selected_node)
                    expanded_node_candidates.append(expanded_node_candidate)
                    if not self.parallel_multiple_visits:
                        if not expanded_node_candidate["name"] in expandable_node_names:
                            raise ValueError(
                                f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names"
                            )
                        expandable_node_names.remove(expanded_node_candidate["name"])
                    psn -= 1
                if not self.parallel_multiple_visits:
                    if len(expandable_node_names) == 0:
                        break
            if len(selected_nodes) == 0:
                break
            selection_end_time = time.time()
            tree.selection_time.append(selection_end_time - selection_start_time)

            # ------------------------------------------------------------------
            # Dynamic Start & Goal Selection for each expansion candidate
            # ------------------------------------------------------------------
            # Filter out nodes with uninitialized obs
            valid_candidates = []
            for info in expanded_node_candidates:
                if info["parent_node"].obs is not None:
                    valid_candidates.append(info)


            # If no valid candidates (all had None obs), skip diffusion and continue
            if not valid_candidates:
                break  # Exit search loop

            assert tree.plan_tokens % self.sequence_dividing_factor == 0, (
                f"plan_tokens {tree.plan_tokens} is not divisible by sequence_dividing_factor {self.sequence_dividing_factor}"
            )
            seg_size = tree.plan_tokens // self.sequence_dividing_factor

            eff_obs_norm_list, eff_goal_norm_list = [], []
            eff_start_np_list, eff_goal_np_list = [], []

            for info in valid_candidates:
                parent_node = info["parent_node"]
                parent_obs = parent_node.obs

                # Start: Normalized parent position for planning context
                # obs is already indexed by obs_dim_indices; Method B for normalization stats
                obs_mean_np = np.array(self.observation_mean)
                obs_std_np = np.array(self.observation_std)
                eff_start_np_list.append(parent_obs[None])
                p_norm = torch.tensor(
                    (parent_obs - obs_mean_np) / obs_std_np,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                eff_obs_norm_list.append(p_norm)

                # Goal: Dynamic selection from all nodes in the opposite tree
                all_opposite_nodes = opposite_tree.get_all_nodes()
                assert len(all_opposite_nodes) > 0, "opposite_tree has no nodes"
                target_node = self._select_dynamic_goal(
                    current_leaf_obs=parent_obs,
                    opposite_tree_all_nodes=all_opposite_nodes,
                )
                info["target_node"] = (
                    target_node  # Will be propagated to child TreeNode via expand()
                )
                target_pos = target_node.obs

                eff_goal_np_list.append(target_pos[None])
                g_norm = torch.tensor(
                    (target_pos - obs_mean_np) / obs_std_np,
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
            filtered_replanned_plan_hists = [None] * len(
                valid_candidates
            )
            # Per-candidate uncertainty plan hists: populated by NODE UNCERTAINTY CHECK block.
            # Each entry is (fast_steps+1, plan_tokens*fs, K, c) or None.
            uncertainty_plan_hists_per_candidate: list = [None] * len(valid_candidates)
            # Fallback buffer: always stores the last generated unc plan regardless of feasibility.
            # Used to fill uncertainty_plan_hists_per_candidate[i] when a candidate's plan
            # remains infeasible across all retries (mirrors the plan fallback at line ~3061).
            _fallback_unc_plan_hists: list = [None] * len(valid_candidates)

            for _ in range(
                self.num_tries_for_bad_plans
            ):  # resample when the generated plan is terrible (e.g., not moving plans)
                ###############################
                # Expansion
                expansion_start_time = time.time()
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
                    parent_levels
                )  # (b, m, plan_tokens(=t))
                expanded_node_updated_levels = expanded_node_noise_levels[
                    :, -1, :
                ]  # (b, plan_tokens(=t))

                # Tag current tree for guidance JSONL logging (forward vs backward)
                self._current_tree_tag = tree.tag

                # Expansion: single batched parallel_plan call over all B candidates.
                # group_ids assigns each candidate an integer sibling-group id (same parent → same id)
                # so that particle_guidance only repels within-sibling groups, not across parents.
                # output plan_hist: (m+1, plan_tokens*fs, B, c)
                _parent_to_gid: dict = {}
                _group_ids: list = []
                for _cand in expanded_node_candidates:
                    _pname = _cand["parent_node"].name
                    if _pname not in _parent_to_gid:
                        _parent_to_gid[_pname] = len(_parent_to_gid)
                    _group_ids.append(_parent_to_gid[_pname])
                # _group_ids: list[int] of length B — same value = same sibling group

                expanded_node_plan_hists = self.parallel_plan(
                    start=effective_obs_normalized,            # (B, obs_dim)
                    goal=effective_goal_normalized,            # (B, obs_dim)
                    horizon=horizon,
                    conditions=conditions,
                    guidance_scale=expanded_node_guidance_scales,   # (B,)
                    noise_level=expanded_node_noise_levels,         # (B, m, plan_tokens)
                    plans=expanded_node_plans,                      # list of B tensors
                    prefix_len_list=prefix_len_list,                # list of B ints
                    particle_guidance_scale=self.particle_guidance_scale,
                    group_ids=_group_ids,
                    call_type="expansion",
                )  # (m+1, plan_tokens*fs, B, c)

                # Build per-candidate step captures for video visualization.
                # Stored in self._expansion_step_captures_by_name (name → list of per-step dicts).
                _raw_captures = getattr(self, '_parallel_plan_step_captures', None)
                _exp_sc_by_name: dict = {}
                if _raw_captures:
                    for _sci, _cand in enumerate(expanded_node_candidates):
                        _exp_sc_by_name[_cand["name"]] = [
                            {
                                'prior_pred_noise': step['prior_pred_noise'][:, _sci] if step['prior_pred_noise'] is not None else None,
                                'guidance_grads': {k: v[:, _sci] for k, v in step['guidance_grads'].items()},
                                'guidance_grads_clean': {k: v[:, _sci] for k, v in step.get('guidance_grads_clean', {}).items()},
                                'pred_x_start_pos': step['pred_x_start_pos'][:, _sci] if step.get('pred_x_start_pos') is not None else None,
                                'noise_level': step['noise_level'][:, _sci] if step.get('noise_level') is not None else None,
                            }
                            for step in _raw_captures
                        ]
                self._expansion_step_captures_by_name = _exp_sc_by_name

                # [GUIDANCE LOGGING] Log guidance quality data to validation_anal_*.jsonl
                _g_losses = getattr(self, '_last_guidance_losses', {})
                if _g_losses or expanded_node_guidance_scales is not None:
                    # Compute per-batch distance from final plan token to goal (unnormalized positions)
                    _final_plan = self._unnormalize_x(expanded_node_plan_hists[-1])  # (plan_tokens*fs, B, c)
                    _goal_unnorm = self._unnormalize_x(effective_goal_normalized)    # (B, obs_dim)
                    _final_pos = _final_plan[-1, :, self.pos_dim_indices].detach().cpu().numpy()  # (B, pos_dim)
                    _goal_pos = _goal_unnorm[:, self.pos_dim_indices].detach().cpu().numpy()       # (B, pos_dim)
                    _dist_per_batch = np.linalg.norm(_final_pos - _goal_pos, axis=-1).tolist()  # [B]
                    _scales = (
                        expanded_node_guidance_scales.tolist()
                        if hasattr(expanded_node_guidance_scales, 'tolist')
                        else list(expanded_node_guidance_scales)
                        if expanded_node_guidance_scales is not None
                        else []
                    )
                    _eff_scale = float(np.mean(_scales)) if _scales else 0.0
                    self._glog("guidance.combined", {
                        "tree_tag": tree.tag,
                        "search_num": tree.search_num,
                        "eff_goal_scale": _eff_scale,
                        "batch_size": len(expanded_node_candidates),
                        "dist_per_batch": [round(d, 4) for d in _dist_per_batch],
                        "final_token_dist": round(float(np.mean(_dist_per_batch)), 4),
                        "anchor_loss": round(_g_losses.get("anchor", 0.0), 6),
                        "goal_loss": round(_g_losses.get("goal", 0.0), 6),
                        "rdf_loss": round(_g_losses.get("rdf", 0.0), 6),
                        "particle_loss": round(_g_losses.get("particle", 0.0), 6),
                        "goal_anchor_ratio": round(
                            abs(_g_losses.get("goal", 0.0)) / (abs(_g_losses.get("anchor", 1e-9)) + 1e-9), 4
                        ),
                    }, depth=1)

                # [GRAD COMPARISON] Log prior norm vs guidance grad norm from last denoising step
                if _raw_captures and len(_raw_captures) > 1:
                    _last_cap = _raw_captures[-1]
                    _pn = _last_cap.get('prior_pred_noise')  # (plan_tokens*fs, B, c) or None
                    _gg = _last_cap.get('guidance_grads', {})
                    if _pn is not None and _gg:
                        _prior_norm = float(_pn.norm().item())
                        _guidance_norms = {k: round(float(v.norm().item()), 6) for k, v in _gg.items()}
                        _guidance_total = sum(_guidance_norms.values())
                        self._glog("diffusion.grad_comparison", {
                            "tree_tag": tree.tag,
                            "search_num": tree.search_num,
                            "prior_norm": round(_prior_norm, 6),
                            "guidance_total_norm": round(_guidance_total, 6),
                            "ratio": round(_guidance_total / (_prior_norm + 1e-9), 4),
                            "per_guidance_norms": _guidance_norms,
                        }, depth=1)

                # Validate expanded_node_plan_hists shape: (m, plan_tokens*fs, B, c)
                assert expanded_node_plan_hists.ndim == 4, (
                    f"expanded_node_plan_hists.ndim={expanded_node_plan_hists.ndim}, expected 4"
                )
                assert expanded_node_plan_hists.shape[2] == len(
                    expanded_node_candidates
                ), (
                    f"expanded_node_plan_hists.shape[2]={expanded_node_plan_hists.shape[2]}, expected {len(expanded_node_candidates)}"
                )

                expansion_end_time = time.time()
                tree.expansion_time.append(expansion_end_time - expansion_start_time)

                ###############################
                # Replanning
                #  It includes the noise level zero-padding, finding the max denoising steps, replanning, value calculation and node allocation
                replanning_start_time = time.time()

                def is_feasible_and_not_short_plan_hists(plan_hists): # plan_hists: (m, plan_tokens*fs, b, c)
                    plans = (
                        self._unnormalize_x(plan_hists[-1])[:-1]
                        .detach()
                        .cpu()
                        .numpy()
                    )  # (t*fs-1, b, c)

                    # Prepend each candidate's parent obs as frame 0 so that the
                    # parent→first_plan_token continuity is also checked.
                    parent_obs_list = [
                        expanded_node_candidates[i]["parent_node"].obs
                        for i in range(plans.shape[1])
                    ]
                    if all(p is not None for p in parent_obs_list):
                        parent_obs_np = np.stack(parent_obs_list, axis=0)[np.newaxis]  # (1, b, c)
                        plans = np.concatenate([parent_obs_np, plans], axis=0)  # (t*fs, b, c)

                    diffs = np.linalg.norm(
                        plans[1:] - plans[:-1], axis=-1
                    )  # (plan_len-1, b)

                    # FIX: is_feasible size should match diffs.shape[1] (number of plans), not expanded_node_candidates
                    is_feasible = [False] * diffs.shape[1]
                    for i in range(diffs.shape[1]):
                        is_feasible[i] = np.all(
                            diffs[:, i] < self.plan_feasibility_delta
                        )

                    # Progress filter: kill plans whose sub_plan start-end distance is too small
                    if self.min_progress_threshold > 0.0:
                        for i in range(diffs.shape[1]):
                            if is_feasible[i]:
                                parent_node = expanded_node_candidates[i]["parent_node"]
                                if parent_node.obs is not None:
                                    start_xy = parent_node.obs[:2]
                                    end_xy = plans[-1, i, :2]
                                    progress = float(np.linalg.norm(end_xy - start_xy))
                                    if progress < self.min_progress_threshold:
                                        is_feasible[i] = False

                    return is_feasible


                if not self.mcts_use_replan:
                    # Skip replanning: use expansion results directly for value.
                    assert prefix_len_list is not None
                    batch_size = expanded_node_plan_hists.shape[2]
                    plans_tokens = rearrange(expanded_node_plan_hists, "m (t fs) b c -> m t fs b c", fs=self.frame_stack)
                    num_tokens_to_check = seg_size

                    # All candidates in a batch come from parents at the same depth, so prefix_len
                    # is uniform across the batch. Use a simple slice to avoid PyTorch non-adjacent
                    # advanced indexing, which swaps the indexed dimensions to the front and produces
                    # shape (T_check, B, m, fs, c) instead of the expected (m, T_check, fs, B, c).
                    assert len(set(prefix_len_list)) == 1, \
                        f"Expected uniform prefix_len across batch, got {prefix_len_list}"
                    plen = prefix_len_list[0]
                    t_end = min(plen + num_tokens_to_check, plans_tokens.shape[1])
                    sliced_hists = plans_tokens[:, plen:t_end, :, :, :]  # (m, T_check, fs, B, c)
                    processed_hists = rearrange(sliced_hists, "m t fs b c -> m (t fs) b c")

                    is_feasible = is_feasible_and_not_short_plan_hists(processed_hists)
                    # val_plan_hists: plan used for value / filtering (no replan → same as expansion)
                    val_plan_hists = expanded_node_plan_hists

                else: # REPLANNING to get more feasible trajectory
                    # Pad the noise levels - Sequential
                    replan_noiselevel_zero_padding_start = time.time()
                    replan_init_plans = []
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
                        _parent_node = expanded_node_candidates[i]["parent_node"]
                        _sim_plan, _ = self._build_plan_from_leaf(
                            _parent_node, _plan_tokens_val, seg_size, expanded_plan=_plan_rearranged
                        )  # (n_tokens, 1, fs*c)
                        replan_init_plans.append(_sim_plan)

                    replan_noiselevel_zero_padding_end = time.time()
                    tree.replan_noiselevel_zero_padding_time.append(
                        replan_noiselevel_zero_padding_end
                        - replan_noiselevel_zero_padding_start
                    )

                    # Replanning - Diffusion
                    replan_diffusion_start = time.time()

                    # Prepare expanded node's denoising state for replanning
                    replan_initial_levels_list = []
                    for i in range(len(expanded_node_candidates)):
                        if expanded_node_updated_levels is not None:
                            replan_initial_levels_list.append(
                                expanded_node_updated_levels[i : i + 1]
                            )
                        else:
                            replan_initial_levels_list.append(None)

                    assert expanded_node_updated_levels is not None, (
                        "expanded_node_updated_levels must be set"
                    )
                    replan_initial_levels = np.concatenate(
                        replan_initial_levels_list, axis=0
                    )  # (b, plan_tokens)
                    # Generate Schedule for Replanning (Complete Denoising)
                    replan_noise_levels = (
                        self._generate_bidirectional_schedule(
                            replan_initial_levels,
                            is_replanning = True
                        )
                    )  # (b, m, plan_tokens)

                    # input plans: list of (n_tokens, 1, fs*c)
                    # output plan_hists: (m+1, plan_tokens*fs, b, c)
                    replanned_plan_hists = self.parallel_plan(
                        effective_obs_normalized,
                        effective_goal_normalized,
                        horizon,
                        conditions,
                        guidance_scale=torch.zeros_like(expanded_node_guidance_scales), # None,
                        noise_level=replan_noise_levels,
                        plans=replan_init_plans,
                        prefix_len_list=prefix_len_list,
                        call_type="replan",
                    )

                    # Validate replanned_plan_hists shape: (m, plan_tokens*fs, B, c)
                    assert replanned_plan_hists.ndim == 4, (
                        f"replanned_plan_hists.ndim={replanned_plan_hists.ndim}, expected 4"
                    )
                    assert replanned_plan_hists.shape[2] == len(
                        expanded_node_candidates
                    ), (
                        f"replanned_plan_hists.shape[2]={replanned_plan_hists.shape[2]}, expected {len(expanded_node_candidates)}"
                    )

                    replan_diffusion_end = time.time()

                    is_feasible = is_feasible_and_not_short_plan_hists(replanned_plan_hists)
                    # val_plan_hists: plan used for value / filtering (replan → replanned)
                    val_plan_hists = replanned_plan_hists

                ##### NODE UNCERTAINTY CHECK #####

                # uncertainty_plan_hists_per_candidate is initialized before the retry loop.
                # Here we populate it for each candidate when use_uncertainty_as_value=True.
                # Input: val_plan_hists (replanned if mcts_use_replan, else expanded).
                if self.use_uncertainty_as_value:
                    B_cands = len(expanded_node_candidates)
                    K = self.fast_sampling_multiple

                    # Build fast noise schedule using fast_sampling_steps.
                    # Same starting levels as replanning (expanded_node_updated_levels).
                    assert expanded_node_updated_levels is not None, (
                        "expanded_node_updated_levels must be set for uncertainty sampling"
                    )
                    unc_noise_levels = self._generate_bidirectional_schedule(
                        expanded_node_updated_levels,
                        num_denoising_steps_override=self.fast_sampling_steps,
                    )  # (B_cands, fast_steps+1, plan_tokens)

                    # Build G*K independent init plans per candidate.
                    # Batch order: [cand0·g0·k0, cand0·g0·k1, …, cand0·g1·k0, …, cand_{B-1}·g_{G-1}·k_{K-1}]
                    # Strategy: build ONE base plan per candidate from val_plan_hists (the replanned/
                    # expanded plan), preserving the denoised prefix + obs_parent + first replanned
                    # segment.  For each K copy, clone base and replace only the future undenoised
                    # tokens [child_depth*seg_size+1 : plan_tokens+1] with fresh randn.  This keeps
                    # the already-denoised trajectory intact while giving K independently diverse
                    # future trajectories.
                    G = len(self.mctd_guidance_scales)
                    _plan_tokens_val = horizon // self.frame_stack
                    unc_noise_levels_rep = np.repeat(unc_noise_levels, K * G, axis=0)  # (B*G*K, fast_steps+1, plan_tokens)
                    unc_init_plans_rep = []
                    unc_guidance_scale_vals = []  # per-sample guidance scale (length B*G*K)
                    for _i in range(B_cands):
                        _parent_node = expanded_node_candidates[_i]["parent_node"]
                        # Build base plan from val_plan_hists (replanned or expanded plan)
                        _plan_t_fs = val_plan_hists[-1, :, _i].unsqueeze(1)  # (plan_tokens*fs, 1, c)
                        _plan_rearranged_unc = rearrange(
                            _plan_t_fs, "(t fs) b c -> t b (fs c)", fs=self.frame_stack
                        )  # (plan_tokens, 1, fs*c)
                        _base_plan, _ = self._build_plan_from_leaf(
                            _parent_node, _plan_tokens_val, seg_size,
                            expanded_plan=_plan_rearranged_unc,
                        )  # (n_tokens, 1, fs*c)
                        # Future undenoised tokens: child_depth*seg_size+1 onward (token-space index)
                        _child_depth = _parent_node.depth + 1
                        _future_start = _child_depth * seg_size + 1
                        _n_future = _plan_tokens_val + 1 - _future_start
                        for g_scale in self.mctd_guidance_scales:
                            for _ in range(K):
                                _unc_plan = _base_plan.clone()
                                if _n_future > 0:
                                    _unc_plan[_future_start : _plan_tokens_val + 1] = (
                                        self._sample_clamped_noise(_n_future)
                                    )
                                unc_init_plans_rep.append(_unc_plan)
                                unc_guidance_scale_vals.append(g_scale)
                    unc_guidance_scale_tensor = torch.tensor(
                        unc_guidance_scale_vals, dtype=torch.float32, device=self.device
                    )  # (B*G*K,)

                    # Run parallel_plan for fast uncertainty sampling.
                    # Replicate obs/goal/prefix_len to match B*G*K batch size.
                    unc_obs = effective_obs_normalized.repeat_interleave(G * K, dim=0)   # (B*G*K, D)
                    unc_goal = effective_goal_normalized.repeat_interleave(G * K, dim=0)  # (B*G*K, D)
                    unc_prefix_len_list = [pl for pl in prefix_len_list for _ in range(G * K)]
                    unc_batch_plan_hists = self.parallel_plan(
                        unc_obs,
                        unc_goal,
                        horizon,
                        conditions,
                        guidance_scale=unc_guidance_scale_tensor,
                        noise_level=unc_noise_levels_rep,
                        plans=unc_init_plans_rep,
                        prefix_len_list=unc_prefix_len_list,
                        call_type="uncertainty",
                    )  # (fast_steps+1, plan_tokens*fs, B*G*K, c)

                    # Reshape into per-candidate tensors: (fast_steps+1, plan_tokens*fs, G*K, c).
                    for _i in range(B_cands):
                        start_idx_k = _i * G * K
                        end_idx_k = start_idx_k + G * K
                        unc_plan = unc_batch_plan_hists[:, :, start_idx_k:end_idx_k, :]
                        # Always save the latest generated plan as fallback (regardless of feasibility).
                        # This ensures uncertainty_plan_hists_per_candidate[i] is never None after
                        # the retry loop, even if the expansion plan was infeasible across all retries.
                        _fallback_unc_plan_hists[_i] = unc_plan
                        uncertainty_plan_hists_per_candidate[_i] = unc_plan


                for i in range(len(is_feasible)):
                        if is_feasible[i] and filtered_expanded_node_plan_hists[i] is None:
                            filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                            filtered_replanned_plan_hists[i] = val_plan_hists[:, :, i]

                # Kill uncertainty plans for candidates that didn't pass the feasibility filter.
                # Alive plans (filtered_expanded_node_plan_hists[i] is not None) keep their
                # uncertainty plans; dead plans discard theirs.
                for _i in range(len(expanded_node_candidates)):
                    if filtered_expanded_node_plan_hists[_i] is None:
                        uncertainty_plan_hists_per_candidate[_i] = None

                if None in filtered_expanded_node_plan_hists:
                    replanning_end_time = time.time()
                    tree.replanning_time.append(
                        replanning_end_time - replanning_start_time
                    )
                    continue
                else:
                    break
                

            # ----------------------self.num_tries_for_bad_plans LOOP END----------------------------------------

            # Track which candidates had a feasible plan across all retries (before fallback fill)
            final_is_feasible = [fh is not None for fh in filtered_expanded_node_plan_hists]  # len B

            for i in range(len(filtered_expanded_node_plan_hists)):
                if filtered_expanded_node_plan_hists[i] is None:
                    filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[
                        :, :, i
                    ]
                    filtered_replanned_plan_hists[i] = (
                        replanned_plan_hists[:, :, i] if self.mcts_use_replan else expanded_node_plan_hists[:, :, i]
                    )
                    # [A4] Inject fallback uncertainty plan for infeasible candidates.
                    # uncertainty_plan_hists_per_candidate[i] is killed (set to None) after each
                    # failed retry; _fallback_unc_plan_hists[i] always holds the last generated one.
                    if self.use_uncertainty_as_value and uncertainty_plan_hists_per_candidate[i] is None:
                        uncertainty_plan_hists_per_candidate[i] = _fallback_unc_plan_hists[i]
            expanded_node_plan_hists = torch.stack(
                filtered_expanded_node_plan_hists, dim=2
            )  # m (t fs) 'B' c
            replanned_plan_hists = torch.stack(
                filtered_replanned_plan_hists, dim=2
            )  # m (t fs) 'B' c

            # Value Calculation
            replan_value_calculation_start = time.time()
            achieved_indices = []
            final_best_plans = replanned_plan_hists[-1]  # (plan_tokens*fs, b, c)

            unc_results: dict = {}  # candidate index → _compute_node_uncertainty result dict

            if self.use_uncertainty_as_value:
                # [A5] Achieved detection without Warp: value is decided solely by uncertainty.
                # Warp is not checked — only goal proximity matters for recording Achieved.
                achieved_infos, achieved_ts = self._check_achieved_bidir(
                    expanded_node_candidates, final_best_plans, tree
                )
                for i in range(len(achieved_infos)):
                    if achieved_infos[i] == "Achieved":
                        tree.achieved = True
                        achieved_indices.append(i)

                # Compute -uncertainty as value for each candidate.
                values = np.zeros(len(expanded_node_candidates))
                for i, candidate in enumerate(expanded_node_candidates):
                    target_node_i = candidate["target_node"]
                    curr_depth_i = candidate["depth"]
                    unc_hists_i   = uncertainty_plan_hists_per_candidate[i]
                    assert unc_hists_i is not None, (
                        f"[Uncertainty] uncertainty_plan_hists_per_candidate[{i}] is None "
                        "after fallback injection — this should not happen."
                    )
                    # Each K copy was built with fresh noise (expanded_plan=None) → G*K
                    # samples are independent.  Use all G*K for sigma estimation.
                    tail_obs_i = self._extract_obs_at_boundary(
                        unc_hists_i[-1],           # (plan_tokens*fs, G*K, c)
                        depth=curr_depth_i + 1,   # next segment boundary (undenoised region)
                        seg_size=seg_size,
                    )  # (G*K, obs_dim)
                    # Extract current node's obs from the plan at depth=curr_depth_i boundary.
                    # candidate["obs"] is set later (post-dedup), so we derive it here directly.
                    curr_obs_i = self._extract_obs_at_boundary(
                        expanded_node_plan_hists[-1, :, i].unsqueeze(1),  # (plan_tokens*fs, 1, c)
                        depth=curr_depth_i,
                        seg_size=seg_size,
                    )[0]  # (obs_dim,)
                    unc_result = self._compute_node_uncertainty(
                        curr_obs=curr_obs_i,
                        target_node=target_node_i,
                        tail_obs=tail_obs_i,
                    )
                    values[i] = -unc_result['U']
                    unc_results[i] = unc_result  # store for visualization
            else:
                values, achieved_infos, achieved_ts = self.calculate_values_bidir(
                    expanded_node_candidates, final_best_plans, tree
                )
                for i in range(len(achieved_infos)):
                    if achieved_infos[i] == "Achieved":
                        tree.achieved = True
                        achieved_indices.append(i)

            replan_value_calculation_end = time.time()

            # Endpoint Deduplication: extract obs per candidate, then kill duplicates
            candidate_obses = []  # list of np.ndarray (observation_dim,), len B
            for i in range(len(expanded_node_candidates)):
                child_depth = expanded_node_candidates[i]["depth"]
                plan_hists_last_i = expanded_node_plan_hists[-1, :, i]  # (plan_tokens*fs, c)
                obs_i = self._extract_obs_at_boundary(
                    plan_hists_last_i.unsqueeze(1),  # (plan_tokens*fs, 1, c)
                    depth=child_depth,
                    seg_size=seg_size,
                )[0]  # (observation_dim,)
                candidate_obses.append(obs_i)

            is_kept = self._deduplicate_by_endpoint(
                expanded_node_candidates, candidate_obses, final_is_feasible,
            )  # list of bool, len B

            # Node Allocation (skip deduplicated/infeasible candidates)
            replan_node_allocation_start = time.time()
            # Dedup-killed slots are permanently dead: never attempt to expand them again.
            # Kept slots that weren't expanded also release their virtually_visited lock.
            for i in range(len(expanded_node_candidates)):
                name = expanded_node_candidates[i]["name"]
                slot_index = int(name.split('-')[-1])
                parent_node = expanded_node_candidates[i]["parent_node"]
                if not is_kept[i]:
                    parent_node.mark_slot_permanently_dead(slot_index)
                else:
                    parent_node._children_nodes[slot_index]["virtually_visited"] = False

            selected_nodes_for_expansion = {}
            expanded_node_infos = {}
            for i in range(len(expanded_node_candidates)):  # B
                if not is_kept[i]:
                    continue  # slot remains empty, available for future expansion rounds
                name = expanded_node_candidates[i]["name"]
                if name not in expanded_node_infos:
                    selected_nodes_for_expansion[name] = selected_nodes[i]
                    expanded_node_infos[name] = expanded_node_candidates[i]
                    expanded_node_infos[name]["plan_history"].append([])
                    # Note: is_tree1 is a property of MCTSTreeState (tree), not individual nodes
                value = values[i]
                plan_hist = expanded_node_plan_hists[:, :, i] if not self.mcts_use_replan else replanned_plan_hists[:, :, i] # m (t fs) c
                expanded_plan_hist_i = expanded_node_plan_hists[:, :, i]  # always expanded stage
                replanned_plan_hist_i = replanned_plan_hists[:, :, i] if self.mcts_use_replan else None  # replanning stage if available
                replanned_plan = replanned_plan_hists[-1, :, i]

                # Store updated denoising state for child node
                if expanded_node_updated_levels is not None:
                    updated_level = expanded_node_updated_levels[
                        i : i + 1
                    ]  # Shape: (1, plan_tokens)
                else:
                    updated_level = None

                if expanded_node_infos[name]["value"] is None:
                    expanded_node_infos[name]["value"] = value
                    expanded_node_infos[name]["replanned_plan"] = (
                        replanned_plan
                    )
                    expanded_node_infos[name]["plan_history"][-1] = (
                        plan_hist  # d m (t fs) c
                    )
                    expanded_node_infos[name]["expanded_plan_hist_frame"] = expanded_plan_hist_i
                    expanded_node_infos[name]["replanned_plan_hist_frame"] = replanned_plan_hist_i
                    expanded_node_infos[name]["uncertainty_plan_hist_frame"] = uncertainty_plan_hists_per_candidate[i]
                    expanded_node_infos[name]["unc_diagnostics"] = unc_results.get(i)
                    expanded_node_infos[name]["current_levels"] = updated_level
                else:
                    if value > expanded_node_infos[name]["value"]:
                        expanded_node_infos[name]["value"] = value
                        expanded_node_infos[name]["replanned_plan"] = (
                            replanned_plan
                        )
                        expanded_node_infos[name]["plan_history"][-1] = plan_hist
                        expanded_node_infos[name]["expanded_plan_hist_frame"] = expanded_plan_hist_i
                        expanded_node_infos[name]["replanned_plan_hist_frame"] = replanned_plan_hist_i
                        expanded_node_infos[name]["uncertainty_plan_hist_frame"] = uncertainty_plan_hists_per_candidate[i]
                        expanded_node_infos[name]["unc_diagnostics"] = unc_results.get(i)
                        expanded_node_infos[name]["current_levels"] = updated_level

            for name in selected_nodes_for_expansion:
                parent_node_for_expand = selected_nodes_for_expansion[name]
                expand_kwargs = {k: v for k, v in expanded_node_infos[name].items()
                                 if k not in ("expanded_plan_hist_frame", "replanned_plan_hist_frame", "uncertainty_plan_hist_frame", "unc_diagnostics")}
                child_node = parent_node_for_expand.expand(
                    **expand_kwargs
                )
                expanded_node_infos[name]["node"] = child_node

            replan_node_allocation_end = time.time()
            tree.replan_node_allocation_time.append(
                replan_node_allocation_end - replan_node_allocation_start
            )

            replanning_end_time = time.time()
            tree.replanning_time.append(replanning_end_time - replanning_start_time)

            ######################
            # Backpropagation
            #  When leaf parallelization is True, then the backpropagation is done in partially parallel (the leafs from same parent node are backpropagated at the same time)
            #  When leaf parallelization is False, then the backpropagation is done in fully sequential (only one node is backpropagated at a time)
            backprop_start_time = time.time()

            # Only backpropagate nodes that actually had children created this round
            distinct_selected_nodes = np.unique(selected_nodes)
            for selected_node in distinct_selected_nodes:
                if any(c["node"] is not None for c in selected_node._children_nodes):
                    selected_node.backpropagate()

            backprop_end_time = time.time()
            tree.backprop_time.append(backprop_end_time - backprop_start_time)

            ######################
            # Early Termination
            early_termination_start_time = time.time()

            tree.search_num += 1
            tree.p_search_num += len(expanded_node_candidates)
            tree.pbar.update(len(expanded_node_candidates))
            tree.max_depth = max(
                tree.max_depth,
                max([info["depth"] for info in expanded_node_candidates]),
            )

            is_early_termination = tree.achieved

            _viz_fp_t0 = time.time()
            _viz_fp_node_ms = 0.0
            _viz_fp_compare_ms = 0.0
            if self.viz_final_plans:
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

                if len(visualize_indices) > 0:
                    terminal_values = values[visualize_indices]
                    terminal_names = [
                        expanded_node_candidates[i]["name"] for i in visualize_indices
                    ]
                    terminal_expanded_plans = expanded_node_plan_hists[
                        -1, :, visualize_indices
                    ]  # m (t fs) b c
                    is_achieved_plan = [True if i in achieved_indices else False for i in visualize_indices]

                    # Compute how many tokens are actually denoised per candidate (depth-based)
                    visualize_denoised_lens = [
                        self._get_prefix_len_frames_from_depth(expanded_node_candidates[i]["depth"], seg_size)
                        for i in visualize_indices
                    ]

                    # For goal tree visualization, flip the plans so they appear in start→goal direction
                    if "from_goal" in tree.tag:
                        terminal_expanded_plans = torch.flip(terminal_expanded_plans, [0])

                    _viz_node_t0 = time.time()
                    self.visualize_node_value_plans(
                        is_achieved_plan,
                        tree.search_num,
                        terminal_values,
                        terminal_names,
                        terminal_expanded_plans,
                        start,
                        goal,
                        tag=tree.tag,
                        denoised_lens=visualize_denoised_lens,
                    )
                    _viz_fp_node_ms = (time.time() - _viz_node_t0) * 1000

                    if self.viz_compare_expanded_to_value:
                        _viz_cmp_t0 = time.time()
                        terminal_value_plans = replanned_plan_hists[
                            -1, :, visualize_indices
                        ]  # (t fs) b c
                        if "from_goal" in tree.tag:
                            terminal_value_plans = torch.flip(terminal_value_plans, [0])
                        self.visualize_expanded_vs_value_plans(
                            is_achieved_plan,
                            terminal_names,
                            terminal_expanded_plans,
                            terminal_value_plans,
                            start,
                            goal,
                        )
                        _viz_fp_compare_ms = (time.time() - _viz_cmp_t0) * 1000
            _viz_fp_total_ms = (time.time() - _viz_fp_t0) * 1000
            self._tlog("timing.viz_final_plans", {
                "node_value_ms": round(_viz_fp_node_ms, 1),
                "compare_ms": round(_viz_fp_compare_ms, 1),
                "total_ms": round(_viz_fp_total_ms, 1),
            }, depth=1)

            early_termination_end_time = time.time()
            tree.early_termination_time.append(
                early_termination_end_time - early_termination_start_time
            )

            if is_early_termination:
                break

            # ------------------------------------------------------------------
            # single_step mode: exit after 1 iteration (expanded_node_infos already set)
            # ------------------------------------------------------------------
            if single_step:
                # Log per-expansion timing so FWD and BWD both appear in the jsonl.
                # timing.phase_breakdown (logged below) gives cumulative totals; this
                # record gives one row per expansion so you can track iteration-by-iteration.
                if tree.expansion_time:
                    self._tlog("timing.expansion", {
                        "tree_tag": tree.tag,
                        "expansion_idx": tree.search_num,
                        "expansion_ms": round(tree.expansion_time[-1] * 1000, 1),
                        "selection_ms": round(tree.selection_time[-1] * 1000, 1) if tree.selection_time else 0.0,
                        "replanning_ms": round(tree.replanning_time[-1] * 1000, 1) if tree.replanning_time else 0.0,
                        "backprop_ms": round(tree.backprop_time[-1] * 1000, 1) if tree.backprop_time else 0.0,
                        "early_term_ms": round(tree.early_termination_time[-1] * 1000, 1) if tree.early_termination_time else 0.0,
                    }, depth=0)
                break

        if not single_step:
            tree.pbar.close()

        # [LOGGING] Record search completion → guidance_anal (tree quality) + timing_anal (phase breakdown)
        terminal_depth_reached = tree.max_depth >= tree.terminal_depth
        self._glog("tree.search.complete", {
            "tree_tag": tree.tag,
            "final_search_num": tree.search_num,
            "final_max_depth": tree.max_depth,
            "terminal_depth": tree.terminal_depth,
            "terminal_depth_reached": terminal_depth_reached,
            "total_nodes_expanded": tree.search_num,
        }, depth=0)

        # [TIMING SUMMARY] Phase breakdown for bottleneck analysis → timestamp_anal
        def _ms(lst): return round(sum(lst) * 1000, 1) if lst else 0.0
        def _ms_mean(lst): return round((sum(lst) / len(lst)) * 1000, 1) if lst else 0.0
        sel_ms   = _ms(tree.selection_time)
        exp_ms   = _ms(tree.expansion_time)
        replan_ms = _ms(tree.replanning_time)
        bp_ms    = _ms(tree.backprop_time)
        et_ms    = _ms(tree.early_termination_time)
        nzp_ms   = _ms(tree.replan_noiselevel_zero_padding_time)
        val_ms   = _ms(tree.replan_diffusion_time)
        calc_ms  = _ms(tree.replan_value_calculation_time)
        alloc_ms = _ms(tree.replan_node_allocation_time)
        total_ms = sel_ms + exp_ms + replan_ms + bp_ms + et_ms
        self._tlog("timing.phase_breakdown", {
            "tree_tag": tree.tag,
            "n_iters": tree.search_num,
            "n_nodes": tree.p_search_num,
            "parallel_search_num": self.parallel_search_num,
            "total_ms": total_ms,
            "selection_ms": sel_ms,
            "expansion_ms": exp_ms,
            "replanning_ms": replan_ms,
            "backprop_ms": bp_ms,
            "early_term_ms": et_ms,
            "replan_noise_zeropad_ms": nzp_ms,
            "replan_diffusion_ms": val_ms,
            "replan_value_calculation_ms": calc_ms,
            "replan_node_allocation_ms": alloc_ms,
            "mean_selection_ms": _ms_mean(tree.selection_time),
            "mean_expansion_ms": _ms_mean(tree.expansion_time),
            "mean_replanning_ms": _ms_mean(tree.replanning_time),
            "mean_backprop_ms": _ms_mean(tree.backprop_time),
            "expansion_ratio": round(exp_ms / total_ms, 3) if total_ms > 0 else 0,
            "replanning_ratio": round(replan_ms / total_ms, 3) if total_ms > 0 else 0,
            "selection_ratio": round(sel_ms / total_ms, 3) if total_ms > 0 else 0,
        }, depth=0)

        return tree, expanded_node_infos

    # =========================================================================
    # Helper functions for bidirectional alternating MCTS
    # =========================================================================

    def _sample_clamped_noise(self, n_tokens: int, batch_size: int = 1) -> torch.Tensor:
        """Sample (n_tokens, batch_size, *x_stacked_shape) randn clamped to clip_noise."""
        noise = torch.randn((n_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        return torch.clamp(noise, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)

    def _build_obs_parent_token(self, parent_node: "TreeNode") -> torch.Tensor:
        """Build the obs_parent_token (position 0 anchor) for a plan tensor.

        Returns shape (1, 1, fs*c) normalized, ready to be prepended to a plan.
        Respects use_dynamic_obs_padding: when False, walks to the root node.
        """
        if self.use_dynamic_obs_padding:
            padding_node = parent_node
        else:
            padding_node = parent_node
            while padding_node._parent_node is not None:
                padding_node = padding_node._parent_node

        parent_obs = padding_node.obs  # Raw (unnormalized) world coordinate

        # Method B: use stored observation_mean/std directly (no data_mean slicing)
        # obs is already indexed by obs_dim_indices, no further slicing needed
        obs_mean_np = np.array(self.observation_mean)
        obs_std_np = np.array(self.observation_std)
        parent_obs_normalized = (parent_obs - obs_mean_np) / obs_std_np

        parent_obs_tensor = torch.tensor(
            parent_obs_normalized, dtype=torch.float32, device=self.device
        )
        obs_parent_token_raw = self.pad_init(
            parent_obs_tensor.unsqueeze(0)
        )  # (fs, 1, c) - normalized
        obs_parent_token = rearrange(
            obs_parent_token_raw, "fs b c -> 1 b (fs c)"
        )  # (1, 1, fs*c) - normalized
        return obs_parent_token

    def _build_plan_from_leaf(
        self,
        parent_node: "TreeNode",
        plan_tokens: int,
        segment_size: int,
        expanded_plan: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        # Assembles a diffusion sequence: [prior trajectory | current obs | random noise | padding]
        """Construct the full plan_with_given_tokens for a new leaf node expansion.

        Returns a tensor of shape (n_tokens, 1, fs*c) with layout:
            [denoised_prefix(prefix_len) | obs_parent_token(1) | noisy_chunk | padding]
        When denoised_prefix is empty (root depth=0):
            [obs_parent_token(1) | noisy_chunk | padding]

        When expanded_plan (plan_tokens, 1, fs*c) is provided (replanning),
        the full assembly flow is identical — only noisy_parts is replaced with
        expanded_plan[prefix_len:] instead of random noise. use_dynamic_obs_padding
        branching and padding are applied unchanged.

        This output is ready to be passed directly to parallel_plan (pre-built format).
        """
        # Structural guarantee: plan_history is ALWAYS a list (never None)
        # This is guaranteed by TreeNode.__init__ and get_expandable_candidate
        assert isinstance(parent_node.plan_history, list), \
            f"plan_history must be list, got {type(parent_node.plan_history)}"

        obs_parent_token = self._build_obs_parent_token(parent_node)  # (1, 1, fs*c)

        prefix_len = 0 # initial value

        # --- Build denoised prefix from parent's plan_history ---
        # Structural guarantee: plan_history is ALWAYS a list (initialized to [] in TreeNode.__init__)
        # - Root nodes: plan_history = [] (empty, evaluates to False)
        # - Non-root nodes: plan_history has accumulated plan segments (non-empty, evaluates to True)
        # This condition checks: "if parent is NOT a root node" to use accumulated prefix
        if parent_node.plan_history: # and self.use_dynamic_obs_padding:
            # plan_history stores plans in canonical (forward) order via flip_plan_for_insert_hist.
            latest_plan_canonical = parent_node.plan_history[-1][-1]  # (plan_tokens*fs, c)
            prefix_len = parent_node.depth * segment_size  # token 단위 (noisy_total, expanded_plan 슬라이싱에 사용)
            prefix_len_frames = self._get_prefix_len_frames_from_depth(parent_node.depth, segment_size)
            full_prefix_canonical = latest_plan_canonical[:prefix_len_frames].unsqueeze(
                1
            )  # (prefix_len*fs, 1, c)

            # plan_history already contains normalized data from diffusion model output.
            # Do NOT normalize again - plan_history is already in normalized space.
            full_prefix = full_prefix_canonical  # (prefix_len*fs, 1, c) - already normalized
            denoised_prefix = rearrange(
                full_prefix, "(t fs) b c -> t b (fs c)", fs=self.frame_stack
            )  # (prefix_len, 1, fs*c)

        else:
            denoised_prefix = None
            prefix_len = 0


        # if use_dynamic_obs_padding: plan_tokens_with_parent_obs= [prefix(prefix_len) | obs_parent(1) | noisy(plan_tokens-prefix_len)] Totally, plan_tokens + 1.
        # else: plan_tokens_with_parent_obs: [obs_parent(1) | prefix(prefix_len) | noisy(plan_tokens-prefix_len)] Totally, plan_tokens + 1.
        noisy_total = plan_tokens - prefix_len
        assert noisy_total >= 0, f"Noisy total must be non-negative: {noisy_total}"

        batch_size = obs_parent_token.shape[1]  # always 1 per leaf
        if expanded_plan is not None:
            # Simulation mode: use the already-denoised segment instead of random noise.
            # expanded_plan is the full plan_tokens output from the expansion step;
            # slice off the prefix portion that is already covered by denoised_prefix.
            assert expanded_plan.shape[0] == plan_tokens, (
                f"expanded_plan length {expanded_plan.shape[0]} != plan_tokens {plan_tokens}"
            )
            noisy_parts = expanded_plan[prefix_len:]  # (noisy_total, 1, fs*c)
        else:
            noisy_parts = self._sample_clamped_noise(noisy_total, batch_size)

        # Assemble plan_tokens-length chunk: [prefix | obs_parent | noisy]
        # obs_parent token is inserted only when using dynamic padding (parent's obs varies by depth)
        if denoised_prefix is not None and self.use_dynamic_obs_padding:
            plan_chunk_with_parent_obs = torch.cat(
                [denoised_prefix, obs_parent_token, noisy_parts],
                dim=0,  # (plan_tokens+1, 1, fs*c)
            )
        elif denoised_prefix is not None and not self.use_dynamic_obs_padding:
            plan_chunk_with_parent_obs = torch.cat(
                [obs_parent_token, denoised_prefix, noisy_parts],
                dim=0,  # (plan_tokens+1, 1, fs*c)
            )
        else:
            plan_chunk_with_parent_obs = torch.cat(
                [obs_parent_token, noisy_parts], dim=0
            )  # (plan_tokens+1, 1, fs*c)


        # Append zero-padding to reach n_tokens.
        # padding formula should be: pad_tokens = n_tokens - (plan_tokens+1)
        plan_chunk_len = plan_chunk_with_parent_obs.shape[0]
        pad_tokens = self.n_tokens - plan_chunk_len


        pad = torch.zeros(
            (pad_tokens, batch_size, *self.x_stacked_shape), device=self.device
        )

        result = torch.cat([plan_chunk_with_parent_obs, pad], dim=0)  # (n_tokens, 1, fs*c)

        assert plan_chunk_with_parent_obs.shape[0] == plan_tokens+1, (
            f"Plan chunk length mismatch: {plan_chunk_with_parent_obs.shape[0]} != {plan_tokens+1}"
        )
        assert pad_tokens >= 0, f"pad_tokens must be non-negative: {pad_tokens}"
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
        opposite_tree_all_nodes: List["TreeNode"],
    ) -> "TreeNode":
        """Select the best goal from the opposite tree's nodes using HILP value.

        Computes V(current_leaf_obs, candidate.obs) for each candidate in
        `opposite_tree_all_nodes` and returns the node with the highest value
        (i.e., temporally closest to `current_leaf_obs`).

        Args:
            current_leaf_obs: Unnormalized observation of the leaf node being expanded,
                              shape (obs_dim,).
            opposite_tree_all_nodes: List of TreeNode objects from the opposite tree
                                 (all nodes, not just leaves).

        Returns:
            best_node: The TreeNode from opposite_tree_all_nodes with the highest HILP value.
        """
        assert all(n.obs is not None for n in opposite_tree_all_nodes), \
            "All opposite_tree_all_nodes must have obs set before goal selection"
        targets = np.stack([n.obs for n in opposite_tree_all_nodes])  # (N, D) — unnormalized world coords
        obs_expanded = np.tile(current_leaf_obs, (targets.shape[0], 1))  # (N, D)
        values = self._compute_hilp_values(obs_expanded, targets, use_no_grad=True)

        best_idx = torch.argmax(values).item()
        best_value = values[best_idx].item()
        best_node = opposite_tree_all_nodes[best_idx]

        return best_node

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

        best_info = max(
            expanded_node_infos.values(),
            key=lambda info: info["value"]
            if info.get("value") is not None
            else float("-inf"),
        )
        
        return best_info

    def _node_path_label(
        self,
        expanding_node: "TreeNode",
        is_forward_tree: bool,
        target_node: Optional["TreeNode"],
    ) -> str:
        """Build a human-readable path label for plan visualization.

        The label always reads forward-root → ... → backward-root, with the
        currently-expanding node wrapped in parentheses, and the two trees
        separated by '_'.

        Examples (node names encoded as last component of the TreeNode name):
          Backward tree expanding "0-1-3", target forward "0-2-5"
            → "0-2-5_(3)-1-0"
          Forward tree expanding "0-2-5", target backward "0-1-3"
            → "0-2-(5)_3-1-0"

        Args:
            expanding_node: The TreeNode just expanded.
            is_forward_tree: True if expanding_node belongs to the forward (start-rooted) tree.
            target_node: The node in the opposite tree being targeted (may be None).

        Returns:
            Label string, e.g. "0-2-5_(3)-1-0".
        """
        exp_parts = expanding_node.name.split("-")  # e.g. ["0","1","3"]

        if target_node is not None:
            tgt_parts = target_node.name.split("-")  # e.g. ["0","2","5"]
        else:
            tgt_parts = ["?"]

        if is_forward_tree:
            # Forward side: root → ... → (expanding_node)  e.g. "0-2-(5)"
            fwd_parts = exp_parts[:-1] + [f"({exp_parts[-1]})"]
            fwd_label = "-".join(fwd_parts)

            # Backward side: target_node → ... → backward_root  e.g. "3-1-0"
            bwd_label = "-".join(reversed(tgt_parts))
        else:
            # Forward side: root → ... → target_node  e.g. "0-2-5"
            fwd_label = "-".join(tgt_parts)

            # Backward side: (expanding_node) → ... → backward_root  e.g. "(3)-1-0"
            bwd_parts = list(reversed(exp_parts))
            bwd_parts[0] = f"({bwd_parts[0]})"
            bwd_label = "-".join(bwd_parts)

        return f"{fwd_label}_{bwd_label}"

