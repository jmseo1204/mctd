from typing import Optional, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import contextlib
import io
import math
import os
import time
import json
import os
import numpy as np
from random import random
import random as py_random
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
from utils.planning_utils import episode_len_to_plan_tokens
from utils.route_metric_utils import (
    anchor_short_label,
    compute_pairwise_temporal_distance_matrix,
    solve_fixed_endpoint_hamiltonian_path,
    solve_fixed_endpoint_hamiltonian_path_with_forced_adjacency,
)
from utils.task_override_loader import (
    apply_task_overrides_to_env,
    inject_task_metadata_into_reset_info,
    load_task_override_payload,
    resolve_task_override_path,
)
from .tree_node import TreeNode
from . import guidance
from .hilp_loader import HILPMemoizedWrapper, get_hilp_fn
from .env_executor import PlanExecutorMixin
from .plan_postproc import PlanPostprocMixin
from .kde_estimator import KDEEstimatorMixin
from .noise_schedule import NoiseScheduleMixin
from .plan_viz import PlanVizMixin  # owns visualization-only frame/window helpers used via self.*
from .sampled_graph_estimator import (
    SampledGraphEstimatorMixin,
    extract_shortest_path_submatrix,
    query_distinct_nearest_nodes,
    query_shortest_path_between_node_indices,
)
from .uncertainty_estimator import compute_uncertainty_from_embeddings, compute_uncertainty_variance, cluster_tail_by_temporal_dist

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

GROUNDTRUTH_TEMPORAL_VIZ_GAMMA = 0.995
GROUNDTRUTH_TEMPORAL_VIZ_GRID_RES = 100
GROUNDTRUTH_TEMPORAL_VIZ_GRAD_GRID_STEP = 2.0


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "t", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "f", "no", "n", "off"):
            return False
    return bool(value)


def _find_hilp_matches(td_models_dir: str, tokens: List[str]) -> List[str]:
    tokens = [str(token) for token in tokens if token]
    if not tokens:
        return []
    pkl_paths = sorted(
        os.path.join(td_models_dir, name)
        for name in os.listdir(td_models_dir)
        if name.endswith(".pkl")
    )
    matches = []
    for path in pkl_paths:
        filename = os.path.basename(path)
        if any(token in filename for token in tokens):
            matches.append(path)
    return matches


def _find_dql_result_dirs(results_root: str, dataset_name: str, ckpt_id: int = 200) -> List[str]:
    if not dataset_name or not os.path.isdir(results_root):
        return []

    prefix = f"{dataset_name}|"
    required_files = (
        f"actor_{ckpt_id}.pth",
        f"critic_{ckpt_id}.pth",
        f"critic_target_{ckpt_id}.pth",
    )
    matches: List[Tuple[float, str]] = []

    for entry in os.listdir(results_root):
        full_path = os.path.join(results_root, entry)
        if not os.path.isdir(full_path) or not entry.startswith(prefix):
            continue
        if not all(os.path.isfile(os.path.join(full_path, fname)) for fname in required_files):
            continue
        actor_path = os.path.join(full_path, required_files[0])
        try:
            mtime = os.path.getmtime(actor_path)
        except OSError:
            mtime = 0.0
        matches.append((mtime, full_path))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in matches]


def _resolve_dql_result_dir(results_root: str, dataset_name: str, ckpt_id: int = 200) -> Tuple[str, str]:
    exact_matches = _find_dql_result_dirs(results_root, dataset_name, ckpt_id=ckpt_id)
    if exact_matches:
        return exact_matches[0], "exact"

    fallback_datasets: List[str] = []
    if dataset_name.endswith("-stitch-v0"):
        fallback_datasets.append(dataset_name.replace("-stitch-v0", "-navigate-v0"))

    for fallback_dataset in fallback_datasets:
        fallback_matches = _find_dql_result_dirs(results_root, fallback_dataset, ckpt_id=ckpt_id)
        if fallback_matches:
            return fallback_matches[0], f"fallback:{fallback_dataset}"

    available_dirs = []
    if os.path.isdir(results_root):
        available_dirs = sorted(
            name for name in os.listdir(results_root)
            if os.path.isdir(os.path.join(results_root, name))
        )
    raise FileNotFoundError(
        f"No DQL checkpoint directory found for dataset '{dataset_name}' under {results_root}. "
        f"Need actor_{ckpt_id}.pth/critic_{ckpt_id}.pth/critic_target_{ckpt_id}.pth. "
        f"Available dirs: {available_dirs}"
    )


def _load_dataset_name_from_config(dataset_config_name: str) -> Optional[str]:
    dataset_cfg_path = os.path.join(
        _repo_root(),
        "configurations",
        "dataset",
        f"{dataset_config_name}.yaml",
    )
    if not os.path.isfile(dataset_cfg_path):
        return None
    try:
        dataset_cfg = OmegaConf.load(dataset_cfg_path)
    except Exception:
        return None
    dataset_name = dataset_cfg.get("dataset", None)
    return str(dataset_name) if dataset_name is not None else None


def _detect_hilp_checkpoint_path(cfg: DictConfig) -> str:
    td_models_dir = os.path.join(_repo_root(), "td_models")
    assert os.path.isdir(td_models_dir), f"HILP auto-detection failed: td_models directory not found: {td_models_dir}"

    pkl_files = sorted(
        os.path.join(td_models_dir, name)
        for name in os.listdir(td_models_dir)
        if name.endswith(".pkl")
    )
    assert pkl_files, f"HILP auto-detection failed: no .pkl files found in {td_models_dir}"

    dataset_config_name = cfg.get("train_dataset_config", None)
    if dataset_config_name is None:
        dataset_config_name = cfg.get("dataset_config", None)
    dataset_config_name = str(dataset_config_name) if dataset_config_name is not None else None

    primary_groups: List[Tuple[str, List[str]]] = []
    if dataset_config_name:
        dataset_name = _load_dataset_name_from_config(dataset_config_name)
        if dataset_name:
            primary_groups.append(("dataset yaml `dataset`", [dataset_name]))
        primary_groups.append((
            "dataset config name",
            [dataset_config_name, dataset_config_name.replace("_", "-")],
        ))

    fallback_groups: List[Tuple[str, List[str]]] = []
    dataset_name_from_cfg = cfg.get("dataset", None)
    if dataset_name_from_cfg is not None:
        dataset_name_from_cfg = str(dataset_name_from_cfg)
        fallback_groups.append(("algorithm.dataset", [dataset_name_from_cfg]))

    search_groups = primary_groups + fallback_groups
    assert search_groups, (
        "HILP auto-detection failed: neither algorithm.train_dataset_config nor "
        "algorithm.dataset_config nor algorithm.dataset is available."
    )

    for label, tokens in search_groups:
        matches = sorted(set(_find_hilp_matches(td_models_dir, tokens)))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            match_names = [os.path.basename(path) for path in matches]
            raise AssertionError(
                f"HILP auto-detection ambiguous via {label}: tokens={tokens} matched {match_names}"
            )

    available = [os.path.basename(path) for path in pkl_files]
    raise AssertionError(
        "HILP auto-detection failed: no td_models/*.pkl matched "
        f"dataset_config={dataset_config_name!r}, dataset={cfg.get('dataset', None)!r}. "
        f"Available files: {available}"
    )



@dataclass
class MCTSTreeState:
    """Container holding all state for a single MCTS tree instance."""

    # --- Static config (set at init, never mutated) ---
    root_node: TreeNode
    terminal_depth: int
    noise_level: Optional[
        np.ndarray
    ]  # always None; tree-specific schedules are generated dynamically per expansion round
    children_node_guidance_scales: list
    skip_level_steps: int
    tag: str
    is_tree1: bool = True  # Prefix-orientation metadata; anchor trees keep forward-prefix semantics.
    # Root observation (unnormalized) for this tree's anchor.
    # Used to track agent positions across expansion rounds.
    tree_root_obs: Optional[np.ndarray] = None  # shape (obs_dim,)
    anchor_idx: int = 0
    anchor_name: str = "S"
    anchor_xy: Optional[np.ndarray] = None
    # --- Mutable search state (updated by _run_mcts_search) ---
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


class DiffusionForcingPlanning(
    KDEEstimatorMixin,
    SampledGraphEstimatorMixin,
    NoiseScheduleMixin,
    PlanVizMixin,
    PlanExecutorMixin,
    PlanPostprocMixin,
    DiffusionForcingBase,
):
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
        # Manually initialize frame_stack as requested to solve dependency order
        self.frame_stack = cfg.frame_stack
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
        self.ogbench_enable_reset_perturb = _coerce_bool(
            cfg.get("ogbench_enable_reset_perturb", True),
            default=True,
        )
        self.task_id = cfg.get("task_id", None)
        self.task_override_path = cfg.get("task_override_path", None)
        self.task_override_waypoint_group_idx = cfg.get("task_override_waypoint_group_idx", None)
        self.task_override_resolved_path = resolve_task_override_path(
            self.task_override_path,
            repo_root=_repo_root(),
        ) if self.task_override_path not in (None, "") else None
        self.sampled_graph_cache_path_override = None
        if self.task_override_resolved_path is not None:
            _, _task_override_payload = load_task_override_payload(
                self.task_override_resolved_path,
                repo_root=_repo_root(),
            )
            _cache_override = _task_override_payload.get("sampled_graph_cache_path")
            if _cache_override not in (None, ""):
                _expanded_cache_path = os.path.expanduser(str(_cache_override))
                if not os.path.isabs(_expanded_cache_path):
                    _expanded_cache_path = os.path.join(_repo_root(), _expanded_cache_path)
                _resolved_cache_path = os.path.realpath(_expanded_cache_path)
                if not os.path.exists(_resolved_cache_path):
                    _cache_basename = os.path.basename(_resolved_cache_path)
                    _sampled_graph_dir = os.path.realpath(
                        os.path.expanduser(
                            cfg.get("sampled_graph_save_dir", cfg.get("kde_save_dir", "~/.ogbench/data"))
                        )
                    )
                    _remapped_cache_path = os.path.join(_sampled_graph_dir, _cache_basename)
                    if os.path.exists(_remapped_cache_path):
                        print(
                            "[INIT] Remapped sampled graph cache override "
                            f"from {_resolved_cache_path} to {_remapped_cache_path}",
                            flush=True,
                        )
                        _resolved_cache_path = os.path.realpath(_remapped_cache_path)
                self.sampled_graph_cache_path_override = _resolved_cache_path
        self._current_task_start_xy = None
        self._current_task_goal_xy = None
        self._current_task_waypoints = None
        self._current_task_names = None
        # task_ids: list of task IDs to evaluate sequentially in one process.
        # Set by run_jobs.py when multiple same-config jobs are batched together.
        raw_ids = cfg.get("task_ids", None)
        self.task_ids = list(raw_ids) if raw_ids is not None else None  # Optional[List[int]]
        self.eval_repeat_id = cfg.get("eval_repeat_id", None)
        self.benchmark_num_rollouts = int(cfg.get("benchmark_num_rollouts", 1))
        self.benchmark_rollout_seed_base = int(cfg.get("benchmark_rollout_seed_base", 0))
        self.benchmark_results_path = cfg.get("benchmark_results_path", None)
        self.benchmark_model_id = cfg.get("benchmark_model_id", None)
        _benchmark_top_n = cfg.get("benchmark_waypoint_top_n", None)
        self.benchmark_waypoint_top_n = (
            int(_benchmark_top_n) if _benchmark_top_n is not None else None
        )
        self.benchmark_variant_name = str(
            cfg.get("benchmark_variant_name", "online_hemiltonian")
        )
        self.benchmark_plan_only = _coerce_bool(
            cfg.get("benchmark_plan_only", False),
            default=False,
        )
        self.benchmark_log_groundtruth_media = _coerce_bool(
            cfg.get("benchmark_log_groundtruth_media", True),
            default=True,
        )
        self.benchmark_postprocessed_plan_media_key = str(
            cfg.get("benchmark_postprocessed_plan_media_key", "postprocessed_plan")
        )
        self._benchmark_override_payload_cache: Optional[dict] = None
        self._current_wandb_visual_prefix: Optional[str] = None
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

        # [Eval / Planning] ─────────────────────────────────────────────────────
        # segment_episode_len: raw episode length per segment (pre-jump).
        # effective episode = segment_episode_len * sequence_dividing_factor.
        # Only used in eval (planning). Training code must NOT depend on this.
        self.sequence_dividing_factor = cfg.get("sequence_dividing_factor", None)
        self.segment_episode_len = cfg.get("segment_episode_len", None)
        if self.segment_episode_len is None:
            # Fallback: use episode_len from ckpt training_hparams (restored by _apply_ckpt_hparams_to_cfg).
            # effective planning horizon = episode_len * sequence_dividing_factor.
            _ckpt_ep = cfg.get("episode_len", None)
            if _ckpt_ep is not None:
                self.segment_episode_len = int(_ckpt_ep)
        if self.segment_episode_len is not None and self.sequence_dividing_factor is not None:
            _effective_ep = self.segment_episode_len * self.sequence_dividing_factor
            assert _effective_ep % (self.jump * self.frame_stack) == 0, (
                f"segment_episode_len*sequence_dividing_factor={_effective_ep} must be divisible by "
                f"jump*frame_stack={self.jump}*{self.frame_stack}={self.jump * self.frame_stack}"
            )
            self.n_tokens = episode_len_to_plan_tokens(_effective_ep, self.jump, self.frame_stack) + 1
        else:
            self.n_tokens = None  # training: n_tokens determined from batch shape

        # [Train / Visualization] ───────────────────────────────────────────────
        # episode_len: raw sliding-window length from dataset (pre-jump), set in
        # train_df_planning.yaml as ${dataset.episode_len}. Only used for training
        # visualization. eval_episode_len must NOT be used here.
        _train_ep = cfg.get("episode_len", None)
        if _train_ep is not None:
            _valid_ep = int(self.valid_episode_len_multiple * _train_ep)
            self._valid_n_tokens = episode_len_to_plan_tokens(_valid_ep, self.jump, self.frame_stack) + 1
        else:
            self._valid_n_tokens = None  # eval mode: no training visualization

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
        self.thres_min_dist_to_prior_plan = float(
            cfg.get("thres_min_dist_to_prior_plan", 0.0)
        )
        self.max_child_resets = cfg.get("max_child_resets", 3)
        self.particle_guidance_scale = cfg.get("particle_guidance_scale", 0.0)
        self.use_TD_metric_as_dist = cfg.get("use_TD_metric_as_dist", False)
        self.debug_memory_profile = cfg.get("debug_memory_profile", False)
        self.profiler_snapshot_frames = cfg.get("profiler_snapshot_frames", cfg.get("max_plan_hist_keep", 20))  # Number of denoising frames kept for video
        self.noise_level_building_way = cfg.get("noise_level_building_way", "pyramid")

        # HILP value function guidance
        self.hilp_checkpoint_path = None  # auto-detected lazily on first HILP use
        self.hilp_obs_dim = cfg.get("hilp_obs_dim", 29)    # used only for legacy .pt checkpoints
        self.hilp_skill_dim = cfg.get("hilp_skill_dim", 256)  # used only for legacy .pt checkpoints
        # HILP value function instance will be loaded lazily and stored in _hilp_value_fn_instance
        # We don't initialize it here to prevent PyTorch from registering it as a submodule
        self.anchor_guidance_scale_ratio = cfg.get("anchor_guidance_scale_ratio", 1.0)
        self.rdf_guidance_scale = cfg.get("rdf_guidance_scale", 2.0)
        self.rdf_sigma = cfg.get("rdf_sigma", 1.0)
        self.rmse_guidance_scale = cfg.get("rmse_guidance_scale", 1.0)
        self.use_score_func_with_TD = cfg.get("use_score_func_with_TD", True)
        self.TD_thres_for_far_target = cfg.get("TD_thres_for_far_target", None)
        self.kde_sigma = cfg.get("kde_sigma", 0.3)
        self.kde_grad_thres_sigma_coeff = cfg.get("kde_grad_thres_sigma_coeff", 0.3)
        self.kde_lam = cfg.get("kde_lam", 2.0)
        self.regularize_goal_guidance = cfg.get("regularize_goal_guidance", False)
        self.kde_sample_ratio = cfg.get("kde_sample_ratio", 0.1)
        self._kde_save_dir = os.path.expanduser(cfg.get("kde_save_dir", "~/.ogbench/data"))
        self.sampled_graph_sample_ratio = float(cfg.get("sampled_graph_sample_ratio", 0.005))
        self.sampled_graph_edge_radius = float(cfg.get("sampled_graph_edge_radius", 2.0))
        self.sampled_graph_seed = int(cfg.get("sampled_graph_seed", 42))
        self._sampled_graph_cache_dir = os.path.expanduser(
            cfg.get("sampled_graph_save_dir", cfg.get("kde_save_dir", "~/.ogbench/data"))
        )
        self._sampled_graph_data_dir = self._kde_save_dir
        self.mcts_use_replan = cfg.get("mcts_use_replan", False)
        self.viz_replanning: bool = cfg.get("viz_replanning", True)
        self.use_uncertainty_as_value: bool = cfg.get("use_uncertainty_as_value", False)
        self.uncertainty_mode: str = cfg.get("uncertainty_mode", "entropy")
        self.use_anchor_planner: bool = bool(cfg.get("use_anchor_planner", True))
        self.multi_tree_route_mode: str = str(cfg.get("multi_tree_route_mode", "online"))
        self.reliable_TD_threshold: float = float(cfg.get("reliable_TD_threshold", 30.0))
        self.temporal_dist_overestimate_coeff: float = float(
            cfg.get("temporal_dist_overestimate_coeff", 0.001)
        )
        self.uncertainty_lambda: float = float(cfg.get("uncertainty_lambda", 1.0))
        self.uncertainty_eta: float = float(cfg.get("uncertainty_eta", 1.0))
        self.uncertainty_max_intra_cluster_dist: float = float(cfg.get("uncertainty_max_intra_cluster_dist", 20.0))
        self.use_cluster_subplan_as_expansion: bool = cfg.get("use_cluster_subplan_as_expansion", False)
        self.use_kde_maximin_for_selecting_subplan_in_cluster: bool = cfg.get(
            "use_kde_maximin_for_selecting_subplan_in_cluster", False
        )
        self.viz_uncertain_next_subplan_last_obs: bool = cfg.get("viz_uncertain_next_subplan_last_obs", False)
        self.fast_sampling_multiple: int = cfg.get("fast_sampling_multiple", 5)
        self.fast_sampling_steps: int = cfg.get("fast_sampling_steps", 10)
        self.global_selection_count: int = 0
        self.global_search_num: int = 0
        self.current_plan_tokens: Optional[int] = None
        self.use_rollout: bool = cfg.get("use_rollout", False)
        self.use_dynamic_obs_padding: bool = cfg.get("use_dynamic_obs_padding", True)
        self.use_segment_wise_sliding_window: bool = cfg.get("use_segment_wise_sliding_window", False)

        super().__init__(cfg)
        self.plot_end_points = cfg.get("plot_start_goal", False)
        
        self.frame_sampling_way: str = cfg.get("frame_sampling_way", "linear")
        self.validation_video_max_frames: int = int(cfg.get("validation_video_max_frames", 200))
        self.validation_video_path_stride: int = int(cfg.get("validation_video_path_stride", 4))
        self.validation_video_fps: int = int(cfg.get("validation_video_fps", 8))
        self.viz_subplan_denoising: bool = bool(cfg.get("viz_subplan_denoising", False))
        self.viz_agent_rollout: bool = bool(cfg.get("viz_agent_rollout", False))
        self.use_directly_inject_guidance_to_x0: bool = bool(cfg.get("use_directly_inject_guidance_to_x0", False))
        self.direct_x0_guidance_scale: float = float(cfg.get("direct_x0_guidance_scale", 0.2))
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
                "segment_episode_len": self.segment_episode_len,
                "job_name": _ti_job_name,
            },
            debug_mode=self.cfg.get("DEBUG", True),
        )
        self.tracer.__enter__()  # Open file now (before interact() is called)
        # Only set as default if no tracer is already active (e.g. validation_anal tracer
        # set by exp_base.py before _build_algo() is called).
        if get_tracer() is None:
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
            if self.hilp_checkpoint_path is None:
                self.hilp_checkpoint_path = _detect_hilp_checkpoint_path(self.cfg)
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
        temporal_dist = self.emb_dist_to_temporal_dist((-v).cpu().numpy(), gamma=0.995)
        td_to_real_metric_ratio = 0.2
        return temporal_dist * td_to_real_metric_ratio 

    def _compute_raw_state_temporal_dist_np(
        self,
        obs_a_np: np.ndarray,  # (N, D) float32 — unnormalized observations
        obs_b_np: np.ndarray,  # (N, D) float32 — unnormalized observations (same shape)
    ) -> np.ndarray:           # (N,) float32 — raw temporal distance
        """Compute raw HILP-based temporal distance without extra metric scaling.

        This must stay on the same scale as T_curr returned by
        _compute_node_uncertainty so cumulative root-distance bookkeeping and
        expected_root_node_dist values remain internally consistent.
        """
        v = self._compute_hilp_values(obs_a_np, obs_b_np)
        temporal_dist = self.emb_dist_to_temporal_dist((-v).cpu().numpy(), gamma=0.995)
        return np.asarray(temporal_dist, dtype=np.float32)

    def _transform_temporal_dist_for_total_cost(self, temporal_dist: float) -> float:
        temporal_dist = float(temporal_dist)
        return temporal_dist + self.temporal_dist_overestimate_coeff * (temporal_dist ** 2)

    def _compute_distance(
        self,
        state1: np.ndarray,  # (N, d) — full obs (d==obs_dim) or position-only (d==pos_dim)
        state2: np.ndarray,  # (N, d) — same
    ) -> np.ndarray:          # (N,) — distance array
        """Unified distance computation based on use_TD_metric_as_dist flag.

        Accepts states with full obs dimension (len(obs_dim_indices)) or any
        partial dimension. When use_TD_metric_as_dist=True, partial inputs are
        zero-padded at missing obs_dim_indices positions before the TD metric
        is applied. When False, pos_dim_indices are extracted for Euclidean
        distance (or the input is used directly if already at pos_dim size).
        """
        obs_len = len(self.obs_dim_indices)
        pos_len = len(self.pos_dim_indices)
        d = state1.shape[-1]

        if self.use_TD_metric_as_dist:
            if d < obs_len:
                # Partial input — zero-pad to full obs dim at pos_dim_indices positions
                shape = state1.shape[:-1] + (obs_len,)
                full1 = np.zeros(shape, dtype=state1.dtype)
                full2 = np.zeros(shape, dtype=state2.dtype)
                full1[..., self.pos_dim_indices] = state1
                full2[..., self.pos_dim_indices] = state2
                state1, state2 = full1, full2
            return self._compute_state_temporal_dist_np(state1, state2)
        else:
            if d > pos_len:
                state1 = state1[..., self.pos_dim_indices]
                state2 = state2[..., self.pos_dim_indices]
            return np.linalg.norm(state1 - state2, axis=-1)

    def _get_root_obs(self, node: Optional["TreeNode"]) -> Optional[np.ndarray]:
        """Return the root observation for a node's tree."""
        if node is None:
            return None
        root_node = node
        while root_node._parent_node is not None:
            root_node = root_node._parent_node
        return root_node.obs

    def _get_root_node(self, node: Optional["TreeNode"]) -> Optional["TreeNode"]:
        if node is None:
            return None
        root_node = node
        while root_node._parent_node is not None:
            root_node = root_node._parent_node
        return root_node

    def _is_multi_tree_anchor_tag(self, tag: Optional[str]) -> bool:
        return isinstance(tag, str) and tag.startswith("multi_tree_")

    def _is_multi_tree_anchor_tree(self, tree: Optional["MCTSTreeState"]) -> bool:
        if tree is None:
            return False
        return self._is_multi_tree_anchor_tag(getattr(tree, "tag", None))

    def _tree_uses_forward_prefix_semantics(
        self,
        tree: Optional["MCTSTreeState"],
    ) -> bool:
        if tree is None:
            return True
        if self._is_multi_tree_anchor_tree(tree):
            return True
        return bool(getattr(tree, "is_tree1", True))

    def _node_anchor_name(
        self,
        node: Optional["TreeNode"],
        default: str = "?",
    ) -> str:
        root_node = self._get_root_node(node)
        return str(
            getattr(root_node, "anchor_name", None)
            or getattr(node, "anchor_name", None)
            or default
        )

    def _normalize_obs_np(self, obs_np: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            (np.asarray(obs_np, dtype=np.float32) - np.asarray(self.observation_mean, dtype=np.float32))
            / np.asarray(self.observation_std, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

    def _anchor_pair_key(self, a: int, b: int) -> Tuple[int, int]:
        return (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))

    def _required_anchor_degree(self, anchor_idx: int, n_anchors: int) -> int:
        if anchor_idx in (0, n_anchors - 1):
            return 1
        return 2

    def _is_anchor_satisfied(self, planner_state: dict, anchor_idx: int) -> bool:
        accepted_neighbors = planner_state["accepted_neighbors"][int(anchor_idx)]
        return len(accepted_neighbors) >= self._required_anchor_degree(
            int(anchor_idx),
            len(planner_state["trees"]),
        )

    def _get_forced_adjacent_pairs(self, planner_state: dict) -> list[tuple[int, int]]:
        return sorted(planner_state["accepted_pair_edges"].keys())

    def _compute_tentative_hamiltonian_solution(self, planner_state: dict) -> dict:
        if self.multi_tree_route_mode == "fixed_temporal":
            solver_result = planner_state.get("fixed_tentative_solver_result")
            if solver_result is None:
                raise ValueError("fixed_temporal route mode requires fixed_tentative_solver_result")
            planner_state["tentative_solver_result"] = dict(solver_result)
            return planner_state["tentative_solver_result"]

        effective_pair_cost = np.asarray(planner_state["effective_pair_cost"], dtype=np.float32)
        solver_result = solve_fixed_endpoint_hamiltonian_path_with_forced_adjacency(
            effective_pair_cost,
            forced_adjacent_pairs=self._get_forced_adjacent_pairs(planner_state),
        )
        planner_state["tentative_solver_result"] = solver_result
        return solver_result

    def _load_benchmark_override_payload(self) -> dict:
        if self._benchmark_override_payload_cache is None:
            _, payload = load_task_override_payload(
                self.task_override_resolved_path,
                repo_root=_repo_root(),
            )
            self._benchmark_override_payload_cache = payload or {}
        return self._benchmark_override_payload_cache

    def _get_benchmark_waypoint_groups_for_task(self, task_id: int) -> list[tuple[int, list]]:
        payload = self._load_benchmark_override_payload()
        tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
        if not isinstance(tasks, dict):
            raise ValueError("Task override payload must contain a mapping at 'tasks'")
        task_override = tasks.get(int(task_id))
        if task_override is None:
            task_override = tasks.get(str(task_id))
        if task_override is None:
            raise ValueError(
                f"Task override payload does not define task {task_id}: {self.task_override_resolved_path}"
            )
        if not isinstance(task_override, dict):
            raise ValueError(f"Task override for task {task_id} must be a mapping")
        waypoint_groups = task_override.get("waypoint_ij_groups", [])
        if not isinstance(waypoint_groups, list):
            raise ValueError(f"task {task_id} waypoint_ij_groups must be a list")
        return [(idx, group) for idx, group in enumerate(waypoint_groups)]

    @staticmethod
    def _build_groundtruth_anchor_queries(
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        waypoints_xy: np.ndarray,
    ) -> dict[str, Any]:
        waypoints_xy = np.asarray(waypoints_xy, dtype=np.float32)
        if waypoints_xy.size == 0:
            waypoints_xy = np.zeros((0, 2), dtype=np.float32)
        else:
            waypoints_xy = waypoints_xy.reshape(-1, 2)

        anchor_xys = np.concatenate(
            [
                np.asarray(start_xy, dtype=np.float32).reshape(1, 2),
                waypoints_xy,
                np.asarray(goal_xy, dtype=np.float32).reshape(1, 2),
            ],
            axis=0,
        )
        n_waypoints = int(len(waypoints_xy))
        anchor_labels = ["start"] + [f"waypoint_{idx}" for idx in range(1, n_waypoints + 1)] + ["goal"]
        priority_order = np.asarray([0, n_waypoints + 1] + list(range(1, n_waypoints + 1)), dtype=np.int32)
        return {
            "anchor_xys": anchor_xys,
            "anchor_labels": anchor_labels,
            "priority_order": priority_order,
        }

    def _expand_groundtruth_route_segments(
        self,
        graph_cache: dict,
        anchor_node_indices: np.ndarray,
        anchor_order: np.ndarray,
    ) -> dict[str, Any]:
        anchor_order = np.asarray(anchor_order, dtype=np.int32).reshape(-1)
        if len(anchor_order) < 2:
            return {
                "reachable": False,
                "segments": [],
                "total_distance": np.inf,
            }

        segments = []
        total_distance = 0.0
        reachable = True
        for segment_idx, (src_anchor_idx, dst_anchor_idx) in enumerate(
            zip(anchor_order[:-1].tolist(), anchor_order[1:].tolist()),
            start=1,
        ):
            segment = query_shortest_path_between_node_indices(
                graph_cache,
                src_node_index=int(anchor_node_indices[src_anchor_idx]),
                dst_node_index=int(anchor_node_indices[dst_anchor_idx]),
            )
            segment["segment_index"] = segment_idx
            segment["src_anchor_index"] = int(src_anchor_idx)
            segment["dst_anchor_index"] = int(dst_anchor_idx)
            segments.append(segment)
            if not bool(segment["reachable"]):
                reachable = False
            else:
                total_distance += float(segment["shortest_distance"])

        if not reachable:
            total_distance = np.inf

        return {
            "reachable": reachable,
            "segments": segments,
            "total_distance": float(total_distance),
        }

    @staticmethod
    def _compute_plan_polyline_length(plan_xy: np.ndarray) -> float:
        pts = np.asarray(plan_xy, dtype=np.float32)
        if pts.size == 0:
            return 0.0
        if pts.ndim == 3:
            pts = pts[:, 0, :]
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        return float(np.linalg.norm(diffs, axis=-1).sum(dtype=np.float64))

    def _benchmark_metric_prefix(self) -> str:
        return f"benchmark/{self.benchmark_variant_name}"

    def _benchmark_media_key(self, leaf_key: str) -> str:
        if self._current_wandb_visual_prefix is not None:
            return f"{self._current_wandb_visual_prefix}/{leaf_key}"
        return leaf_key

    def _format_postprocessed_plan_title(
        self,
        route_text: str,
        total_length: Optional[float],
    ) -> str:
        route_text = str(route_text or "unknown")
        if total_length is None or not np.isfinite(float(total_length)):
            total_text = "unknown"
        else:
            total_text = f"{float(total_length):.3f}"
        return f"route={route_text}\ntotal={total_text}"

    def _compute_fixed_temporal_route_solution(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        waypoints_xy: np.ndarray,
    ) -> dict[str, Any]:
        anchor_bundle = self._build_groundtruth_anchor_queries(
            start_xy=start_xy,
            goal_xy=goal_xy,
            waypoints_xy=waypoints_xy,
        )
        temporal_anchor_dists = compute_pairwise_temporal_distance_matrix(
            self,
            src_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
            dst_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
            gamma=GROUNDTRUTH_TEMPORAL_VIZ_GAMMA,
        )
        temporal_route_solution = solve_fixed_endpoint_hamiltonian_path(temporal_anchor_dists)
        if bool(temporal_route_solution.get("feasible", False)):
            temporal_route_segments = self._build_groundtruth_straight_anchor_route_segments(
                anchor_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
                anchor_order=np.asarray(temporal_route_solution["anchor_order"], dtype=np.int32),
            )
        else:
            temporal_route_segments = {
                "reachable": False,
                "segments": [],
                "total_distance": np.inf,
            }
        return {
            "anchor_bundle": anchor_bundle,
            "temporal_route_solution": temporal_route_solution,
            "temporal_route_segments": temporal_route_segments,
        }

    @staticmethod
    def _extract_interact_task_geometry(
        interact_result: dict[str, Any],
        *,
        include_waypoints: bool,
    ) -> dict[str, Any]:
        start_xy = np.asarray(interact_result.get("start_xy", []), dtype=np.float32).reshape(-1, 2)
        goal_xy = np.asarray(interact_result.get("goal_xy", []), dtype=np.float32).reshape(-1, 2)
        if len(start_xy) == 0 or len(goal_xy) == 0:
            raise RuntimeError("benchmark requires interact() to expose start_xy and goal_xy")

        if include_waypoints:
            waypoints_xy = np.asarray(interact_result.get("waypoints_xy", []), dtype=np.float32)
            if waypoints_xy.size == 0:
                waypoints_xy = np.zeros((0, 2), dtype=np.float32)
            else:
                waypoints_xy = waypoints_xy.reshape(-1, 2)
        else:
            waypoints_xy = np.zeros((0, 2), dtype=np.float32)

        task_name = None
        if interact_result.get("task_names"):
            task_name = str(interact_result["task_names"][0])

        return {
            "start_xy": np.asarray(start_xy[0], dtype=np.float32).reshape(2),
            "goal_xy": np.asarray(goal_xy[0], dtype=np.float32).reshape(2),
            "waypoints_xy": waypoints_xy,
            "task_name": task_name,
        }

    def _compute_groundtruth_path_info(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        waypoints_xy: np.ndarray,
        task_name: Optional[str] = None,
    ) -> dict[str, Any]:
        t_total = time.time()
        print(
            "[BENCH-GT] Ground-truth info start "
            f"(task={task_name}, n_waypoints={len(waypoints_xy)})",
            flush=True,
        )
        t_stage = time.time()
        print("[BENCH-GT] Ground-truth stage: sampled_graph_cache", flush=True)
        graph_cache = self._get_sampled_graph_cache()
        print(
            "[BENCH-GT] Ground-truth stage done: sampled_graph_cache "
            f"(elapsed={time.time() - t_stage:.1f}s)",
            flush=True,
        )
        anchor_bundle = self._build_groundtruth_anchor_queries(
            start_xy=start_xy,
            goal_xy=goal_xy,
            waypoints_xy=waypoints_xy,
        )
        t_stage = time.time()
        print("[BENCH-GT] Ground-truth stage: temporal_anchor_matrix", flush=True)
        temporal_anchor_dists = compute_pairwise_temporal_distance_matrix(
            self,
            src_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
            dst_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
            gamma=GROUNDTRUTH_TEMPORAL_VIZ_GAMMA,
        )
        print(
            "[BENCH-GT] Ground-truth stage done: temporal_anchor_matrix "
            f"(elapsed={time.time() - t_stage:.1f}s)",
            flush=True,
        )
        temporal_route_solution = solve_fixed_endpoint_hamiltonian_path(temporal_anchor_dists)
        if bool(temporal_route_solution.get("feasible", False)):
            temporal_route_segments = self._build_groundtruth_straight_anchor_route_segments(
                anchor_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
                anchor_order=np.asarray(temporal_route_solution["anchor_order"], dtype=np.int32),
            )
        else:
            temporal_route_segments = {
                "reachable": False,
                "segments": [],
                "total_distance": np.inf,
            }
        t_stage = time.time()
        print("[BENCH-GT] Ground-truth stage: sampled_graph_anchor_assignment", flush=True)
        anchor_assignment = query_distinct_nearest_nodes(
            graph_cache,
            query_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
            priority_order=np.asarray(anchor_bundle["priority_order"], dtype=np.int32),
            query_labels=list(anchor_bundle["anchor_labels"]),
        )
        print(
            "[BENCH-GT] Ground-truth stage done: sampled_graph_anchor_assignment "
            f"(elapsed={time.time() - t_stage:.1f}s)",
            flush=True,
        )
        anchor_node_indices = np.asarray(anchor_assignment["node_indices"], dtype=np.int32)
        anchor_node_xys = np.asarray(anchor_assignment["node_xys"], dtype=np.float32)
        anchor_node_dists = np.asarray(anchor_assignment["node_euclidean_dists"], dtype=np.float32)

        t_stage = time.time()
        print("[BENCH-GT] Ground-truth stage: sampled_graph_route", flush=True)
        graph_anchor_dists = extract_shortest_path_submatrix(graph_cache, anchor_node_indices)
        graph_route_solution = solve_fixed_endpoint_hamiltonian_path(graph_anchor_dists)
        if bool(graph_route_solution.get("feasible", False)):
            graph_route_segments = self._expand_groundtruth_route_segments(
                graph_cache,
                anchor_node_indices=anchor_node_indices,
                anchor_order=np.asarray(graph_route_solution["anchor_order"], dtype=np.int32),
            )
        else:
            graph_route_segments = {
                "reachable": False,
                "segments": [],
                "total_distance": np.inf,
            }
        print(
            "[BENCH-GT] Ground-truth stage done: sampled_graph_route "
            f"(elapsed={time.time() - t_stage:.1f}s)",
            flush=True,
        )

        sampled_xy = np.asarray(graph_cache["points_xy"], dtype=np.float32)
        n_total = int(graph_cache["n_total"])
        goal_node_index = int(anchor_node_indices[-1])
        goal_shortest_dists = np.asarray(
            graph_cache["shortest_dists"][goal_node_index],
            dtype=np.float32,
        )
        result = {
            "task_name": task_name,
            "start_xy": np.asarray(start_xy, dtype=np.float32).reshape(2),
            "goal_xy": np.asarray(goal_xy, dtype=np.float32).reshape(2),
            "waypoints_xy": np.asarray(waypoints_xy, dtype=np.float32).reshape(-1, 2),
            "anchor_bundle": anchor_bundle,
            "temporal_route_solution": temporal_route_solution,
            "temporal_route_segments": temporal_route_segments,
            "anchor_assignment": anchor_assignment,
            "anchor_node_indices": anchor_node_indices,
            "anchor_node_xys": anchor_node_xys,
            "anchor_node_dists": anchor_node_dists,
            "route_solution": graph_route_solution,
            "route_segments": graph_route_segments,
            "sampled_xy": sampled_xy,
            "goal_shortest_dists": goal_shortest_dists,
            "goal_node_index": goal_node_index,
            "n_total": n_total,
            "n_reachable": int(np.isfinite(goal_shortest_dists).sum()),
        }
        print(
            "[BENCH-GT] Ground-truth info done "
            f"(task={task_name}, elapsed={time.time() - t_total:.1f}s)",
            flush=True,
        )
        return result

    def _compute_groundtruth_hamiltonian_info(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        waypoints_xy: np.ndarray,
        task_name: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._compute_groundtruth_path_info(
            start_xy=start_xy,
            goal_xy=goal_xy,
            waypoints_xy=waypoints_xy,
            task_name=task_name,
        )

    def _build_groundtruth_straight_anchor_route_segments(
        self,
        anchor_xys: np.ndarray,
        anchor_order: np.ndarray,
    ) -> dict[str, Any]:
        anchor_xys = np.asarray(anchor_xys, dtype=np.float32).reshape(-1, 2)
        anchor_order = np.asarray(anchor_order, dtype=np.int32).reshape(-1)
        if len(anchor_order) < 2:
            return {
                "reachable": False,
                "segments": [],
                "total_distance": np.inf,
            }

        segments = []
        total_distance = 0.0
        for segment_idx, (src_anchor_idx, dst_anchor_idx) in enumerate(
            zip(anchor_order[:-1].tolist(), anchor_order[1:].tolist()),
            start=1,
        ):
            path_xy = np.asarray(
                [anchor_xys[int(src_anchor_idx)], anchor_xys[int(dst_anchor_idx)]],
                dtype=np.float32,
            )
            step_distance = float(np.linalg.norm(path_xy[1] - path_xy[0], ord=2))
            segments.append(
                {
                    "segment_index": segment_idx,
                    "src_anchor_index": int(src_anchor_idx),
                    "dst_anchor_index": int(dst_anchor_idx),
                    "path_xy": path_xy,
                    "shortest_distance": step_distance,
                    "reachable": True,
                }
            )
            total_distance += step_distance

        return {
            "reachable": True,
            "segments": segments,
            "total_distance": float(total_distance),
        }

    def _render_groundtruth_temporal_distance_image(
        self,
        groundtruth_info: dict[str, Any],
    ) -> Optional[Image.Image]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scripts.temporal_dist_heatmap import (
                _compute_temporal_distance_heatmap,
                _plot_temporal_panel,
            )
        except Exception as exc:
            print(f"[WARNING] Could not import ground-truth temporal plotting stack: {exc}", flush=True)
            return None

        fig = None
        try:
            goal_xy = np.asarray(groundtruth_info["goal_xy"], dtype=np.float32).reshape(2)
            heatmap = _compute_temporal_distance_heatmap(
                self,
                goal_xy=goal_xy,
                grid_res=GROUNDTRUTH_TEMPORAL_VIZ_GRID_RES,
                gamma=GROUNDTRUTH_TEMPORAL_VIZ_GAMMA,
            )
            grad_field = self._compute_guidance_grad_fields(
                goal_xy.astype(np.float32),
                grid_step=GROUNDTRUTH_TEMPORAL_VIZ_GRAD_GRID_STEP,
            )
            fig, ax = plt.subplots(1, 1, figsize=(9, 8))
            _plot_temporal_panel(
                ax,
                fig,
                self.env_id,
                heatmap,
                grad_field,
                start_xy=np.asarray(groundtruth_info["start_xy"], dtype=np.float32).reshape(2),
                goal_xy=goal_xy,
                waypoints_xy=np.asarray(groundtruth_info["waypoints_xy"], dtype=np.float32).reshape(-1, 2),
                route_solution=dict(groundtruth_info["temporal_route_solution"]),
                route_segments=dict(groundtruth_info["temporal_route_segments"]),
                task_name=groundtruth_info.get("task_name"),
            )
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
            buf.seek(0)
            with Image.open(buf) as image:
                return image.convert("RGB")
        except Exception as exc:
            print(f"[WARNING] Failed to render ground-truth temporal plot: {exc}", flush=True)
            return None
        finally:
            if fig is not None:
                plt.close(fig)

    def _render_groundtruth_path_image(self, groundtruth_info: dict[str, Any]) -> Optional[Image.Image]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from scripts.temporal_dist_heatmap import _plot_sample_panel
        except Exception as exc:
            print(f"[WARNING] Could not import ground-truth path plotting stack: {exc}", flush=True)
            return None

        fig = None
        try:
            fig, ax = plt.subplots(1, 1, figsize=(9, 8))
            anchor_bundle = groundtruth_info["anchor_bundle"]
            _plot_sample_panel(
                ax,
                fig,
                self.env_id,
                np.asarray(groundtruth_info["sampled_xy"], dtype=np.float32),
                shortest_dists=np.asarray(groundtruth_info["goal_shortest_dists"], dtype=np.float32),
                anchor_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
                anchor_node_xys=np.asarray(groundtruth_info["anchor_node_xys"], dtype=np.float32),
                anchor_node_dists=np.asarray(groundtruth_info["anchor_node_dists"], dtype=np.float32),
                sample_ratio=float(self.sampled_graph_sample_ratio),
                edge_radius=float(self.sampled_graph_edge_radius),
                n_total=int(groundtruth_info["n_total"]),
                n_reachable=int(groundtruth_info["n_reachable"]),
                route_solution=dict(groundtruth_info["route_solution"]),
                route_segments=dict(groundtruth_info["route_segments"]),
                task_name=groundtruth_info.get("task_name"),
            )
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
            buf.seek(0)
            with Image.open(buf) as image:
                return image.convert("RGB")
        except Exception as exc:
            print(f"[WARNING] Failed to render ground-truth path plot: {exc}", flush=True)
            return None
        finally:
            if fig is not None:
                plt.close(fig)

    def _render_groundtruth_hamiltonian_image(self, groundtruth_info: dict[str, Any]) -> Optional[Image.Image]:
        return self._render_groundtruth_path_image(groundtruth_info)

    def _log_groundtruth_path_media(
        self,
        groundtruth_info: dict[str, Any],
        *,
        stage_prefix: str,
        media_leaf_key: str,
    ) -> None:
        if not self.benchmark_log_groundtruth_media:
            return

        t_stage = time.time()
        print(f"{stage_prefix} stage=render_groundtruth_path start", flush=True)
        gt_image = self._render_groundtruth_path_image(groundtruth_info)
        print(
            f"{stage_prefix} stage=render_groundtruth_path done "
            f"(elapsed={time.time() - t_stage:.1f}s, image={'yes' if gt_image is not None else 'no'})",
            flush=True,
        )
        if gt_image is None:
            return

        t_log = time.time()
        print(f"{stage_prefix} stage=log_image_groundtruth_path start", flush=True)
        self.log_image(
            self._benchmark_media_key(media_leaf_key),
            gt_image,
        )
        print(
            f"{stage_prefix} stage=log_image_groundtruth_path done "
            f"(elapsed={time.time() - t_log:.1f}s)",
            flush=True,
        )

    def _augment_interact_result_with_groundtruth_path_metrics(
        self,
        interact_result: dict,
        groundtruth_info: dict[str, Any],
    ) -> dict:
        route_solution = groundtruth_info["route_solution"]
        route_segments = groundtruth_info["route_segments"]
        gt_reachable = bool(route_solution.get("feasible", False)) and bool(
            route_segments.get("reachable", False)
        )
        gt_path_length = float(route_segments.get("total_distance", np.inf))
        pred_plan_length = interact_result.get("postprocessed_plan_length", None)
        rel_error = None
        if pred_plan_length is not None and np.isfinite(gt_path_length) and gt_path_length > 0.0:
            rel_error = abs(float(pred_plan_length) - gt_path_length) / gt_path_length

        interact_result["groundtruth_route_text"] = str(route_solution.get("route_text", ""))
        interact_result["groundtruth_path_length"] = gt_path_length
        interact_result["groundtruth_min_distance"] = gt_path_length
        interact_result["groundtruth_snap_distances"] = [
            float(v) for v in np.asarray(groundtruth_info["anchor_node_dists"], dtype=np.float32).tolist()
        ]
        interact_result["groundtruth_reachable"] = gt_reachable
        interact_result["postprocessed_plan_length_rel_error"] = (
            float(rel_error) if rel_error is not None else None
        )
        return interact_result

    def _augment_interact_result_with_hamiltonian_groundtruth(
        self,
        interact_result: dict,
        groundtruth_info: dict[str, Any],
    ) -> dict:
        interact_result = self._augment_interact_result_with_groundtruth_path_metrics(
            interact_result,
            groundtruth_info,
        )
        route_solution = groundtruth_info["route_solution"]
        gt_anchor_order = np.asarray(route_solution.get("anchor_order", []), dtype=np.int32).tolist()
        pred_anchor_order = [
            int(idx) for idx in interact_result.get("predicted_anchor_order", [])
        ]
        gt_reachable = bool(interact_result.get("groundtruth_reachable", False))
        hamiltonian_success = bool(gt_reachable and pred_anchor_order == gt_anchor_order)

        interact_result["groundtruth_anchor_order"] = gt_anchor_order
        interact_result["hamiltonian_path_success"] = hamiltonian_success
        return interact_result

    def _get_tentative_neighbor_indices(self, planner_state: dict, anchor_idx: int) -> list[int]:
        solver_result = planner_state.get("tentative_solver_result", {})
        order = np.asarray(solver_result.get("anchor_order", []), dtype=np.int32)
        if order.size == 0:
            return []
        anchor_idx = int(anchor_idx)
        match = np.where(order == anchor_idx)[0]
        if match.size == 0:
            return []
        pos = int(match[0])
        neighbors: list[int] = []
        if pos - 1 >= 0:
            neighbors.append(int(order[pos - 1]))
        if pos + 1 < len(order):
            neighbors.append(int(order[pos + 1]))
        return neighbors

    def _select_multi_tree_target_node(
        self,
        planner_state: dict,
        source_tree: "MCTSTreeState",
        current_leaf_obs: Optional[np.ndarray] = None,
        child_node: Optional["TreeNode"] = None,
        use_segment_metric: bool = False,
    ) -> Optional[dict]:
        source_anchor_idx = int(source_tree.anchor_idx)
        accepted_neighbors = planner_state["accepted_neighbors"][source_anchor_idx]
        tentative_neighbor_indices = set(
            self._get_tentative_neighbor_indices(planner_state, source_anchor_idx)
        )

        if use_segment_metric:
            if child_node is None or child_node.obs is None:
                return None
            plan_tokens = self._require_current_plan_tokens()
            segment_obs = self._extract_new_segment_obs(
                child_node,
                plan_tokens,
            )
            if segment_obs is None or segment_obs.size == 0:
                return None
            if not tentative_neighbor_indices:
                return None
            query_obs = np.asarray(segment_obs, dtype=np.float32).reshape(
                -1,
                child_node.obs.shape[0],
            )
        else:
            if current_leaf_obs is None:
                return None
            query_obs = np.asarray(current_leaf_obs, dtype=np.float32).reshape(1, -1)

        candidate_rows: list[tuple[float, int, str, "MCTSTreeState", "TreeNode"]] = []
        for target_tree in planner_state["trees"]:
            target_anchor_idx = int(target_tree.anchor_idx)
            if target_anchor_idx == source_anchor_idx:
                continue
            if target_anchor_idx in accepted_neighbors:
                continue
            if self._is_anchor_satisfied(planner_state, target_anchor_idx):
                continue
            if use_segment_metric and target_anchor_idx not in tentative_neighbor_indices:
                continue

            if use_segment_metric:
                target_nodes = [
                    node
                    for node in target_tree.get_all_nodes()
                    if node.obs is not None
                    and len(node.plan_history) > 0
                    and len(node.plan_history[-1]) > 0
                ]
            else:
                target_nodes = [
                    node
                    for node in target_tree.get_all_nodes()
                    if node.obs is not None
                ]
            if not target_nodes:
                continue

            target_obs = np.stack([node.obs for node in target_nodes], axis=0).astype(np.float32)
            query_rep = np.repeat(query_obs, len(target_nodes), axis=0)
            tgt_rep = np.tile(target_obs, (query_obs.shape[0], 1))
            temporal_dists = self._compute_raw_state_temporal_dist_np(query_rep, tgt_rep).reshape(
                query_obs.shape[0],
                len(target_nodes),
            )

            if use_segment_metric:
                node_scores = np.min(temporal_dists, axis=0)
            else:
                node_scores = temporal_dists[0]

            for score, node in zip(node_scores.tolist(), target_nodes):
                candidate_rows.append(
                    (
                        float(score),
                        target_anchor_idx,
                        node.name,
                        target_tree,
                        node,
                    )
                )

        if not candidate_rows:
            return None

        candidate_rows.sort(key=lambda item: (item[0], item[1], item[2]))
        if use_segment_metric:
            best_score, _, _, best_tree, best_node = candidate_rows[0]
            return {
                "target_tree": best_tree,
                "target_node": best_node,
                "segment_td": float(best_score),
            }

        for score, _, _, target_tree, target_node in candidate_rows:
            target_anchor_idx = int(target_tree.anchor_idx)
            if score < self.reliable_TD_threshold:
                if target_anchor_idx in tentative_neighbor_indices:
                    return {
                        "target_tree": target_tree,
                        "target_node": target_node,
                        "score": float(score),
                    }
                continue
            return {
                "target_tree": target_tree,
                "target_node": target_node,
                "score": float(score),
            }

        fallback_score, _, _, fallback_tree, fallback_node = candidate_rows[0]
        return {
            "target_tree": fallback_tree,
            "target_node": fallback_node,
            "score": float(fallback_score),
        }

    def _get_multi_tree_for_node(
        self,
        planner_state: dict,
        node: Optional["TreeNode"],
    ) -> Optional["MCTSTreeState"]:
        if node is None:
            return None
        root_node = self._get_root_node(node)
        return next(
            (
                tree
                for tree in planner_state["trees"]
                if tree.root_node is root_node
            ),
            None,
        )

    def _update_multi_tree_meeting_targets_from_expansions(
        self,
        planner_state: dict,
        tree_batches: dict[str, dict],
    ) -> None:
        for tree_batch in tree_batches.values():
            source_tree: MCTSTreeState = tree_batch["tree"]
            for info in tree_batch["expanded_node_infos"].values():
                meeting_info = self._select_multi_tree_target_node(
                    planner_state=planner_state,
                    source_tree=source_tree,
                    child_node=info.get("node"),
                    use_segment_metric=True,
                )
                info["meeting_target_node"] = (
                    None if meeting_info is None else meeting_info["target_node"]
                )
                info["meeting_target_tree"] = (
                    None if meeting_info is None else meeting_info["target_tree"]
                )
                info["meeting_target_td"] = (
                    None if meeting_info is None else float(meeting_info["segment_td"])
                )

    def _check_plan_batch_feasibility(
        self,
        plan_hists: torch.Tensor,  # (m, plan_tokens*fs, B, c)
        root_obs_list: List[Optional[np.ndarray]],
        progress_obs_list: List[Optional[np.ndarray]],
        prefix_len_frames_list: Optional[List[int]] = None,
        subplan_tail_depths: Optional[List[int]] = None,
        seg_size: Optional[int] = None,
    ) -> List[bool]:
        """Run the planner's full-plan feasibility checks on a batch of plans.

        This intentionally checks the entire reconstructed plan, not just the local
        subplan. When `use_dynamic_obs_padding=False`, `parallel_plan()` strips the
        root anchor token from the returned history, so we prepend each candidate's
        root observation before running continuity checks.

        The progress filter remains a full-plan check: it measures progress between
        the current boundary observation and the final frame of the full plan.
        The prior-plan-distance filter is the new local-subplan constraint: it
        compares the current subplan tail against frames before `prefix_len` in the
        raw plan sequence (without the synthetic root prepend).
        """
        raw_plans = (
            self._unnormalize_x(plan_hists[-1])[:-1].detach().cpu().numpy()
        )  # (t*fs-1, B, c)
        plans = raw_plans

        if root_obs_list and all(root_obs is not None for root_obs in root_obs_list):
            root_obs_np = np.stack(root_obs_list, axis=0)[np.newaxis]  # (1, B, c)
            plans = np.concatenate([root_obs_np, plans], axis=0)

        plan_len, batch_size, channels = plans[:-1].shape
        diffs = self._compute_distance(
            plans[:-1].reshape(plan_len * batch_size, channels),
            plans[1:].reshape(plan_len * batch_size, channels),
        ).reshape(plan_len, batch_size)

        is_proximal = [
            bool(np.all(diffs[:, i] < self.plan_feasibility_delta))
            for i in range(batch_size)
        ]

        if self.min_progress_threshold > 0.0:
            is_not_stagnant = [False] * batch_size
            for i, progress_obs in enumerate(progress_obs_list):
                if progress_obs is None:
                    continue
                progress = float(
                    self._compute_distance(
                        progress_obs[np.newaxis],
                        plans[-1, i][np.newaxis],
                    )[0]
                )
                if progress > self.min_progress_threshold:
                    is_not_stagnant[i] = True
        else:
            is_not_stagnant = [True] * batch_size

        is_far_from_prior = [True] * batch_size
        if (
            self.thres_min_dist_to_prior_plan > 0.0
            and prefix_len_frames_list is not None
            and subplan_tail_depths is not None
            and seg_size is not None
        ):
            for i in range(batch_size):
                prefix_len_frames = max(0, int(prefix_len_frames_list[i]))
                if prefix_len_frames <= 0:
                    continue

                prior_end = min(prefix_len_frames, raw_plans.shape[0])
                if prior_end <= 0:
                    continue

                tail_depth = int(subplan_tail_depths[i])
                tail_idx = min(
                    self._get_prefix_len_frames_from_depth(tail_depth, seg_size) - 1,
                    raw_plans.shape[0] - 1,
                )
                if tail_idx < 0:
                    continue

                tail_obs = raw_plans[tail_idx, i : i + 1]
                prior_obs = raw_plans[:prior_end, i]
                if prior_obs.size == 0:
                    continue

                tail_rep = np.repeat(tail_obs, prior_obs.shape[0], axis=0)
                min_dist_to_prior = float(
                    np.min(self._compute_distance(prior_obs, tail_rep))
                )
                if min_dist_to_prior < self.thres_min_dist_to_prior_plan:
                    is_far_from_prior[i] = False

        return [
            is_proximal[i] and is_not_stagnant[i] and is_far_from_prior[i]
            for i in range(batch_size)
        ]

    def _compute_local_subplan_kde_maximin_scores(
        self,
        plan_batch: torch.Tensor,  # (plan_tokens*fs, B, c) — last denoising step
        parent_depth: int,
        seg_size: int,
    ) -> np.ndarray:
        """Score each plan by the minimum KDE log-density over its new local subplan.

        The KDE cache is built on dataset world ``(x, y)`` coordinates, so plans are
        first unnormalized back to world space and then sliced to the newly expanded
        local segment ``[depth, depth + 1)`` before interpolation. We use log-density
        instead of density directly because it is already cached and preserves the same
        maximin ordering.
        """
        if plan_batch.ndim != 3:
            raise ValueError(
                f"plan_batch must have shape (T, B, c), got {tuple(plan_batch.shape)}"
            )
        if len(self.pos_dim_indices) < 2:
            raise ValueError(
                "KDE maximin subplan selection requires at least 2 positional dims"
            )

        plan_unnorm = self._unnormalize_x(plan_batch.unsqueeze(0))[0][:-1]
        obs_raw, _, _ = self.split_bundle(plan_unnorm)
        obs_np = obs_raw.detach().cpu().numpy()
        if obs_np.shape[0] == 0:
            return np.full(plan_batch.shape[1], -1e12, dtype=np.float32)

        local_start = min(
            self._get_prefix_len_frames_from_depth(parent_depth, seg_size),
            obs_np.shape[0],
        )
        local_end = min(
            self._get_prefix_len_frames_from_depth(parent_depth + 1, seg_size),
            obs_np.shape[0],
        )

        pos_idx = self.pos_dim_indices[:2]
        if local_end <= local_start:
            fallback_idx = min(max(local_start - 1, 0), obs_np.shape[0] - 1)
            local_obs_xy = obs_np[fallback_idx : fallback_idx + 1, :, pos_idx]
        else:
            local_obs_xy = obs_np[local_start:local_end, :, pos_idx]

        query_xy = local_obs_xy.reshape(-1, 2)
        kde_vals = self._get_kde_log_density_grid(query_xy).reshape(
            local_obs_xy.shape[0], local_obs_xy.shape[1]
        )
        return np.min(kde_vals, axis=0).astype(np.float32)

    def _require_current_plan_tokens(self) -> int:
        """Return the planner-global plan token count for the current episode."""
        assert self.current_plan_tokens is not None, (
            "current_plan_tokens must be initialized before planning helpers run"
        )
        return self.current_plan_tokens

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

        # Visualization: generate a fresh plan of valid_n_tokens tokens via DDIM
        # (valid_n_tokens = episode_len * valid_episode_len_multiple // (jump * frame_stack) + 1)
        if self.global_step % 10000 == 0 and self._valid_n_tokens is not None:
            valid_n_tokens = self._valid_n_tokens
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
                    _x_t = self.diffusion_model.sample_step(_x_t, None, _from_nl, _to_nl, force_ddim=True)
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
        successes = []
        for tid in task_ids:
            if tid != self.task_id:
                self.task_id = tid
            task_ns = f"{namespace}/task{tid}" if self.task_ids is not None else namespace
            self.interact(batch_size, conditions, task_ns)
            if hasattr(self, '_last_interact_success') and self._last_interact_success is not None:
                successes.append(self._last_interact_success)

        if successes and self.task_ids is not None:
            self._safe_log_metric(f"{namespace}/total_success_count", float(sum(successes)))
            self._safe_log_metric(f"{namespace}/total_success_rate", float(sum(successes) / len(successes)))

    def _safe_log_metric(self, key: str, value: Any) -> None:
        if isinstance(value, np.generic):
            value = value.item()
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            self.log(key, value)
            return

        logger = self._resolve_logger()
        if logger is None or getattr(logger, "experiment", None) is None:
            return

        try:
            logger.experiment.log({key: value})
        except Exception:
            pass

    def _seed_interaction_envs(self, envs) -> None:
        if self.interaction_seed is None:
            return

        seed = int(self.interaction_seed)
        try:
            envs.seed(seed)
            return
        except Exception:
            pass

        seeded = False
        for idx, env in enumerate(getattr(envs, "envs", [])):
            env_seed = seed + idx
            try:
                env.seed(env_seed)
                seeded = True
                continue
            except Exception:
                pass
        if not seeded:
            print(f"[WARN] Failed to apply interaction_seed={seed} to envs", flush=True)

    def _seed_benchmark_rollout_rng(self, seed: int) -> None:
        np.random.seed(seed)
        py_random.seed(seed)

    def _verify_local_ogbench_runtime(self) -> None:
        if os.environ.get("MCTD_USE_LOCAL_OGBENCH") != "1":
            return

        import ogbench

        expected_root = os.path.realpath(os.environ.get("MCTD_LOCAL_OGBENCH_ROOT", ""))
        actual_file = os.path.realpath(getattr(ogbench, "__file__", ""))
        if not expected_root or not actual_file.startswith(expected_root):
            raise RuntimeError(
                f"Benchmark job expected local ogbench under {expected_root}, got {actual_file}"
            )
        print(f"[BENCHMARK] Using local ogbench source: {actual_file}", flush=True)

    def _get_ogbench_make_maze_env(self):
        """Resolve OGBench's maze factory across both installed and local source layouts."""
        import importlib
        import ogbench

        maze_module = getattr(getattr(ogbench, "locomaze", None), "maze", None)
        if maze_module is not None and hasattr(maze_module, "make_maze_env"):
            return maze_module.make_maze_env
        return importlib.import_module("ogbench.locomaze.maze").make_maze_env

    def _get_ogbench_maze_env_extra_kwargs(self) -> dict[str, Any]:
        extra_kwargs: dict[str, Any] = {}
        if not self.ogbench_enable_reset_perturb and "antmaze" in self.env_id:
            # Disable Ant base reset noise so benchmark reset positions stay deterministic.
            extra_kwargs["reset_noise_scale"] = 0.0
        return extra_kwargs

    @staticmethod
    def _get_single_env_sim_state(env) -> Optional[dict[str, np.ndarray]]:
        data = getattr(env, "data", None)
        if data is None:
            return None
        return {
            "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
        }

    @staticmethod
    def _set_single_env_sim_state(env, sim_state: Optional[dict[str, np.ndarray]]) -> None:
        if sim_state is None or not hasattr(env, "set_state"):
            return
        env.set_state(sim_state["qpos"], sim_state["qvel"])

    def _force_exact_ogbench_task_reset(
        self,
        env,
        ob: Any,
        info: Optional[dict[str, Any]],
        task_info: Optional[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        info_out = dict(info or {})
        if task_info is None:
            return np.asarray(ob, dtype=np.float32), info_out
        if not all(hasattr(env, attr) for attr in ("set_goal", "set_xy", "get_ob")):
            return np.asarray(ob, dtype=np.float32), info_out

        init_xy = task_info.get("init_xy")
        goal_xy = task_info.get("goal_xy")
        if init_xy is None or goal_xy is None:
            return np.asarray(ob, dtype=np.float32), info_out

        init_xy = np.asarray(init_xy, dtype=np.float32).reshape(-1)[:2]
        goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)[:2]
        saved_sim_state = self._get_single_env_sim_state(env)
        goal_ob = info_out.get("goal")

        try:
            env.set_goal(goal_xy=goal_xy)
            env.set_xy(goal_xy)
            goal_ob = np.asarray(env.get_ob(), dtype=np.float32).copy()
        except Exception:
            pass
        finally:
            self._set_single_env_sim_state(env, saved_sim_state)

        env.set_goal(goal_xy=goal_xy)
        env.set_xy(init_xy)
        ob_out = np.asarray(env.get_ob(), dtype=np.float32).copy()
        if goal_ob is not None:
            info_out["goal"] = np.asarray(goal_ob, dtype=np.float32).copy()
        info_out["exact_task_reset"] = True
        return ob_out, info_out

    def _print_ogbench_task_assignment_summary(self, env, task_id: int) -> None:
        cur_task_info = dict(getattr(env, "cur_task_info", None) or {})
        waypoint_group = cur_task_info.get(
            "waypoint_ij_group",
            cur_task_info.get("waypoint_ijs", []),
        )
        waypoint_count = len(waypoint_group) if waypoint_group is not None else 0
        print(
            "Task "
            f"{task_id} is set: "
            f"init_ij={cur_task_info.get('init_ij')} "
            f"goal_ij={cur_task_info.get('goal_ij')} "
            f"active_waypoint_group_idx={cur_task_info.get('active_waypoint_group_idx')} "
            f"waypoint_count={waypoint_count}",
            flush=True,
        )

    def _ensure_ogbench_set_task_compat(self, env) -> None:
        if getattr(env, "_mctd_set_task_compat", False):
            return
        if not hasattr(env, "task_infos") or not hasattr(env, "reset"):
            raise RuntimeError("Env does not expose task_infos/reset for OGBench task compatibility shim")

        original_reset = env.reset

        if hasattr(env, "set_task"):
            original_set_task = env.set_task

            def _set_task(task_id, *args, **kwargs):
                assert 1 <= task_id <= env.num_tasks, f"Task ID must be in [1, {env.num_tasks}]."
                suppress_buf = io.StringIO()
                with contextlib.redirect_stdout(suppress_buf), contextlib.redirect_stderr(suppress_buf):
                    result = original_set_task(task_id, *args, **kwargs)
                filtered_output = []
                for line in suppress_buf.getvalue().splitlines():
                    if line.startswith("Task ") and " is set:" in line:
                        continue
                    filtered_output.append(line)
                for line in filtered_output:
                    if line:
                        print(line, flush=True)
                if hasattr(env, "task_infos") and 1 <= int(task_id) <= len(env.task_infos):
                    env.cur_task_id = int(task_id)
                    env.cur_task_info = env.task_infos[int(task_id) - 1]
                self._print_ogbench_task_assignment_summary(env, int(task_id))
                return result

            env.set_task = _set_task
            env._mctd_set_task_compat = True
            return

        def _set_task(task_id):
            assert 1 <= task_id <= env.num_tasks, f"Task ID must be in [1, {env.num_tasks}]."
            env._mctd_pending_task_id = int(task_id)
            env.cur_task_id = int(task_id)
            env.cur_task_info = env.task_infos[int(task_id) - 1]
            self._print_ogbench_task_assignment_summary(env, int(task_id))

        def _reset(*args, **kwargs):
            options = kwargs.pop("options", None)
            if options is None:
                options = {}
            else:
                options = dict(options)
            pending_task_id = getattr(env, "_mctd_pending_task_id", None)
            if pending_task_id is not None and "task_id" not in options:
                options["task_id"] = pending_task_id
            return original_reset(*args, options=options, **kwargs)

        env._mctd_pending_task_id = None
        env.set_task = _set_task
        env.reset = _reset
        env._mctd_set_task_compat = True

    def _maybe_apply_ogbench_task_overrides(self, env) -> None:
        if self.task_override_resolved_path is None:
            return
        override_signature = (
            self.task_override_resolved_path,
            None if self.task_override_waypoint_group_idx is None else int(self.task_override_waypoint_group_idx),
        )
        already_applied = getattr(env, "_mctd_task_override_signature", None)
        if already_applied == override_signature:
            return
        applied_path = apply_task_overrides_to_env(
            env,
            self.task_override_resolved_path,
            expected_env_id=self.env_id,
            repo_root=_repo_root(),
            waypoint_group_idx_override=self.task_override_waypoint_group_idx,
        )
        env._mctd_task_override_signature = override_signature
        print(
            f"[OGBENCH] Applied task override: {applied_path} "
            f"(group_idx={override_signature[1]})",
            flush=True,
        )

    def _ensure_ogbench_task_metadata_compat(self, env) -> None:
        if self.task_override_resolved_path is None:
            return
        if getattr(env, "_mctd_task_metadata_compat", False):
            return

        original_reset = env.reset

        def _reset(*args, **kwargs):
            ob, info = original_reset(*args, **kwargs)
            if not self.ogbench_enable_reset_perturb:
                ob, info = self._force_exact_ogbench_task_reset(
                    env,
                    ob,
                    info,
                    getattr(env, "cur_task_info", None),
                )
            info = inject_task_metadata_into_reset_info(ob, info, getattr(env, "cur_task_info", None))
            return ob, info

        env.reset = _reset
        env._mctd_task_metadata_compat = True

    def _prepare_ogbench_env(self, env) -> None:
        self._ensure_ogbench_set_task_compat(env)
        self._maybe_apply_ogbench_task_overrides(env)
        self._ensure_ogbench_task_metadata_compat(env)

    def _capture_current_task_metadata(self, obs: torch.Tensor, goal: torch.Tensor, reset_infos: list[dict]) -> None:
        obs_np = obs.detach().cpu().numpy()[:, self.pos_dim_indices]
        goal_np = goal.detach().cpu().numpy()[:, self.pos_dim_indices]
        reset_infos = list(reset_infos or [])
        if len(reset_infos) < len(obs_np):
            reset_infos.extend({} for _ in range(len(obs_np) - len(reset_infos)))
        start_xy_list = []
        goal_xy_list = []
        waypoint_list = []
        task_name_list = []

        for idx, info in enumerate(reset_infos):
            info = info or {}
            start_xy = np.asarray(info.get("start_xy", obs_np[idx]), dtype=np.float32).reshape(-1)[:2]
            goal_xy = np.asarray(info.get("goal_xy", goal_np[idx]), dtype=np.float32).reshape(-1)[:2]
            waypoints_xy = np.asarray(info.get("waypoints_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            if waypoints_xy.size == 0:
                waypoints_xy = np.zeros((0, 2), dtype=np.float32)
            else:
                waypoints_xy = waypoints_xy.reshape(-1, 2)
            start_xy_list.append(start_xy)
            goal_xy_list.append(goal_xy)
            waypoint_list.append(waypoints_xy)
            task_name_list.append(info.get("task_name"))

        self._current_task_start_xy = np.stack(start_xy_list, axis=0) if start_xy_list else None
        self._current_task_goal_xy = np.stack(goal_xy_list, axis=0) if goal_xy_list else None
        self._current_task_waypoints = waypoint_list
        self._current_task_names = task_name_list

    def _make_single_ogbench_env(self, maze_type: str):
        make_maze_env = self._get_ogbench_make_maze_env()
        extra_kwargs = self._get_ogbench_maze_env_extra_kwargs()

        if "pointmaze" in self.env_id:
            env = make_maze_env(
                "point",
                "maze",
                maze_type=maze_type,
                width=200,
                height=200,
                **extra_kwargs,
            )
        elif "antmaze" in self.env_id:
            env = make_maze_env(
                "ant",
                "maze",
                maze_type=maze_type,
                width=200,
                height=200,
                **extra_kwargs,
            )
        else:
            raise RuntimeError(f"Unsupported OGBench env for benchmark preflight: {self.env_id}")
        self._prepare_ogbench_env(env)
        return env

    def _benchmark_preflight_check(self, task_id: int) -> None:
        if getattr(self, "_benchmark_preflight_done", False):
            return
        if self.env_id not in OGBENCH_ENVS:
            raise RuntimeError(f"benchmark preflight only supports OGBench envs, got {self.env_id}")

        self._verify_local_ogbench_runtime()
        maze_type = self.env_id.split("-")[1]
        env = self._make_single_ogbench_env(maze_type)
        rounded_starts = []
        rounded_goals = []

        try:
            for offset in range(5):
                reset_seed = self.benchmark_rollout_seed_base + offset
                self._seed_benchmark_rollout_rng(reset_seed)
                ob, info = env.reset(options=dict(task_id=int(task_id)))
                start_xy = tuple(round(float(v), 4) for v in np.asarray(ob)[:2])
                goal_xy = tuple(round(float(v), 4) for v in np.asarray(info["goal"])[:2])
                rounded_starts.append(start_xy)
                rounded_goals.append(goal_xy)
        finally:
            env.close()

        unique_starts = len(set(rounded_starts))
        unique_goals = len(set(rounded_goals))
        print(
            f"[BENCHMARK] Preflight perturbation check | task={task_id} "
            f"starts={rounded_starts} goals={rounded_goals}",
            flush=True,
        )
        if self.ogbench_enable_reset_perturb:
            if unique_starts < 2:
                raise RuntimeError(
                    f"Benchmark preflight failed: start perturbation not observed for task {task_id}. "
                    f"Samples={rounded_starts}"
                )
            if unique_goals < 2:
                raise RuntimeError(
                    f"Benchmark preflight failed: goal perturbation not observed for task {task_id}. "
                    f"Samples={rounded_goals}"
                )
        else:
            if unique_starts != 1:
                raise RuntimeError(
                    f"Benchmark preflight failed: start perturbation remained enabled for task {task_id}. "
                    f"Samples={rounded_starts}"
                )
            if unique_goals != 1:
                raise RuntimeError(
                    f"Benchmark preflight failed: goal perturbation remained enabled for task {task_id}. "
                    f"Samples={rounded_goals}"
                )
        self._benchmark_preflight_done = True

    def _write_benchmark_results(self, payload: dict) -> None:
        if not self.benchmark_results_path:
            raise ValueError("benchmark_results_path must be set for benchmark runs")

        import os

        os.makedirs(os.path.dirname(self.benchmark_results_path), exist_ok=True)
        with open(self.benchmark_results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _clear_parallel_plan_aux_state(self) -> None:
        """Drop stale denoising-side artifacts when a round skips parallel_plan()."""
        self._parallel_plan_step_captures = None
        self._parallel_plan_step_indices = None
        self._parallel_plan_step_total = None
        self._last_guidance_losses = {}

    @staticmethod
    def _compact_saved_task_info(task_info: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Trim large duplicated task metadata before writing benchmark payloads."""
        if task_info is None:
            return None

        compact = dict(task_info)
        compact.pop("waypoint_xy_groups", None)
        compact.pop("waypoint_ij_groups", None)

        waypoint_ij_group = compact.get("waypoint_ij_group")
        waypoint_ijs = compact.get("waypoint_ijs")
        if waypoint_ij_group is not None and waypoint_ijs is not None:
            if json.loads(json.dumps(waypoint_ij_group)) == json.loads(json.dumps(waypoint_ijs)):
                compact.pop("waypoint_ijs", None)

        return compact

    def _build_standard_benchmark_payload(self, all_task_results: list[dict]) -> dict:
        payload = {
            "model_id": self.benchmark_model_id,
            "eval_repeat_id": self.eval_repeat_id,
            "num_tasks": len(all_task_results),
            "task_results": all_task_results,
        }
        if all_task_results:
            repeat_success_mean = float(
                np.mean([task_result["task_success_mean"] for task_result in all_task_results])
            )
            payload["repeat_success_mean"] = repeat_success_mean
            self._safe_log_metric(
                f"benchmark/repeat{self.eval_repeat_id}/overall_success",
                repeat_success_mean,
            )
            repeat_post_lengths = [
                float(task_result["task_postprocessed_plan_length_mean"])
                for task_result in all_task_results
                if task_result.get("task_postprocessed_plan_length_mean") is not None
            ]
            payload["repeat_postprocessed_plan_length_mean"] = (
                float(np.mean(repeat_post_lengths))
                if repeat_post_lengths else None
            )
            if payload["repeat_postprocessed_plan_length_mean"] is not None:
                self._safe_log_metric(
                    f"benchmark/repeat{self.eval_repeat_id}/overall_postprocessed_plan_length",
                    float(payload["repeat_postprocessed_plan_length_mean"]),
                )
            repeat_groundtruth_lengths = [
                float(task_result["task_groundtruth_path_length_mean"])
                for task_result in all_task_results
                if task_result.get("task_groundtruth_path_length_mean") is not None
            ]
            payload["repeat_groundtruth_path_length_mean"] = (
                float(np.mean(repeat_groundtruth_lengths))
                if repeat_groundtruth_lengths else None
            )
            if payload["repeat_groundtruth_path_length_mean"] is not None:
                self._safe_log_metric(
                    f"benchmark/repeat{self.eval_repeat_id}/overall_groundtruth_path_length",
                    float(payload["repeat_groundtruth_path_length_mean"]),
                )
            repeat_rel_errors = [
                float(task_result["task_postprocessed_plan_length_rel_error_mean"])
                for task_result in all_task_results
                if task_result.get("task_postprocessed_plan_length_rel_error_mean") is not None
            ]
            payload["repeat_postprocessed_plan_length_rel_error_mean"] = (
                float(np.mean(repeat_rel_errors))
                if repeat_rel_errors else None
            )
            if payload["repeat_postprocessed_plan_length_rel_error_mean"] is not None:
                self._safe_log_metric(
                    f"benchmark/repeat{self.eval_repeat_id}/overall_postprocessed_plan_length_rel_error",
                    float(payload["repeat_postprocessed_plan_length_rel_error_mean"]),
                )
        if len(all_task_results) == 1:
            payload["task_id"] = all_task_results[0]["task_id"]
            payload["task_success_mean"] = all_task_results[0]["task_success_mean"]
            payload["task_postprocessed_plan_length_mean"] = all_task_results[0][
                "task_postprocessed_plan_length_mean"
            ]
            payload["task_groundtruth_path_length_mean"] = all_task_results[0][
                "task_groundtruth_path_length_mean"
            ]
            payload["task_postprocessed_plan_length_rel_error_mean"] = all_task_results[0][
                "task_postprocessed_plan_length_rel_error_mean"
            ]
        return payload

    def _build_hamiltonian_benchmark_payload(
        self,
        all_task_results: list[dict],
        metric_prefix: str,
    ) -> dict:
        payload = {
            "mode": "hamiltonian_benchmark",
            "model_id": self.benchmark_model_id,
            "planner_variant": self.benchmark_variant_name,
            "plan_only": bool(self.benchmark_plan_only),
            "eval_repeat_id": self.eval_repeat_id,
            "requested_top_n": int(self.benchmark_waypoint_top_n),
            "num_tasks": len(all_task_results),
            "task_results": all_task_results,
        }
        if all_task_results:
            repeat_success_vals = [
                float(task_result["task_agent_success_mean"])
                for task_result in all_task_results
                if task_result.get("task_agent_success_mean") is not None
            ]
            payload["repeat_agent_success_mean"] = (
                float(np.mean(repeat_success_vals)) if repeat_success_vals else None
            )
            payload["repeat_hamiltonian_path_success_mean"] = float(
                np.mean([task_result["task_hamiltonian_path_success_mean"] for task_result in all_task_results])
            )
            repeat_rel_errors = [
                float(task_result["task_postprocessed_plan_length_rel_error_mean"])
                for task_result in all_task_results
                if task_result.get("task_postprocessed_plan_length_rel_error_mean") is not None
            ]
            payload["repeat_postprocessed_plan_length_rel_error_mean"] = (
                float(np.mean(repeat_rel_errors)) if repeat_rel_errors else None
            )
            if payload["repeat_agent_success_mean"] is not None:
                self._safe_log_metric(
                    f"{metric_prefix}/repeat{self.eval_repeat_id}/overall_agent_success",
                    payload["repeat_agent_success_mean"],
                )
            self._safe_log_metric(
                f"{metric_prefix}/repeat{self.eval_repeat_id}/overall_hamiltonian_path_success",
                payload["repeat_hamiltonian_path_success_mean"],
            )
            if payload["repeat_postprocessed_plan_length_rel_error_mean"] is not None:
                self._safe_log_metric(
                    f"{metric_prefix}/repeat{self.eval_repeat_id}/overall_postprocessed_plan_length_rel_error",
                    float(payload["repeat_postprocessed_plan_length_rel_error_mean"]),
                )
        if len(all_task_results) == 1:
            payload["task_id"] = all_task_results[0]["task_id"]
            payload["task_agent_success_mean"] = all_task_results[0]["task_agent_success_mean"]
            payload["task_hamiltonian_path_success_mean"] = all_task_results[0]["task_hamiltonian_path_success_mean"]
            payload["task_postprocessed_plan_length_rel_error_mean"] = all_task_results[0][
                "task_postprocessed_plan_length_rel_error_mean"
            ]
        return payload

    def _run_standard_benchmark(self, task_ids: list[int]) -> dict:
        all_task_results = []
        original_task_id = self.task_id
        original_seed = self.interaction_seed
        original_visual_prefix = self._current_wandb_visual_prefix

        try:
            for task_idx, task_id in enumerate(task_ids):
                self._benchmark_preflight_check(int(task_id))
                task_rollouts = []
                task_seed_base = self.benchmark_rollout_seed_base + task_idx * self.benchmark_num_rollouts
                rollout_pbar = tqdm(
                    total=self.benchmark_num_rollouts,
                    desc=f"Benchmark R{self.eval_repeat_id} T{task_id}",
                    leave=True,
                    dynamic_ncols=True,
                )
                for rollout_idx in range(self.benchmark_num_rollouts):
                    rollout_seed = task_seed_base + rollout_idx
                    self._seed_benchmark_rollout_rng(rollout_seed)
                    self.task_id = int(task_id)
                    self.interaction_seed = rollout_seed
                    self._current_wandb_visual_prefix = f"task{self.task_id}_rollout{rollout_idx}"
                    namespace = (
                        f"benchmark/repeat{self.eval_repeat_id}/task{self.task_id}/rollout{rollout_idx}"
                    )
                    self.interact(
                        batch_size=1,
                        conditions=None,
                        namespace=namespace,
                        terminate_on_done=True,
                    )
                    interact_result = dict(getattr(self, "_last_interact_result", {}))
                    interact_result.update(
                        {
                            "rollout_id": rollout_idx,
                            "seed": rollout_seed,
                        }
                    )

                    task_geometry = self._extract_interact_task_geometry(
                        interact_result,
                        include_waypoints=False,
                    )
                    stage_prefix = (
                        f"[BENCH] task={self.task_id} rollout={rollout_idx} "
                        f"seed={rollout_seed}"
                    )
                    t_stage = time.time()
                    print(f"{stage_prefix} stage=groundtruth_info start", flush=True)
                    groundtruth_info = self._compute_groundtruth_path_info(**task_geometry)
                    print(
                        f"{stage_prefix} stage=groundtruth_info done "
                        f"(elapsed={time.time() - t_stage:.1f}s)",
                        flush=True,
                    )
                    interact_result = self._augment_interact_result_with_groundtruth_path_metrics(
                        interact_result,
                        groundtruth_info,
                    )
                    self._log_groundtruth_path_media(
                        groundtruth_info,
                        stage_prefix=stage_prefix,
                        media_leaf_key="groundtruth_path",
                    )

                    task_rollouts.append(interact_result)
                    valid_successes = [
                        float(rollout["success"])
                        for rollout in task_rollouts
                        if rollout.get("success") is not None
                    ]
                    running_success = (
                        float(np.mean(valid_successes))
                        if valid_successes else float("nan")
                    )
                    valid_post_lengths = [
                        float(rollout["postprocessed_plan_length"])
                        for rollout in task_rollouts
                        if rollout.get("postprocessed_plan_length") is not None
                    ]
                    running_post_length = (
                        float(np.mean(valid_post_lengths))
                        if valid_post_lengths else float("nan")
                    )
                    valid_groundtruth_lengths = [
                        float(rollout["groundtruth_path_length"])
                        for rollout in task_rollouts
                        if rollout.get("groundtruth_path_length") is not None
                        and np.isfinite(float(rollout["groundtruth_path_length"]))
                    ]
                    running_groundtruth_length = (
                        float(np.mean(valid_groundtruth_lengths))
                        if valid_groundtruth_lengths else float("nan")
                    )
                    valid_rel_errors = [
                        float(rollout["postprocessed_plan_length_rel_error"])
                        for rollout in task_rollouts
                        if rollout.get("postprocessed_plan_length_rel_error") is not None
                    ]
                    running_rel_error = (
                        float(np.mean(valid_rel_errors))
                        if valid_rel_errors else float("nan")
                    )

                    rollout_pbar.update(1)
                    rollout_pbar.set_postfix(
                        {
                            "success": (
                                f"{running_success:.3f}"
                                if np.isfinite(running_success)
                                else "nan"
                            ),
                            "post_len": (
                                f"{running_post_length:.2f}"
                                if np.isfinite(running_post_length)
                                else "nan"
                            ),
                            "gt_len": (
                                f"{running_groundtruth_length:.2f}"
                                if np.isfinite(running_groundtruth_length)
                                else "nan"
                            ),
                            "relerr": (
                                f"{running_rel_error:.3f}"
                                if np.isfinite(running_rel_error)
                                else "nan"
                            ),
                            "last_steps": interact_result.get("steps", 0),
                        },
                        refresh=True,
                    )
                    self._safe_log_metric("benchmark/current_repeat_id", float(self.eval_repeat_id))
                    self._safe_log_metric("benchmark/current_task_id", float(self.task_id))
                    self._safe_log_metric("benchmark/current_rollout", float(rollout_idx + 1))
                    if np.isfinite(running_success):
                        self._safe_log_metric("benchmark/running_task_success", running_success)
                    if interact_result.get("postprocessed_plan_length") is not None:
                        self._safe_log_metric(
                            f"{namespace}/postprocessed_plan_length",
                            float(interact_result["postprocessed_plan_length"]),
                        )
                    self._safe_log_metric(
                        f"{namespace}/groundtruth_path_reachable",
                        float(bool(interact_result.get("groundtruth_reachable", False))),
                    )
                    if interact_result.get("groundtruth_path_length") is not None and np.isfinite(
                        float(interact_result["groundtruth_path_length"])
                    ):
                        self._safe_log_metric(
                            f"{namespace}/groundtruth_path_length",
                            float(interact_result["groundtruth_path_length"]),
                        )
                    if interact_result.get("postprocessed_plan_length_rel_error") is not None:
                        self._safe_log_metric(
                            f"{namespace}/postprocessed_plan_length_rel_error",
                            float(interact_result["postprocessed_plan_length_rel_error"]),
                        )

                valid_task_success = [
                    float(rollout["success"])
                    for rollout in task_rollouts
                    if rollout.get("success") is not None
                ]
                task_success_mean = (
                    float(np.mean(valid_task_success))
                    if valid_task_success else float("nan")
                )
                valid_task_post_lengths = [
                    float(rollout["postprocessed_plan_length"])
                    for rollout in task_rollouts
                    if rollout.get("postprocessed_plan_length") is not None
                ]
                task_postprocessed_plan_length_mean = (
                    float(np.mean(valid_task_post_lengths))
                    if valid_task_post_lengths else None
                )
                valid_task_groundtruth_lengths = [
                    float(rollout["groundtruth_path_length"])
                    for rollout in task_rollouts
                    if rollout.get("groundtruth_path_length") is not None
                    and np.isfinite(float(rollout["groundtruth_path_length"]))
                ]
                task_groundtruth_path_length_mean = (
                    float(np.mean(valid_task_groundtruth_lengths))
                    if valid_task_groundtruth_lengths else None
                )
                valid_task_rel_errors = [
                    float(rollout["postprocessed_plan_length_rel_error"])
                    for rollout in task_rollouts
                    if rollout.get("postprocessed_plan_length_rel_error") is not None
                ]
                task_postprocessed_plan_length_rel_error_mean = (
                    float(np.mean(valid_task_rel_errors))
                    if valid_task_rel_errors else None
                )

                rollout_pbar.close()
                self._safe_log_metric(
                    f"benchmark/repeat{self.eval_repeat_id}/task{task_id}/success_mean",
                    task_success_mean,
                )
                if task_postprocessed_plan_length_mean is not None:
                    self._safe_log_metric(
                        f"benchmark/repeat{self.eval_repeat_id}/task{task_id}/postprocessed_plan_length_mean",
                        task_postprocessed_plan_length_mean,
                    )
                if task_groundtruth_path_length_mean is not None:
                    self._safe_log_metric(
                        f"benchmark/repeat{self.eval_repeat_id}/task{task_id}/groundtruth_path_length_mean",
                        task_groundtruth_path_length_mean,
                    )
                if task_postprocessed_plan_length_rel_error_mean is not None:
                    self._safe_log_metric(
                        f"benchmark/repeat{self.eval_repeat_id}/task{task_id}/postprocessed_plan_length_rel_error_mean",
                        task_postprocessed_plan_length_rel_error_mean,
                    )
                all_task_results.append(
                    {
                        "task_id": int(task_id),
                        "num_rollouts": len(task_rollouts),
                        "task_success_mean": task_success_mean,
                        "task_postprocessed_plan_length_mean": task_postprocessed_plan_length_mean,
                        "task_groundtruth_path_length_mean": task_groundtruth_path_length_mean,
                        "task_postprocessed_plan_length_rel_error_mean": (
                            task_postprocessed_plan_length_rel_error_mean
                        ),
                        "rollouts": task_rollouts,
                    }
                )
                self._write_benchmark_results(
                    self._build_standard_benchmark_payload(all_task_results)
                )
        finally:
            self.task_id = original_task_id
            self.interaction_seed = original_seed
            self._current_wandb_visual_prefix = original_visual_prefix

        payload = self._build_standard_benchmark_payload(all_task_results)
        self._write_benchmark_results(payload)
        return payload

    def _run_hamiltonian_benchmark(self, task_ids: list[int]) -> dict:
        if self.task_override_resolved_path is None:
            raise ValueError("Hamiltonian benchmark requires algorithm.task_override_path")
        if self.benchmark_waypoint_top_n is None or self.benchmark_waypoint_top_n <= 0:
            raise ValueError("Hamiltonian benchmark requires algorithm.benchmark_waypoint_top_n > 0")

        all_task_results = []
        original_task_id = self.task_id
        original_seed = self.interaction_seed
        original_group_idx = self.task_override_waypoint_group_idx
        original_visual_prefix = self._current_wandb_visual_prefix
        metric_prefix = self._benchmark_metric_prefix()

        try:
            for task_idx, task_id in enumerate(task_ids):
                self._benchmark_preflight_check(int(task_id))
                ranked_groups = self._get_benchmark_waypoint_groups_for_task(int(task_id))
                selected_groups = ranked_groups[: self.benchmark_waypoint_top_n]
                if not selected_groups:
                    raise ValueError(
                        f"No waypoint groups available for Hamiltonian benchmark task {task_id} "
                        f"in {self.task_override_resolved_path}"
                    )

                task_group_results = []
                task_seed_base = self.benchmark_rollout_seed_base + task_idx * max(1, len(selected_groups))
                group_pbar = tqdm(
                    total=len(selected_groups),
                    desc=f"HBench R{self.eval_repeat_id} T{task_id}",
                    leave=True,
                    dynamic_ncols=True,
                )

                for local_group_rank, (group_idx, waypoint_ij_group) in enumerate(selected_groups):
                    group_seed = task_seed_base + local_group_rank
                    self._seed_benchmark_rollout_rng(group_seed)
                    self.task_id = int(task_id)
                    self.interaction_seed = group_seed
                    self.task_override_waypoint_group_idx = int(group_idx)
                    self._current_wandb_visual_prefix = f"task{self.task_id}_group{group_idx}"
                    namespace = (
                        f"{metric_prefix}/repeat{self.eval_repeat_id}/task{self.task_id}/group{group_idx}"
                    )
                    self.interact(
                        batch_size=1,
                        conditions=None,
                        namespace=namespace,
                        terminate_on_done=True,
                    )
                    interact_result = dict(getattr(self, "_last_interact_result", {}))
                    interact_result.update(
                        {
                            "waypoint_group_idx": int(group_idx),
                            "waypoint_ij_group": waypoint_ij_group,
                            "seed": int(group_seed),
                        }
                    )

                    task_geometry = self._extract_interact_task_geometry(
                        interact_result,
                        include_waypoints=True,
                    )
                    stage_prefix = (
                        f"[HBENCH] task={self.task_id} group={group_idx} "
                        f"seed={group_seed}"
                    )
                    t_stage = time.time()
                    print(f"{stage_prefix} stage=groundtruth_info start", flush=True)
                    groundtruth_info = self._compute_groundtruth_hamiltonian_info(**task_geometry)
                    print(
                        f"{stage_prefix} stage=groundtruth_info done "
                        f"(elapsed={time.time() - t_stage:.1f}s)",
                        flush=True,
                    )
                    interact_result = self._augment_interact_result_with_hamiltonian_groundtruth(
                        interact_result,
                        groundtruth_info,
                    )

                    if self.benchmark_log_groundtruth_media:
                        self._log_groundtruth_path_media(
                            groundtruth_info,
                            stage_prefix=stage_prefix,
                            media_leaf_key="groundtruth_hemiltonian_path",
                        )
                        t_stage = time.time()
                        print(f"{stage_prefix} stage=render_groundtruth_temporal start", flush=True)
                        gt_temporal_image = self._render_groundtruth_temporal_distance_image(
                            groundtruth_info
                        )
                        print(
                            f"{stage_prefix} stage=render_groundtruth_temporal done "
                            f"(elapsed={time.time() - t_stage:.1f}s, image={'yes' if gt_temporal_image is not None else 'no'})",
                            flush=True,
                        )
                        if gt_temporal_image is not None:
                            t_log = time.time()
                            print(f"{stage_prefix} stage=log_image_groundtruth_temporal start", flush=True)
                            self.log_image(
                                self._benchmark_media_key("groundtruth_temporal_distance"),
                                gt_temporal_image,
                            )
                            print(
                                f"{stage_prefix} stage=log_image_groundtruth_temporal done "
                                f"(elapsed={time.time() - t_log:.1f}s)",
                                flush=True,
                            )

                    task_group_results.append(interact_result)
                    valid_group_success = [
                        float(group["success"])
                        for group in task_group_results
                        if group.get("success") is not None
                    ]
                    running_agent_success = (
                        float(np.mean(valid_group_success))
                        if valid_group_success else None
                    )
                    running_hamiltonian_success = float(
                        np.mean([group["hamiltonian_path_success"] for group in task_group_results])
                    )
                    valid_rel_errors = [
                        float(group["postprocessed_plan_length_rel_error"])
                        for group in task_group_results
                        if group.get("postprocessed_plan_length_rel_error") is not None
                    ]
                    running_rel_error = (
                        float(np.mean(valid_rel_errors))
                        if valid_rel_errors else float("nan")
                    )
                    group_pbar.update(1)
                    group_pbar.set_postfix(
                        {
                            "agent": (
                                f"{running_agent_success:.3f}"
                                if running_agent_success is not None
                                else "n/a"
                            ),
                            "route": f"{running_hamiltonian_success:.3f}",
                            "relerr": (
                                f"{running_rel_error:.3f}"
                                if np.isfinite(running_rel_error)
                                else "nan"
                            ),
                        },
                        refresh=True,
                    )

                    self._safe_log_metric(f"{metric_prefix}/current_repeat_id", float(self.eval_repeat_id))
                    self._safe_log_metric(f"{metric_prefix}/current_task_id", float(self.task_id))
                    self._safe_log_metric(f"{metric_prefix}/current_group_idx", float(group_idx))
                    if interact_result.get("success") is not None:
                        self._safe_log_metric(
                            f"{metric_prefix}/repeat{self.eval_repeat_id}/task{task_id}/group{group_idx}/agent_success",
                            float(interact_result["success"]),
                        )
                    self._safe_log_metric(
                        f"{metric_prefix}/repeat{self.eval_repeat_id}/task{task_id}/group{group_idx}/hamiltonian_path_success",
                        float(interact_result["hamiltonian_path_success"]),
                    )
                    if interact_result.get("postprocessed_plan_length_rel_error") is not None:
                        self._safe_log_metric(
                            f"{metric_prefix}/repeat{self.eval_repeat_id}/task{task_id}/group{group_idx}/postprocessed_plan_length_rel_error",
                            float(interact_result["postprocessed_plan_length_rel_error"]),
                        )

                group_pbar.close()
                task_success_vals = [
                    float(group["success"])
                    for group in task_group_results
                    if group.get("success") is not None
                ]
                task_agent_success_mean = (
                    float(np.mean(task_success_vals)) if task_success_vals else None
                )
                task_hamiltonian_success_mean = float(
                    np.mean([group["hamiltonian_path_success"] for group in task_group_results])
                )
                task_rel_errors = [
                    float(group["postprocessed_plan_length_rel_error"])
                    for group in task_group_results
                    if group.get("postprocessed_plan_length_rel_error") is not None
                ]
                task_length_error_mean = (
                    float(np.mean(task_rel_errors)) if task_rel_errors else None
                )
                if task_agent_success_mean is not None:
                    self._safe_log_metric(
                        f"{metric_prefix}/repeat{self.eval_repeat_id}/task{task_id}/agent_success_mean",
                        task_agent_success_mean,
                    )
                self._safe_log_metric(
                    f"{metric_prefix}/repeat{self.eval_repeat_id}/task{task_id}/hamiltonian_path_success_mean",
                    task_hamiltonian_success_mean,
                )
                if task_length_error_mean is not None:
                    self._safe_log_metric(
                        f"{metric_prefix}/repeat{self.eval_repeat_id}/task{task_id}/postprocessed_plan_length_rel_error_mean",
                        task_length_error_mean,
                    )
                all_task_results.append(
                    {
                        "task_id": int(task_id),
                        "planner_variant": self.benchmark_variant_name,
                        "plan_only": bool(self.benchmark_plan_only),
                        "requested_top_n": int(self.benchmark_waypoint_top_n),
                        "num_groups": len(task_group_results),
                        "task_agent_success_mean": task_agent_success_mean,
                        "task_hamiltonian_path_success_mean": task_hamiltonian_success_mean,
                        "task_postprocessed_plan_length_rel_error_mean": task_length_error_mean,
                        "group_results": task_group_results,
                    }
                )
                self._write_benchmark_results(
                    self._build_hamiltonian_benchmark_payload(all_task_results, metric_prefix)
                )
        finally:
            self.task_id = original_task_id
            self.interaction_seed = original_seed
            self.task_override_waypoint_group_idx = original_group_idx
            self._current_wandb_visual_prefix = original_visual_prefix

        payload = self._build_hamiltonian_benchmark_payload(all_task_results, metric_prefix)
        self._write_benchmark_results(payload)
        return payload

    def run_benchmark(self) -> dict:
        task_ids = self.task_ids if self.task_ids is not None else [self.task_id]
        task_ids = [tid for tid in task_ids if tid is not None]
        if not task_ids:
            raise ValueError("benchmark run requires algorithm.task_id or algorithm.task_ids")
        if (
            self.task_override_resolved_path is not None
            and self.benchmark_waypoint_top_n is not None
        ):
            return self._run_hamiltonian_benchmark(task_ids)
        return self._run_standard_benchmark(task_ids)

    
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
        prefix_len: Optional[int] = None,
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

        if prefix_len is not None:
            start_idx = prefix_len
            if start_idx >= plan_tokens:
                if not is_replanning:
                    return np.expand_dims(level_array, 0)
                # is_replanning + fully denoised: treat same as all-zeros case
                start_idx = plan_tokens
        else:
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
                - m: number of denoising steps (M from the tree-conditioned schedule)
                - plan_tokens: horizon // frame_stack tokens to be denoised
            plans: list of b pre-built plan tensors, each shape (n_tokens, 1, fs*c)
                - n_tokens: total tokens (including padding)
                - 1: batch dimension per leaf
                - fs*c: flattened observation representation

        Returns:
            plan_hist: (m+1, plan_tokens*fs, b, c) tensor of denoising histories
                - m: number of denoising steps (M from the tree-conditioned schedule)
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

            # Sliding-window mode: each batch item has a single active frame range
            # covering [window_start_frame, window_end_frame) in half-open form.
            # Start is (pl+1)*fs (AFTER obs_parent token) to prevent the previous
            # segment's tail from being selected as a guidance target.
            _active_frame_ranges = None
            if self.use_segment_wise_sliding_window and prefix_len_list is not None:
                _seg_t = horizon // self.frame_stack // self.sequence_dividing_factor
                _active_frame_ranges = [
                    (
                        (pl + 1) * self.frame_stack,
                        (pl + 1 + _seg_t) * self.frame_stack,
                    )
                    for pl in prefix_len_list
                ]

            guidance_fn = lambda x: guidance.combined_guidance(
                self, x, goal, horizon, guidance_scale,
                particle_guidance_scale=_pgs, group_ids=_gids,
                active_frame_ranges=_active_frame_ranges,
            )

        # [TIMING] Wrap guidance_fn to measure per-step cost and capture last loss values
        _guidance_call_ms: list = []
        _last_guidance_losses: dict = {}  # populated with scalar loss values from last call
        if guidance_scale is not None:
            _raw_gfn = guidance_fn
            def guidance_fn(x, _fn=_raw_gfn):
                _gt0 = time.time()
                # Sliding window: scatter (1+seg, b, fs*c) → (n_tokens, b, fs*c) so guidance
                # functions (which index by absolute frame position) work correctly.
                r = _fn(_scatter_window_to_full(x, self.n_tokens, _sc_plens) if self.use_segment_wise_sliding_window else x)
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

        # [OOB-LOGGING] Initialize buffer for window out-of-map analysis
        if call_type == "expansion" and self.use_segment_wise_sliding_window:
            self._oob_log_buffer = []
        elif hasattr(self, '_oob_log_buffer'):
            del self._oob_log_buffer

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
        # Sliding-window hook support: precompute per-element prefix offsets (in token units).
        # When use_segment_wise_sliding_window=True, sample_step receives a window of
        # (1+seg, B, fs*c) instead of (n_tokens, B, fs*c).  The hook must:
        #   1. extract with seg tokens and relative prefix_len=0 (obs_parent is always window[0])
        #   2. pad the result back to (plan_tokens*fs, B, c) at the correct per-element offset
        #      so downstream viz code (plan_viz.py tail_vis indexing) stays untouched.
        _sc_seg = plan_tokens // self.sequence_dividing_factor if self.sequence_dividing_factor else plan_tokens
        _sc_plens = [
            (prefix_len_list[i] if prefix_len_list is not None else 0)
            for i in range(batch_size)
        ]  # per-element absolute prefix_len (token units); used regardless of use_dynamic_obs_padding
        _sc_frame_offsets = [pl * _sc_fs for pl in _sc_plens]  # same, but in frame units

        def _scatter_window_to_full(win_tensor, full_t, offsets):
            """Scatter (win_t, B, *rest) into zeros (full_t, B, *rest) at per-element offsets.

            offsets[i] is the start index for batch element i in the full tensor's first dim.
            Uses torch.index_put (non-in-place) so autograd is preserved through win_tensor.
            Used by both the denoising-step hook (capture) and guidance_fn (gradient computation).
            """
            win_t, B = win_tensor.shape[0], win_tensor.shape[1]
            rest = win_tensor.shape[2:]
            full = torch.zeros(full_t, B, *rest, dtype=win_tensor.dtype, device=win_tensor.device)
            idx_t = torch.tensor(
                [[offsets[i] + t for i in range(B)] for t in range(win_t)],
                dtype=torch.long, device=win_tensor.device,
            )  # (win_t, B)
            idx_b = torch.arange(B, device=win_tensor.device).unsqueeze(0).expand_as(idx_t)
            return torch.index_put(
                full,
                (idx_t.reshape(-1), idx_b.reshape(-1)),
                win_tensor.reshape(win_t * B, *rest),
            )

        def _capture_hook(data):
            """Extract plan-space pred_noise & guidance grads, move to CPU."""
            if self.use_segment_wise_sliding_window:
                # Window inputs: (1+seg, B, fs*c).  obs_parent is at window[0], so relative
                # prefix_len=0 for all elements.  Extract seg tokens, then pad to plan_tokens.
                win_plan_tokens = _sc_seg
                win_prefix_list = [0] * batch_size
            else:
                win_plan_tokens = _sc_plan_tokens
                win_prefix_list = _sc_prefix_len_list

            pn = data['prior_pred_noise']
            pn_chunk = extract_plan_chunk(pn, win_plan_tokens, win_prefix_list)
            pn_exp = rearrange(pn_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs)
            if self.use_segment_wise_sliding_window:
                pn_exp = _scatter_window_to_full(pn_exp, _sc_plan_tokens * _sc_fs, _sc_frame_offsets)

            gg_exp = {}
            for k, v in data['guidance_grads'].items():
                v_chunk = extract_plan_chunk(v.detach(), win_plan_tokens, win_prefix_list)
                v_exp = rearrange(v_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs)
                if self.use_segment_wise_sliding_window:
                    v_exp = _scatter_window_to_full(v_exp, _sc_plan_tokens * _sc_fs, _sc_frame_offsets)
                gg_exp[k] = v_exp.cpu()
            # Clean-space grads: ∂V/∂x̂_0 (no Jacobian — matches crimson HILP grad field direction)
            gg_clean_exp = {}
            for k, v in data.get('guidance_grads_clean', {}).items():
                v_chunk = extract_plan_chunk(v.detach(), win_plan_tokens, win_prefix_list)
                v_exp = rearrange(v_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs)
                if self.use_segment_wise_sliding_window:
                    v_exp = _scatter_window_to_full(v_exp, _sc_plan_tokens * _sc_fs, _sc_frame_offsets)
                gg_clean_exp[k] = v_exp.cpu()
            gg_xs_disp_exp = {}
            for k, v in data.get('guidance_xstart_displacements', {}).items():
                v_chunk = extract_plan_chunk(v.detach(), win_plan_tokens, win_prefix_list)
                v_exp = rearrange(v_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs)
                if self.use_segment_wise_sliding_window:
                    v_exp = _scatter_window_to_full(v_exp, _sc_plan_tokens * _sc_fs, _sc_frame_offsets)
                gg_xs_disp_exp[k] = v_exp.cpu()
            # pred_x_start: x̂_0 before guidance (n_tokens, B, fs*c)
            pxs = data.get('pred_x_start')
            if pxs is not None:
                pxs_chunk = extract_plan_chunk(pxs.detach(), win_plan_tokens, win_prefix_list)
                pxs_exp = rearrange(pxs_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs)
                if self.use_segment_wise_sliding_window:
                    pxs_exp = _scatter_window_to_full(pxs_exp, _sc_plan_tokens * _sc_fs, _sc_frame_offsets)
                pxs_exp = pxs_exp.cpu()
            else:
                pxs_exp = None
            # pred_x_start_after: x̂_0 after guidance correction (= before if no guidance)
            pxs_after = data.get('pred_x_start_after')
            if pxs_after is not None:
                pxs_after_chunk = extract_plan_chunk(pxs_after.detach(), win_plan_tokens, win_prefix_list)
                pxs_after_exp = rearrange(pxs_after_chunk, "t b (fs c) -> (t fs) b c", fs=_sc_fs)
                if self.use_segment_wise_sliding_window:
                    pxs_after_exp = _scatter_window_to_full(pxs_after_exp, _sc_plan_tokens * _sc_fs, _sc_frame_offsets)
                pxs_after_exp = pxs_after_exp.cpu()
            else:
                pxs_after_exp = None
            # Capture effective DDIM scale: sqrt(1 - alpha_t) for each plan token.
            # curr_noise_level is an INTEGER timestep index (0..timesteps-1), NOT a float.
            # Look up alphas_cumprod[t] to get the true alpha, then sqrt(1-alpha) ≈ c coefficient.
            # High noise (early denoising, large t) → alpha small → sqrt(1-alpha) ≈ 1 (large arrows)
            # Low noise (late denoising, small t) → alpha → 1 → sqrt(1-alpha) ≈ 0 (small arrows)
            nl = data.get('curr_noise_level')  # (n_tokens, B) integer GPU tensor
            if nl is not None:
                nl_chunk = extract_plan_chunk(nl.unsqueeze(-1), win_plan_tokens, win_prefix_list)  # (win_t, B, 1)
                if self.use_segment_wise_sliding_window:
                    # nl_chunk is in token-space: use token offsets (_sc_plens), not frame offsets
                    nl_chunk = _scatter_window_to_full(nl_chunk, _sc_plan_tokens, _sc_plens)  # (plan_tokens, B, 1)
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
                'guidance_xstart_displacements': gg_xs_disp_exp,  # exact Δx̂_0_k in direct-x0 mode
                'pred_x_start_pos': pxs_exp,               # x̂_0 before guidance (plan_tokens*fs, B, c) or None
                'pred_x_start_pos_after': pxs_after_exp,   # x̂_0 after guidance (plan_tokens*fs, B, c) or None
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

            if self.use_segment_wise_sliding_window:
                # Per-element sliding window: feed [plen_i : plen_i+1+seg] to sample_step.
                # NOTE: tokens are re-indexed from 0 inside sample_step (positional embedding
                # mismatch with full-sequence mode is intentional).
                # _sc_plens and _sc_seg are precomputed in the hook closure above.
                device = plan_with_given_tokens.device
                # indices[t, i] = _sc_plens[i] + t  →  shape (1+_sc_seg, b)
                indices = torch.tensor(
                    [[_sc_plens[i] + t for i in range(batch_size)] for t in range(1 + _sc_seg)],
                    dtype=torch.long, device=device,
                )  # (1+seg, b)
                b_idx = torch.arange(batch_size, device=device)  # (b,)

                plan_win = plan_with_given_tokens[indices, b_idx]  # (1+seg, b, fs*c)
                from_win = from_noise_levels[indices, b_idx]        # (1+seg, b)
                to_win   = to_noise_levels[indices, b_idx]          # (1+seg, b)

                # Keep the stored tail token unchanged, but feed sample_step a training-style
                # init token rebuilt from the last frame of the window anchor.
                plan_win_input = plan_win.clone()
                plan_win_input[0] = self._repad_stacked_init_from_last_frame(plan_win[0])

                sample_win = self.diffusion_model.sample_step(
                    plan_win_input, conditions, from_win, to_win, guidance_fn=guidance_fn,
                )  # (1+seg, b, fs*c)

                update_mask_win = (from_win > to_win).unsqueeze(-1)  # (1+seg, b, 1)
                updated_win = torch.where(update_mask_win, sample_win, plan_win)

                # [CLIP-X-START] Clamp window tokens to valid normalized range to prevent
                # guidance explosion (observed: guidance delta up to 12000 world coords at high noise).
                # Controlled by use_sliding_window_clip_x_start flag; clip value from diffusion.clip_x_start.
                if self.cfg.get('use_sliding_window_clip_x_start', False):
                    _clip_xs = float(self.cfg.diffusion.get('clip_x_start', float('inf')))
                    if _clip_xs < float('inf'):
                        updated_win = updated_win.clamp(-_clip_xs, _clip_xs)

                plan_with_given_tokens = plan_with_given_tokens.clone()
                plan_with_given_tokens[indices, b_idx] = updated_win

                # [OOB-LOGGING] Buffer window positions for out-of-map analysis
                if call_type == "expansion" and hasattr(self, '_oob_log_buffer'):
                    try:
                        _fs = self.frame_stack
                        _pidx = self.pos_dim_indices[:2]
                        # updated_win: (1+seg, b, fs*c) → unnormalize → get xy
                        _win_unnorm = self._unnormalize_x(
                            rearrange(updated_win.detach().cpu(), "t b (fs c) -> (t fs) b c", fs=_fs)
                        )  # ((1+seg)*fs, b, c)
                        _xy = _win_unnorm[:, :, _pidx].numpy()  # ((1+seg)*fs, b, 2)
                        _from_nl = from_win.detach().cpu().numpy()  # (1+seg, b)

                        # [OOB-LOGGING v2] Capture pred_x0 before/after guidance for last seg token
                        # _step_captures[-1] is populated by _capture_hook from ddim_sample_step
                        _pred_x0_before = None
                        _pred_x0_after = None
                        if _step_captures:
                            _cap = _step_captures[-1]
                            # pred_x_start_pos_after: (plan_tokens*fs, B, c) in scattered form
                            _pxs_after = _cap.get('pred_x_start_pos_after')  # may be None
                            _pxs_before = _cap.get('pred_x_start_pos')       # None if no guidance
                            if _pxs_after is not None:
                                # Extract last seg token frames in scattered space.
                                # extract_plan_chunk removes obs_parent token (tok0), so:
                                #   tok k in window → seg-token index (k-1) in chunk → frames (k-1)*fs..(k)*fs
                                # After scatter with offset _sc_frame_offsets[i] = _sc_plens[i]*fs:
                                #   tok5 (= seg index 4) → scattered frames: _sc_plens[i]*fs + 4*fs
                                _tok5_after = []
                                _tok5_before = []
                                for _bi in range(batch_size):
                                    _fstart = _sc_frame_offsets[_bi] + (_sc_seg - 1) * _fs
                                    _fend = _fstart + _fs
                                    _t5a = self._unnormalize_x(_pxs_after[_fstart:_fend, _bi:_bi+1])[:, 0, _pidx]
                                    _tok5_after.append(_t5a.numpy().tolist())
                                    if _pxs_before is not None:
                                        _t5b = self._unnormalize_x(_pxs_before[_fstart:_fend, _bi:_bi+1])[:, 0, _pidx]
                                        _tok5_before.append(_t5b.numpy().tolist())
                                _pred_x0_before = _tok5_before if _tok5_before else None
                                _pred_x0_after = _tok5_after

                        self._oob_log_buffer.append({
                            "step": m,
                            "prefix_lens": _sc_plens,
                            "xy": _xy.tolist(),       # ((1+seg)*fs, b, 2)
                            "from_nl": _from_nl.tolist(),  # (1+seg, b)
                            "pred_x0_before": _pred_x0_before,  # list[b, fs, 2] or None
                            "pred_x0_after": _pred_x0_after,    # list[b, fs, 2] or None
                        })
                    except Exception:
                        pass
            else:
                sample = self.diffusion_model.sample_step(
                    plan_with_given_tokens,  # (n_tokens, b, fs*c)
                    conditions,
                    from_noise_levels,  # (n_tokens, b)
                    to_noise_levels,  # (n_tokens, b)
                    guidance_fn=guidance_fn,
                )  # (n_tokens, b, fs*c)

                # Update only tokens whose noise level is actively decreasing this step.
                # This preserves denoised_prefix (level=0) and obs_parent_token (level=0).
                update_mask = (from_noise_levels > to_noise_levels).unsqueeze(
                    -1
                )  # (n_tokens, b, 1) broadcast mask
                plan_with_given_tokens = torch.where(
                    update_mask, sample, plan_with_given_tokens
                )  # (n_tokens, b, fs*c)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _gpu_step_times.append(time.time() - _gpu_t0)

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
        _step_captures.insert(0, {'prior_pred_noise': None, 'guidance_grads': {}, 'guidance_grads_clean': {}, 'guidance_xstart_displacements': {}, 'pred_x_start_pos': None, 'pred_x_start_pos_after': None})

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

        # [OOB-LOGGING] Flush buffer if any OOB detected
        if call_type == "expansion" and hasattr(self, '_oob_log_buffer') and self._oob_log_buffer:
            try:
                import json, os
                _oob_data = self._oob_log_buffer
                # Detect OOB: unnormalized xy outside ±40 (generous bound for antmaze-giant)
                _OOB_THRESH = 40.0
                _has_oob = False
                for _entry in _oob_data:
                    for _t_row in _entry["xy"]:
                        for _b_xy in _t_row:
                            if abs(_b_xy[0]) > _OOB_THRESH or abs(_b_xy[1]) > _OOB_THRESH:
                                _has_oob = True
                                break
                        if _has_oob:
                            break
                    if _has_oob:
                        break
                if _has_oob:
                    _log_path = os.path.join("logs", "window_oob.jsonl")
                    os.makedirs("logs", exist_ok=True)
                    with open(_log_path, "a") as _f:
                        # Write header record
                        _f.write(json.dumps({
                            "event": "oob_episode",
                            "call_type": call_type,
                            "batch_size": batch_size,
                            "seg_size": _sc_seg,
                            "frame_stack": self.frame_stack,
                            "n_steps": len(_oob_data),
                        }) + "\n")
                        for _entry in _oob_data:
                            _f.write(json.dumps(_entry) + "\n")
                del self._oob_log_buffer
            except Exception as _e:
                pass

        return plan_hist  # (m+1, plan_tokens*fs, b, c)

    def interact(
        self,
        batch_size: int,
        conditions: Optional[Any] = None,
        namespace: str = "validation",
        terminate_on_done: bool = False,
    ) -> None:
        # Lazy KDE load: skip during training, load once on first eval/interact call.
        if self.kde_lam > 0.0 or self.use_kde_maximin_for_selecting_subplan_in_cluster:
            self._ensure_kde_cache_loaded()

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
            # task_id는 set_task()로 전환하므로 캐시 키에서 제외 — 환경 구조는 task마다 동일
            env_cache_key = (
                f"{self.env_id}|task_override={self.task_override_resolved_path or 'none'}"
                f"|reset_perturb={int(self.ogbench_enable_reset_perturb)}"
            )
            if (
                hasattr(self, "_cached_envs")
                and hasattr(self, "_cached_env_key")
                and self._cached_env_key == env_cache_key
            ):
                # Reuse cached single environment
                envs = self._cached_envs
                agent = getattr(self, "_cached_agent", None)
                if self.env_id in OGBENCH_ENVS:
                    self._prepare_ogbench_env(envs.envs[0])
                    envs.envs[0].set_task(self.task_id)  # task 전환 (start/goal 변경)
                self._seed_interaction_envs(envs)
                envs.reset()
            else:
                # Create single environment
                # Use higher resolution when mujoco renderer is enabled
                _render_size = 480 if (self.viz_agent_rollout and self.viz_mujoco_renderer) else 200
                if self.env_id in OGBENCH_ENVS:
                    _make_maze_env = self._get_ogbench_make_maze_env()
                    _ogbench_extra_kwargs = self._get_ogbench_maze_env_extra_kwargs()
                    if "pointmaze" in self.env_id:
                        _maze_type = self.env_id.split("-")[1]
                        env_fn = lambda: _make_maze_env(
                            "point", "maze", maze_type=_maze_type,
                            width=_render_size, height=_render_size,
                            **_ogbench_extra_kwargs,
                        )
                        use_diffused_action = True
                    elif "antmaze" in self.env_id:
                        _maze_type = self.env_id.split("-")[1]
                        env_fn = lambda: _make_maze_env(
                            "ant", "maze", maze_type=_maze_type,
                            width=_render_size, height=_render_size,
                            **_ogbench_extra_kwargs,
                        )
                        from dql.main_Antmaze import resolve_antmaze_hyperparameters
                        from dql.agents.ql_diffusion import Diffusion_QL as Agent

                        params = resolve_antmaze_hyperparameters(self.dataset)

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
                            device=self.device,
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
                        dql_results_root = os.path.join(os.getcwd(), "dql", "results")
                        dql_dir, dql_match_kind = _resolve_dql_result_dir(
                            dql_results_root,
                            self.dataset,
                            ckpt_id=200,
                        )
                        agent.load_model(dql_dir, id=200)
                        print(
                            f"[INIT] DQL agent loaded ({dql_match_kind}): {os.path.basename(dql_dir)}",
                            flush=True,
                        )
                else:
                    env_fn = lambda: gym.make(self.env_id)
                    agent = None

                # Create single DummyVecEnv and set task
                envs = DummyVecEnv([env_fn])
                if self.env_id in OGBENCH_ENVS:
                    self._prepare_ogbench_env(envs.envs[0])
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
            self._seed_interaction_envs(envs)
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

            self._current_task_start_xy = None
            self._current_task_goal_xy = None
            self._current_task_waypoints = None
            self._current_task_names = None

            reset_infos = list(getattr(envs, "reset_infos", []))

            obs = torch.from_numpy(obs).float().to(self.device)
            start = obs.detach()
            obs_normalized = (
                (obs[:, self.obs_dim_indices] - obs_mean[None]) / obs_std[None]
            ).detach()

            if self.env_id in OGBENCH_ENVS:  # OGBench
                goal = np.vstack(
                    [reset_infos[i]["goal"] for i in range(len(reset_infos))]
                )
            else:
                goal = np.concatenate([[env.env._target] for env in envs.envs])
            goal = torch.Tensor(goal).float().to(self.device)
            goal = torch.cat([goal, torch.zeros_like(goal)], -1)
            goal = goal[:, self.obs_dim_indices]  # select obs dims from raw env goal
            goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()
            self._capture_current_task_metadata(obs, goal, reset_infos)

            steps = 0
            loops = 0
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
            horizon: int = episode_len_to_plan_tokens(
                self.segment_episode_len * self.sequence_dividing_factor, self.jump, self.frame_stack
            ) * self.frame_stack
            self.current_plan_tokens = horizon // self.frame_stack
            self.global_search_num = 0
            self.global_selection_count = 0
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

            selected_plan_bundle: Optional[dict] = None
            planner_result: Optional[dict] = None
            planning_start_time: float = time.time()

            _start_pos = _bidir_start_np[0][self.pos_dim_indices].tolist()
            _goal_pos = _bidir_goal_np[0][self.pos_dim_indices].tolist()
            print(
                f"\n[MCTD] Episode start | start={[round(x,1) for x in _start_pos]} → "
                f"goal={[round(x,1) for x in _goal_pos]} | horizon={horizon} | max_loops={self.val_max_loops}",
                flush=True,
            )
            _visual_prefix_override = self._current_wandb_visual_prefix
            _viz_prefix = namespace.split("/")[-1]

            waypoint_xys = None
            if self._current_task_waypoints:
                waypoint_xys = self._current_task_waypoints[0]

            planner_result = self._run_anchor_planner(
                horizon=horizon,
                conditions=conditions,
                start=start,
                goal=goal,
                goal_normalized=goal_normalized,
                initial_sim_state=initial_sim_state,
                waypoint_xys=waypoint_xys,
                agent=agent,
                envs=envs,
                namespace=namespace,
            )

            loops = int(planner_result["loops"])
            selected_plan_bundle = planner_result["selected_plan_bundle"]

            # Single-shot plan extraction and environment execution (after MCTS search completes)
            if selected_plan_bundle is not None:
                _reorder_ms = 0.0

                plan_unnormalized = selected_plan_bundle["plan_unnormalized"]
                postprocessed_plan = selected_plan_bundle["postprocessed_plan"]
                selected_route_text = str(selected_plan_bundle.get("route_text", ""))

                # Visualization for the assembled anchor plan.
                start_numpy = start.cpu().numpy()[:, self.pos_dim_indices]
                goal_numpy = goal.cpu().numpy()[:, self.pos_dim_indices]

                # The assembled anchor bundle does not expose a single leaf target point.
                hilp_heatmap = None
                _hm_ms = 0.0
                hilp_grad_field = None
                _gf_ms = 0.0

                planning_end_time = time.time()
                self._tlog("timing.hilp_viz", {
                    "heatmap_ms": round(_hm_ms, 1),
                    "grad_field_ms": round(_gf_ms, 1),
                    "total_ms": round(_hm_ms + _gf_ms, 1),
                }, depth=0)
                _planning_loop_ms = (planning_end_time - planning_start_time) * 1000
                planning_time.append(planning_end_time - planning_start_time)

                planner_trees = list(planner_result.get("trees", [])) if planner_result else []
                tree_node_counts = [
                    max(len(tree.get_all_nodes()) - 1, 0)
                    for tree in planner_trees
                ]
                print(
                    f"[MCTD] Search complete | loops={loops} | "
                    f"parents={self.global_search_num}/{self.mctd_max_search_num} | "
                    f"tree_nodes={tree_node_counts} | "
                    f"plan_frames={plan_unnormalized.shape[0]} | "
                    f"post_frames={postprocessed_plan.shape[0]} | route={selected_route_text}",
                    flush=True,
                )

                # Visualize postprocessed plan
                _ppviz_t0 = time.time()
                _pp_plan_np = postprocessed_plan[:, :, self.pos_dim_indices].detach().cpu().numpy()  # (K, 1, pos_dim)
                _postprocessed_plan_length = self._compute_plan_polyline_length(_pp_plan_np)
                _postprocessed_plan_title = self._format_postprocessed_plan_title(
                    route_text=selected_route_text,
                    total_length=_postprocessed_plan_length,
                )
                _pp_images = make_trajectory_images(
                    self.env_id,
                    {
                        "plan": _pp_plan_np,
                        "plot_title": _postprocessed_plan_title,
                    },
                    1,
                    start_numpy.tolist(),
                    goal_numpy.tolist(),
                    self.plot_end_points,
                    waypoints=self._current_task_waypoints,
                )
                if namespace.startswith("benchmark/"):
                    _postprocessed_image_key = self._benchmark_media_key(
                        self.benchmark_postprocessed_plan_media_key
                    )
                else:
                    _postprocessed_image_key = (
                        f"{_visual_prefix_override}/postprocessed_plan"
                        if _visual_prefix_override is not None
                        else f"{_viz_prefix}_postprocessed_plan"
                    )
                for _pp_i, _pp_img in enumerate(_pp_images):
                    self.log_image(
                        _postprocessed_image_key,
                        Image.fromarray(_pp_img),
                    )
                _ppviz_ms = (time.time() - _ppviz_t0) * 1000
                self._tlog("timing.plan_postproc", {
                    "n_frames": int(postprocessed_plan.shape[0]),
                    "reorder_ms": round(_reorder_ms, 1),
                    "pre_exec_viz_ms": round(_ppviz_ms, 1),
                    "total_ms": round(_reorder_ms + _ppviz_ms, 1),
                }, depth=0)

                if self.benchmark_plan_only:
                    _execution_loop_ms = 0.0
                    self._tlog("timing.mpc_loop", {
                        "loop": loops,
                        "planning_ms": round(_planning_loop_ms, 1),
                        "execution_ms": 0.0,
                        "total_ms": round(_planning_loop_ms, 1),
                    }, depth=0)
                else:
                    # Use unified plan execution function
                    _exec_start_time = time.time()
                    trajectory_exec, reward_dict, rollout_viz = self._execute_plan_in_env(
                        plan_frame_format=postprocessed_plan,
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
                    elif terminate_on_done and reward_dict.get("done") is not None and reward_dict["done"].any():
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

            print(f"[MCTD] Episode done | loops={loops} | steps={steps} | reached={bool(reached.any())} | reward={float(episode_reward.mean()):.3f}", flush=True)

            self._safe_log_metric(f"{namespace}/task_id", float(self.task_id))
            self._safe_log_metric(f"{namespace}/planning_time", np.sum(planning_time))
            _success_rate = None
            if not self.benchmark_plan_only:
                self._safe_log_metric(f"{namespace}/episode_reward", episode_reward.mean())
                self._safe_log_metric(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
                self._safe_log_metric(f"{namespace}/first_reach", first_reach.mean())
                _success_rate = float(sum(episode_reward >= 1.0) / batch_size)
                self._safe_log_metric(f"{namespace}/success_rate", _success_rate)
            else:
                self._safe_log_metric(f"{namespace}/plan_only", 1.0)
            self._last_interact_success = _success_rate
            self._last_interact_result = {
                "task_id": int(self.task_id) if self.task_id is not None else None,
                "success": _success_rate,
                "episode_reward": (
                    None if self.benchmark_plan_only else float(episode_reward.mean())
                ),
                "episode_reward_if_stay": (
                    None if self.benchmark_plan_only else float(episode_reward_if_stay.mean())
                ),
                "first_reach": (
                    None if self.benchmark_plan_only else float(first_reach.mean())
                ),
                "planning_time": float(np.sum(planning_time)),
                "steps": int(steps),
                "interaction_seed": int(self.interaction_seed) if self.interaction_seed is not None else None,
                "wandb_visual_prefix": self._current_wandb_visual_prefix,
                "benchmark_variant_name": self.benchmark_variant_name,
                "plan_only": bool(self.benchmark_plan_only),
            }
            if selected_plan_bundle is not None:
                _pred_anchor_order = np.asarray(
                    selected_plan_bundle.get(
                        "completed_anchor_order",
                        selected_plan_bundle.get("anchor_order", []),
                    ),
                    dtype=np.int32,
                ).tolist()
                self._last_interact_result["predicted_anchor_order"] = [
                    int(idx) for idx in _pred_anchor_order
                ]
                self._last_interact_result["predicted_route_text"] = str(
                    selected_plan_bundle.get("route_text", "")
                )
                self._last_interact_result["postprocessed_plan_length"] = float(
                    _postprocessed_plan_length
                )
            if self._current_task_start_xy is not None and self._current_task_goal_xy is not None:
                self._last_interact_result["start_xy"] = self._current_task_start_xy.tolist()
                self._last_interact_result["goal_xy"] = self._current_task_goal_xy.tolist()
                self._last_interact_result["waypoints_xy"] = [
                    waypoint_set.tolist() for waypoint_set in (self._current_task_waypoints or [])
                ]
                if self._current_task_names is not None:
                    self._last_interact_result["task_names"] = list(self._current_task_names)
            if reset_infos:
                _reset_info0 = reset_infos[0] or {}
                if "active_waypoint_group_idx" in _reset_info0:
                    self._last_interact_result["active_waypoint_group_idx"] = _reset_info0.get(
                        "active_waypoint_group_idx"
                    )
                if "task_info" in _reset_info0:
                    self._last_interact_result["task_info"] = self._compact_saved_task_info(
                        _reset_info0.get("task_info")
                    )

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
                    self.env_id,
                    trajectory[:, -samples:],
                    samples,
                    start,
                    goal,
                    self.plot_end_points,
                    waypoints=self._current_task_waypoints,
                )
                for i, img in enumerate(images):
                    _agent_rollout_image_key = (
                        f"{_visual_prefix_override}/agent_rollout/sample_{i}"
                        if _visual_prefix_override is not None
                        else f"{_viz_prefix}_agent_rollout/sample_{i}"
                    )
                    self.log_image(
                        _agent_rollout_image_key,
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
                        waypoints=self._current_task_waypoints,
                    )
                    for i, video in enumerate(videos):
                        if video.shape[0] > 0:
                            _video_step = self.reserve_wandb_step()
                            _agent_rollout_video_key = (
                                f"{_visual_prefix_override}/agent_rollout/sample_{i}_video"
                                if _visual_prefix_override is not None
                                else f"{namespace}_interaction/sample_{i}_video"
                            )
                            print(
                                f"[wandb-video-debug] key={_agent_rollout_video_key} "
                                f"shape={video.shape} dtype={video.dtype} fps={self.validation_video_fps} "
                                f"step={_video_step}",
                                flush=True,
                            )
                            self.log_video(
                                _agent_rollout_video_key,
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
                    _mj_step = self.reserve_wandb_step()
                    _mujoco_video_key = (
                        f"{_visual_prefix_override}/agent_rollout/mujoco_render"
                        if _visual_prefix_override is not None
                        else f"{namespace}_interaction/mujoco_render"
                    )
                    print(
                        f"[wandb-video-debug] key={_mujoco_video_key} "
                        f"shape={_mj_video.shape} dtype={_mj_video.dtype} fps={self.validation_video_fps} "
                        f"step={_mj_step}",
                        flush=True,
                    )
                    self.log_video(
                        _mujoco_video_key,
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

    def _repad_stacked_init_from_last_frame(self, stacked_token: torch.Tensor) -> torch.Tensor:
        """Rebuild a training-style init token from the last frame of a stacked token."""
        if stacked_token.ndim != 2:
            raise ValueError(
                f"stacked_token must have shape (b, fs*c), got ndim={stacked_token.ndim}"
            )
        if stacked_token.shape[-1] != self.x_stacked_shape[0]:
            raise ValueError(
                f"stacked_token.shape[-1]={stacked_token.shape[-1]}, expected {self.x_stacked_shape[0]}"
            )

        last_frame_bundle = rearrange(
            stacked_token, "b (fs c) -> b fs c", fs=self.frame_stack
        )[:, -1].clone()
        last_frame_bundle[:, self.non_obs_bundle_indices] = 0
        repadded = self.pad_init(last_frame_bundle, is_start=True, batch_first=True)
        return rearrange(repadded, "b fs c -> b (fs c)").contiguous()

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
        self, is_achieved_plan, viz_step, values, names, plans, starts, goals, tag="mcts_plan",
        valid_frame_bounds=None,
    ):
        # plans: (t fs) b c

        batch_size = plans.shape[1]

        if isinstance(starts, torch.Tensor):
            starts = starts.detach().cpu().numpy()
        if isinstance(goals, torch.Tensor):
            goals = goals.detach().cpu().numpy()

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

        if valid_frame_bounds is not None:
            plan_obs = self._mask_plan_obs_outside_valid_frames(plan_obs, valid_frame_bounds)

        plan_images = make_trajectory_images(
            self.env_id,
            plan_obs,
            batch_size,
            starts,
            goals,
            self.plot_end_points,
            waypoints=self._current_task_waypoints,
        )
        for i in range(len(plan_images)):
            img = plan_images[i]
            self.log_image(
                # f"{tag}/{viz_step+i+1}_{names[i]}_V{values[i]}",
                f"{tag}/{names[i]}/{'achieved' if is_achieved_plan[i] else 'not_achieved'}",
                Image.fromarray(img),
            )

    def visualize_expanded_vs_value_plans(self, is_achieved_plan, names, expanded_plans, value_plans, starts, goals):
        # expanded_plans: (t fs) b c  — plans from expanded_node_plan_hists[-1]
        # value_plans:    (t fs) b c  — plans from replanned_plan_hists[-1]
        batch_size = expanded_plans.shape[1]

        if isinstance(starts, torch.Tensor):
            starts = starts.detach().cpu().numpy()
        if isinstance(goals, torch.Tensor):
            goals = goals.detach().cpu().numpy()

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
            self.env_id,
            expanded_obs,
            batch_size,
            starts,
            goals,
            self.plot_end_points,
            waypoints=self._current_task_waypoints,
        )
        value_images = make_trajectory_images(
            self.env_id,
            value_obs,
            batch_size,
            starts,
            goals,
            self.plot_end_points,
            waypoints=self._current_task_waypoints,
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-node values for two-tree search by pairing current plan with
        the target opposite-tree leaf node's plan.

        For each expanded candidate i:
          1. Slice plan_A from final_best_plans (current tree, depth-based length).
          2. Slice plan_B from target_node.plan_history (opposite tree, depth-based length).
          3. Flip plan_B and concatenate: [plan_A_sliced | flip(plan_B_sliced)].
          4. Delegate Warp/Achieved detection to calculate_values.

        Args:
            expanded_node_candidates: List of candidate dicts (each has 'parent_node').
            final_best_plans: Tensor of shape (T_total*fs, B, c) — fully denoised plan hists.
        Returns:
            values: np.ndarray shape (B,)
            infos:  np.ndarray shape (B,), dtype str
            achieved_ts: np.ndarray shape (B,)
        """
        seg_size: int = self._require_current_plan_tokens() // self.sequence_dividing_factor
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
        seg_size: int = self._require_current_plan_tokens() // self.sequence_dividing_factor
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


    # --- Embedding-distance → temporal distance conversion ---
    # HILP converges to ||phi(s)-phi(g)|| = (1-γ^{d*})/(1-γ)
    # → d*(s,g) = log(1 - emb_dist*(1-γ)) / log(γ)
    def emb_dist_to_temporal_dist(self, emb_d: np.ndarray, gamma = 0.995) -> np.ndarray:
        EPS = 1e-8
        val = 1.0 - np.asarray(emb_d, dtype=np.float64) * (1.0 - gamma)
        val = np.clip(val, EPS, 1.0 - EPS)
        return (np.log(val) / np.log(gamma)).astype(np.float64)

    def _compute_node_uncertainty(
        self,
        curr_obs: np.ndarray,  # (obs_dim,) — unnormalized current observation
        target_node: "TreeNode",
        tail_obs: np.ndarray,  # (G*K, obs_dim) — unnormalized tail observations
        gamma: float = 0.995,
        eps: float = 1e-8,
    ) -> dict:
        """Compute node uncertainty from fast-sampled tail states.

        The local entropy terms are computed in HILP embedding space using the
        radial-angular estimator, while the final multiplier T_curr is obtained
        by converting the current embedding goal-distance into temporal distance.
        """

        hilp_fn = self._get_hilp_value_fn()
        goal_obs: np.ndarray = target_node.obs  # (obs_dim,)

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

        _temporal_dist_fn = lambda emb_d: self.emb_dist_to_temporal_dist(emb_d, gamma)

        # Cluster tail embeddings by temporal distance (complete-linkage).
        # cluster_labels: (G*K,) 0-indexed; n_clusters: int
        cluster_labels = cluster_tail_by_temporal_dist(
            z_tail=Z,
            temporal_dist_fn=_temporal_dist_fn,
            max_intra_dist=self.uncertainty_max_intra_cluster_dist,
        )
        n_clusters = int(cluster_labels.max() + 1) if len(cluster_labels) > 0 else 0

        min_required_samples = {
            "cluster": 1,
            "entropy": 2,
            "variance": 3,
        }.get(self.uncertainty_mode, 1)
        is_degenerate = int(Z.shape[0]) < min_required_samples

        if is_degenerate:
            _emb_dist_curr = float(np.linalg.norm(z_goal - z_curr))
            _t_curr = float(np.asarray(_temporal_dist_fn(np.asarray(_emb_dist_curr))).item())
            result = {
                "U": 0.0,
                "T_curr": _t_curr,
                "degenerate": True,
            }
        elif self.uncertainty_mode in ("temporal_dist", "expected_root_node_dist"):
            _emb_dist_curr = float(np.linalg.norm(z_goal - z_curr))
            _t_curr = float(np.asarray(_temporal_dist_fn(np.asarray(_emb_dist_curr))).item())
            result = {
                "U": 0.0,
                "T_curr": _t_curr,
            }
        elif self.uncertainty_mode == "variance":
            result = compute_uncertainty_variance(
                z_curr=z_curr,
                z_goal=z_goal,
                z_tail=Z,
                temporal_dist_fn=_temporal_dist_fn,
                lambda_weight=self.uncertainty_lambda,
                eta_weight=self.uncertainty_eta,
            )
        elif self.uncertainty_mode == "cluster":
            _emb_dist_curr = float(np.linalg.norm(z_goal - z_curr))
            _t_curr = float(np.asarray(_temporal_dist_fn(np.asarray(_emb_dist_curr))).item())
            _u = math.log(n_clusters + 1) * _t_curr if n_clusters > 0 else 0.0
            result = {"U": _u, "T_curr": _t_curr}
        else:
            result = compute_uncertainty_from_embeddings(
                z_curr=z_curr,
                z_goal=z_goal,
                z_tail=Z,
                temporal_dist_fn=_temporal_dist_fn,
                lambda_weight=self.uncertainty_lambda,
                eta_weight=self.uncertainty_eta,
                eps=eps,
            )

        result["cluster_labels"] = cluster_labels
        result["n_clusters"] = n_clusters
        result["num_samples"] = int(Z.shape[0])
        result.setdefault("degenerate", False)
        return result

    def _compute_selection_value_for_target(
        self,
        source_node: "TreeNode",
        target_node: "TreeNode",
        t_curr: float,
    ) -> float:
        """Compute the scalar selection value for a source-target pair."""
        if self.uncertainty_mode == "expected_root_node_dist":
            return -self._compute_source_target_total_cost(
                source_node=source_node,
                target_node=target_node,
                t_curr=t_curr,
            )
        if self.uncertainty_mode == "temporal_dist":
            return -float(t_curr)
        # Legacy uncertainty modes use -U as before.
        return float("nan")

    def _compute_source_target_total_cost(
        self,
        source_node: "TreeNode",
        target_node: "TreeNode",
        t_curr: float,
    ) -> float:
        source_cum = float(getattr(source_node, "cum_temporal_dist_from_root", 0.0))
        target_cum = float(getattr(target_node, "cum_temporal_dist_from_root", 0.0))
        return (
            source_cum
            + self._transform_temporal_dist_for_total_cost(t_curr)
            + target_cum
        )

    def _run_fast_uncertainty_sampling(
        self,
        parent_nodes: List["TreeNode"],
        val_plan_last_batch: torch.Tensor,  # (plan_tokens*fs, B_nt, c)
        updated_levels: np.ndarray,         # (B_nt, plan_tokens)
        current_prefix_len_per_batch: np.ndarray,  # (B_nt,)
        seg_size: int,
        horizon: int,
        conditions: Optional[Any],
        obs_normalized: torch.Tensor,       # (B_nt, obs_dim)
        opposite_tree: Optional["MCTSTreeState"],
        obs_mean_np: np.ndarray,
        obs_std_np: np.ndarray,
        opposite_trees: Optional[List["MCTSTreeState"]] = None,
        target_nodes_override: Optional[List["TreeNode"]] = None,
    ) -> dict:
        """Run fast uncertainty sampling for a batch of non-terminal candidates.

        Builds G*K init plans per candidate from val_plan_last_batch, runs
        parallel_plan with fast_sampling_steps, and returns per-candidate results.

        Returns:
            unc_hists_per_cand: list[Tensor] of len B_nt,
                each (fast_steps+1, plan_tokens*fs, G*K, c)
            unc_noise_levels_per_cand: list[ndarray] of len B_nt,
                each (G*K, fast_steps+1, plan_tokens)
            unc_guidance_scale_per_cand: list[list[float]] of len B_nt,
                each of length G*K
        """
        B_nt = len(parent_nodes)
        K = self.fast_sampling_multiple
        G = len(self.mctd_guidance_scales)
        _plan_tokens_val = horizon // self.frame_stack
        current_prefix_len_per_batch = np.asarray(
            current_prefix_len_per_batch, dtype=int
        )
        assert current_prefix_len_per_batch.shape == (B_nt,), (
            "current_prefix_len_per_batch must have shape "
            f"({B_nt},), got {current_prefix_len_per_batch.shape}"
        )

        # Build fast noise schedule
        unc_noise_levels_nt = self._generate_tree_conditioned_schedule(
            updated_levels,
            prefix_len_per_batch=current_prefix_len_per_batch,
            num_denoising_steps_override=self.fast_sampling_steps,
        )  # (B_nt, fast_steps+1, plan_tokens)
        unc_noise_levels_rep = np.repeat(unc_noise_levels_nt, K * G, axis=0)
        # (B_nt*G*K, fast_steps+1, plan_tokens)

        unc_init_plans_rep: list = []
        unc_guidance_scale_vals: list = []
        unc_goal_list: list = []

        for _ii, _parent_node in enumerate(parent_nodes):
            _curr_prefix_len = int(current_prefix_len_per_batch[_ii])
            _plan_t_fs = val_plan_last_batch[:, _ii, :].unsqueeze(1)  # (plan_tokens*fs, 1, c)
            _plan_rearranged_unc = rearrange(
                _plan_t_fs, "(t fs) b c -> t b (fs c)", fs=self.frame_stack
            )  # (plan_tokens, 1, fs*c)
            _base_plan, _ = self._build_plan_from_leaf(
                _parent_node, _plan_tokens_val, seg_size,
                expanded_plan=_plan_rearranged_unc,
            )  # (n_tokens, 1, fs*c)
            # Re-noise from the first plan token after obs_parent, starting at the
            # current expansion boundary itself. This keeps feasibility checking and
            # uncertainty sampling aligned to the same next-segment window.
            _future_start = _curr_prefix_len + 1
            _n_future = _plan_tokens_val + 1 - _future_start

            if _curr_prefix_len > 0:
                _curr_depth = _curr_prefix_len // seg_size
                _tail_obs_unnorm = self._extract_obs_at_boundary(
                    _plan_t_fs, depth=_curr_depth, seg_size=seg_size
                )[0]  # (obs_dim,)
            else:
                _tail_obs_unnorm = _parent_node.obs
            if target_nodes_override is not None:
                _unc_target_node = target_nodes_override[_ii]
            else:
                _opp_tree = opposite_trees[_ii] if opposite_trees is not None else opposite_tree
                assert _opp_tree is not None, "opposite_tree must be provided for uncertainty sampling"
                all_opposite_nodes_unc = _opp_tree.get_all_nodes()
                _unc_target_node = self._select_dynamic_goal(
                    current_leaf_obs=_tail_obs_unnorm,
                    opposite_tree_all_nodes=all_opposite_nodes_unc,
                )
            _unc_goal_norm = torch.tensor(
                (_unc_target_node.obs - obs_mean_np) / obs_std_np,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)  # (1, obs_dim)

            for g_scale in self.mctd_guidance_scales:
                for _ in range(K):
                    _unc_plan = _base_plan.clone()
                    if _n_future > 0:
                        _unc_plan[_future_start : _plan_tokens_val + 1] = (
                            self._sample_clamped_noise(_n_future)
                        )
                    unc_init_plans_rep.append(_unc_plan)
                    unc_guidance_scale_vals.append(g_scale)
                    unc_goal_list.append(_unc_goal_norm)

        unc_guidance_scale_tensor = torch.tensor(
            unc_guidance_scale_vals, dtype=torch.float32, device=self.device
        )
        unc_obs = obs_normalized.repeat_interleave(G * K, dim=0)  # (B_nt*G*K, obs_dim)
        unc_goal = torch.cat(unc_goal_list, dim=0)  # (B_nt*G*K, obs_dim)
        unc_prefix_len_list = [
            int(current_prefix_len_per_batch[_ii])
            for _ii in range(B_nt)
            for _ in range(G * K)
        ]
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
        )  # (fast_steps+1, plan_tokens*fs, B_nt*G*K, c)

        # Split results back per candidate
        unc_hists_per_cand = []
        unc_noise_levels_per_cand = []
        unc_guidance_scale_per_cand = []
        for _ii in range(B_nt):
            _s = _ii * G * K
            _e = _s + G * K
            unc_hists_per_cand.append(unc_batch_plan_hists[:, :, _s:_e, :])
            unc_noise_levels_per_cand.append(unc_noise_levels_rep[_s:_e])  # (G*K, fast_steps+1, plan_tokens)
            unc_guidance_scale_per_cand.append(unc_guidance_scale_vals[_s:_e])

        return {
            "unc_hists_per_cand": unc_hists_per_cand,
            "unc_noise_levels_per_cand": unc_noise_levels_per_cand,
            "unc_guidance_scale_per_cand": unc_guidance_scale_per_cand,
        }

    def _compute_uncertainty_and_clusters(
        self,
        unc_hists_per_cand: List[torch.Tensor],     # B_nt × (fast_steps+1, plan_tokens*fs, G*K, c)
        curr_plan_last_batch: torch.Tensor,          # (plan_tokens*fs, B_nt, c)
        parent_nodes: List["TreeNode"],              # B_nt
        node_depths: List[int],                      # B_nt candidate depths
        seg_size: int,
        unc_noise_levels_per_cand: List[np.ndarray], # B_nt × (G*K, fast_steps+1, plan_tokens)
        unc_guidance_scale_per_cand: List[list],     # B_nt × G*K floats
        target_nodes: List["TreeNode"],              # B_nt
    ) -> dict:
        """Compute uncertainty values and build cluster_subplans for each candidate.

        Returns:
            values: list[float] of len B_nt, each entry is -U
            cluster_subplans: list[list[dict]|None] of len B_nt
            unc_results: list[dict] of len B_nt (raw _compute_node_uncertainty output)
        """
        B_nt = len(unc_hists_per_cand)
        values: list = []
        cluster_subplans: list = []
        unc_results_out: list = []
        filtered_unc_hists: list = []
        use_kde_maximin_selection = (
            self.use_cluster_subplan_as_expansion
            and self.use_kde_maximin_for_selecting_subplan_in_cluster
        )

        for _ii in range(B_nt):
            unc_hists_i = unc_hists_per_cand[_ii]  # (fast_steps+1, plan_tokens*fs, G*K, c)
            curr_depth_i = node_depths[_ii]

            curr_obs_i = self._extract_obs_at_boundary(
                curr_plan_last_batch[:, _ii, :].unsqueeze(1),  # (plan_tokens*fs, 1, c)
                depth=curr_depth_i,
                seg_size=seg_size,
            )[0]  # (obs_dim,)

            feasible_mask_i = self._check_plan_batch_feasibility(
                plan_hists=unc_hists_i,
                root_obs_list=[self._get_root_obs(parent_nodes[_ii])] * unc_hists_i.shape[2],
                progress_obs_list=[curr_obs_i] * unc_hists_i.shape[2],
                prefix_len_frames_list=[
                    self._get_prefix_len_frames_from_depth(curr_depth_i, seg_size)
                ] * unc_hists_i.shape[2],
                subplan_tail_depths=[curr_depth_i + 1] * unc_hists_i.shape[2],
                seg_size=seg_size,
            )
            feasible_sample_indices = np.asarray(
                np.where(np.asarray(feasible_mask_i, dtype=bool))[0],
                dtype=int,
            )

            if feasible_sample_indices.size == 0:
                values.append(-np.inf)
                cluster_subplans.append([])
                unc_results_out.append(
                    {
                        "U": np.inf,
                        "cluster_labels": np.array([], dtype=int),
                        "n_clusters": 0,
                        "degenerate": True,
                        "num_samples": 0,
                        "num_raw_samples": int(unc_hists_i.shape[2]),
                        "num_feasible_samples": 0,
                        "feasible_sample_indices": feasible_sample_indices,
                        "sample_guidance_scales": [],
                        "sample_feasible_scores": np.array([], dtype=np.float32),
                        "cluster_representative_indices": np.array([], dtype=int),
                    }
                )
                filtered_unc_hists.append(None)
                continue

            filtered_unc_hists_i = unc_hists_i[:, :, feasible_sample_indices, :]
            filtered_unc_hists.append(filtered_unc_hists_i)
            filtered_noise_levels_i = unc_noise_levels_per_cand[_ii][feasible_sample_indices]
            filtered_guidance_scales_i = [
                float(unc_guidance_scale_per_cand[_ii][_j])
                for _j in feasible_sample_indices
            ]
            sample_feasible_scores_i = None
            if use_kde_maximin_selection:
                sample_feasible_scores_i = self._compute_local_subplan_kde_maximin_scores(
                    plan_batch=filtered_unc_hists_i[-1],
                    parent_depth=curr_depth_i,
                    seg_size=seg_size,
                )

            tail_obs_i = self._extract_obs_at_boundary(
                filtered_unc_hists_i[-1],  # (plan_tokens*fs, N_survive, c)
                depth=curr_depth_i + 1,
                seg_size=seg_size,
            )  # (N_survive, obs_dim)

            unc_result = self._compute_node_uncertainty(
                curr_obs=curr_obs_i,
                target_node=target_nodes[_ii],
                tail_obs=tail_obs_i,
            )
            unc_result["num_raw_samples"] = int(unc_hists_i.shape[2])
            unc_result["num_feasible_samples"] = int(feasible_sample_indices.size)
            unc_result["feasible_sample_indices"] = feasible_sample_indices
            unc_result["sample_guidance_scales"] = filtered_guidance_scales_i
            unc_result["sample_feasible_scores"] = (
                sample_feasible_scores_i.copy()
                if sample_feasible_scores_i is not None
                else np.array([], dtype=np.float32)
            )
            selection_value = None
            if self.uncertainty_mode in ("temporal_dist", "expected_root_node_dist"):
                selection_value = self._compute_selection_value_for_target(
                    source_node=parent_nodes[_ii],
                    target_node=target_nodes[_ii],
                    t_curr=float(unc_result["T_curr"]),
                )
                values.append(float(selection_value))
            else:
                values.append(-unc_result["U"])
            unc_result["selection_value"] = (
                float(selection_value) if selection_value is not None else -float(unc_result["U"])
            )
            unc_results_out.append(unc_result)

            # Build cluster_subplans
            if self.use_cluster_subplan_as_expansion:
                _cl_labels = unc_result["cluster_labels"]
                _n_cl = unc_result["n_clusters"]
                _csp_list = []
                _cluster_rep_indices = []
                for _c in range(_n_cl):
                    _cluster_member_indices = np.where(_cl_labels == _c)[0]
                    if sample_feasible_scores_i is not None:
                        _best_local_idx = int(
                            np.argmax(sample_feasible_scores_i[_cluster_member_indices])
                        )
                        _rep_j = int(_cluster_member_indices[_best_local_idx])
                    else:
                        _rep_j = int(_cluster_member_indices[0])
                    _cluster_rep_indices.append(_rep_j)
                    _ph_j = filtered_unc_hists_i[:, :, _rep_j, :]  # (fast_steps+1, plan_tokens*fs, c)
                    _gs_j = float(filtered_guidance_scales_i[_rep_j])
                    _lvl_j = filtered_noise_levels_i[_rep_j : _rep_j + 1, -1, :]
                    # (1, plan_tokens)
                    _csp_list.append({
                        "plan_hist":      _ph_j,
                        "guidance_scale": _gs_j,
                        "current_levels": np.asarray(_lvl_j, dtype=np.int64),
                        "feasible_score": (
                            float(sample_feasible_scores_i[_rep_j])
                            if sample_feasible_scores_i is not None
                            else None
                        ),
                    })
                unc_result["cluster_representative_indices"] = np.asarray(
                    _cluster_rep_indices, dtype=int
                )
                cluster_subplans.append(_csp_list)
            else:
                unc_result["cluster_representative_indices"] = np.array([], dtype=int)
                cluster_subplans.append(None)

        return {
            "values": values,
            "cluster_subplans": cluster_subplans,
            "unc_results": unc_results_out,
            "filtered_unc_hists": filtered_unc_hists,
        }

    def _init_mcts_tree(
        self,
        horizon: int,
        tag: str,
        is_tree1: bool = True,
        root_obs: Optional[np.ndarray] = None,
        root_sim_state: Optional[dict] = None,
        anchor_idx: int = 0,
        anchor_name: str = "S",
        anchor_xy: Optional[np.ndarray] = None,
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
        skip_level_steps: int = self.mctd_skip_level_steps

        assert plan_tokens <= self.n_tokens - 1, (
            f"Plan tokens must be <= {self.n_tokens - 1}, but got {plan_tokens}"
        )
        terminal_depth: int = self.sequence_dividing_factor
        noise_level: Optional[np.ndarray] = None  # dynamic schedule is generated per expansion round

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
        root_node.anchor_idx = int(anchor_idx)
        root_node.anchor_name = str(anchor_name)
        root_node.set_value(0)  # Initialize the value of the root node

        pbar = None
        if not self.use_uncertainty_as_value:
            pbar = tqdm(
                total=self.mctd_max_search_num,
                desc=f"MCTS ({tag})",
                leave=True,
                dynamic_ncols=True,
            )

        return MCTSTreeState(
            root_node=root_node,
            terminal_depth=terminal_depth,
            noise_level=noise_level,
            children_node_guidance_scales=children_node_guidance_scales,
            skip_level_steps=skip_level_steps,
            tag=tag,
            is_tree1=is_tree1,
            pbar=pbar,
            tree_root_obs=root_obs,
            anchor_idx=int(anchor_idx),
            anchor_name=str(anchor_name),
            anchor_xy=None if anchor_xy is None else np.asarray(anchor_xy, dtype=np.float32).reshape(-1),
        )

    def _init_multi_tree_planner_state(
        self,
        horizon: int,
        start_obs: np.ndarray,
        goal_obs: np.ndarray,
        initial_sim_state: dict,
        waypoint_xys: Optional[np.ndarray],
        fixed_solver_result: Optional[dict[str, Any]] = None,
    ) -> dict:
        anchor_specs: list[dict] = [
            {
                "anchor_idx": 0,
                "anchor_name": "S",
                "tag": "multi_tree_start",
                "root_obs": np.asarray(start_obs, dtype=np.float32).copy(),
                "root_sim_state": initial_sim_state,
                "is_tree1": True,
                "anchor_xy": np.asarray(start_obs, dtype=np.float32)[self.pos_dim_indices],
            }
        ]

        waypoint_xys = (
            np.zeros((0, 2), dtype=np.float32)
            if waypoint_xys is None
            else np.asarray(waypoint_xys, dtype=np.float32).reshape(-1, 2)
        )
        for wp_idx, waypoint_xy in enumerate(waypoint_xys, start=1):
            waypoint_sim_state = {
                "qpos": initial_sim_state["qpos"].copy(),
                "qvel": np.zeros_like(initial_sim_state["qvel"]),
            }
            waypoint_sim_state["qpos"][:2] = waypoint_xy
            waypoint_obs = np.concatenate(
                [waypoint_sim_state["qpos"], waypoint_sim_state["qvel"]]
            )[self.obs_dim_indices].astype(np.float32)
            anchor_specs.append(
                {
                    "anchor_idx": wp_idx,
                    "anchor_name": f"W{wp_idx}",
                    "tag": f"multi_tree_wp{wp_idx}",
                    "root_obs": waypoint_obs,
                    "root_sim_state": waypoint_sim_state,
                    "is_tree1": True,
                    "anchor_xy": np.asarray(waypoint_xy, dtype=np.float32).reshape(-1),
                }
            )

        goal_anchor_idx = len(anchor_specs)
        goal_sim_state = {
            "qpos": initial_sim_state["qpos"].copy(),
            "qvel": np.zeros_like(initial_sim_state["qvel"]),
        }
        goal_sim_state["qpos"][:2] = np.asarray(goal_obs, dtype=np.float32)[self.pos_dim_indices]
        anchor_specs.append(
            {
                "anchor_idx": goal_anchor_idx,
                "anchor_name": "G",
                "tag": "multi_tree_goal",
                "root_obs": np.asarray(goal_obs, dtype=np.float32).copy(),
                "root_sim_state": goal_sim_state,
                "is_tree1": True,
                "anchor_xy": np.asarray(goal_obs, dtype=np.float32)[self.pos_dim_indices],
            }
        )

        trees: list[MCTSTreeState] = []
        for spec in anchor_specs:
            trees.append(
                self._init_mcts_tree(
                    horizon=horizon,
                    tag=spec["tag"],
                    is_tree1=bool(spec["is_tree1"]),
                    root_obs=spec["root_obs"],
                    root_sim_state=spec["root_sim_state"],
                    anchor_idx=int(spec["anchor_idx"]),
                    anchor_name=str(spec["anchor_name"]),
                    anchor_xy=spec["anchor_xy"],
                )
            )

        n_anchors = len(trees)
        root_cost = np.full((n_anchors, n_anchors), np.inf, dtype=np.float32)
        np.fill_diagonal(root_cost, 0.0)
        for i in range(n_anchors):
            for j in range(i + 1, n_anchors):
                td_ij = float(
                    self._compute_raw_state_temporal_dist_np(
                        trees[i].root_node.obs[None].astype(np.float32),
                        trees[j].root_node.obs[None].astype(np.float32),
                    )[0]
                )
                total_cost_ij = self._compute_source_target_total_cost(
                    source_node=trees[i].root_node,
                    target_node=trees[j].root_node,
                    t_curr=td_ij,
                )
                root_cost[i, j] = total_cost_ij
                root_cost[j, i] = total_cost_ij

        planner_state = {
            "trees": trees,
            "anchor_specs": anchor_specs,
            "accepted_pair_edges": {},
            "accepted_neighbors": {int(i): set() for i in range(n_anchors)},
            "best_direct_bridge_raw": root_cost.copy(),
            "best_direct_bridge_info": {},
            "effective_pair_cost": root_cost.copy(),
            "fixed_tentative_solver_result": None if fixed_solver_result is None else dict(fixed_solver_result),
            "tentative_solver_result": {},
        }
        self._compute_tentative_hamiltonian_solution(planner_state)
        return planner_state

    def _build_pairwise_anchor_selection_state(
        self,
        tree1: "MCTSTreeState",
        tree2: "MCTSTreeState",
    ) -> dict:
        trees = [tree1, tree2]
        root_cost = np.full((2, 2), np.inf, dtype=np.float32)
        np.fill_diagonal(root_cost, 0.0)
        td_01 = float(
            self._compute_raw_state_temporal_dist_np(
                tree1.root_node.obs[None].astype(np.float32),
                tree2.root_node.obs[None].astype(np.float32),
            )[0]
        )
        total_cost_01 = self._compute_source_target_total_cost(
            source_node=tree1.root_node,
            target_node=tree2.root_node,
            t_curr=td_01,
        )
        root_cost[0, 1] = total_cost_01
        root_cost[1, 0] = total_cost_01

        fixed_solver_result = None
        if self.multi_tree_route_mode == "fixed_temporal":
            fixed_solver_result = {
                "feasible": True,
                "anchor_order": np.asarray([0, 1], dtype=np.int32),
                "route_text": "S -> G",
                "total_cost": float(total_cost_01),
            }

        planner_state = {
            "trees": trees,
            "accepted_pair_edges": {},
            "accepted_neighbors": {0: set(), 1: set()},
            "best_direct_bridge_raw": root_cost.copy(),
            "best_direct_bridge_info": {},
            "effective_pair_cost": root_cost.copy(),
            "fixed_tentative_solver_result": (
                None if fixed_solver_result is None else dict(fixed_solver_result)
            ),
            "tentative_solver_result": {},
        }
        self._compute_tentative_hamiltonian_solution(planner_state)
        return planner_state

    def _ensure_uncertainty_roots_initialized_multi(
        self,
        planner_state: dict,
        horizon: int,
        conditions: Optional[Any],
    ) -> dict[str, dict]:
        root_uncertainty_infos: dict[str, dict] = {}
        for tree in planner_state["trees"]:
            if self._is_anchor_satisfied(planner_state, tree.anchor_idx):
                continue
            if tree.root_node.cluster_subplans is not None:
                _existing = getattr(tree.root_node, "_root_uncertainty_vinfo", None)
                if _existing is not None:
                    root_uncertainty_infos[tree.tag] = _existing
                continue
            target_info = self._select_multi_tree_target_node(
                planner_state=planner_state,
                source_tree=tree,
                current_leaf_obs=tree.root_node.obs,
                use_segment_metric=False,
            )
            target_node = None if target_info is None else target_info["target_node"]
            if target_node is None:
                continue
            root_uncertainty_infos[tree.tag] = self._init_root_node_uncertainty(
                root_node=tree.root_node,
                opposite_tree=None,
                horizon=horizon,
                conditions=conditions,
                target_node_override=target_node,
            )
        return root_uncertainty_infos

    def _select_multi_tree_expansion_parents(self, planner_state: dict) -> List[dict]:
        remaining_budget = max(self.mctd_max_search_num - self.global_search_num, 0)
        selection_budget = min(self._parent_selection_budget(), remaining_budget)
        if selection_budget <= 0:
            return []

        candidate_pool: list[dict] = []
        for tree in planner_state["trees"]:
            if self._is_anchor_satisfied(planner_state, tree.anchor_idx):
                continue
            for node in tree.get_all_nodes():
                if node.obs is None or node.is_terminal():
                    continue
                if not node.is_expandable(
                    consider_virtually_visited=(not self.parallel_multiple_visits)
                ):
                    continue
                target_info = self._select_multi_tree_target_node(
                    planner_state=planner_state,
                    source_tree=tree,
                    current_leaf_obs=node.obs,
                    use_segment_metric=False,
                )
                target_node = None if target_info is None else target_info["target_node"]
                if target_node is None or target_node.obs is None:
                    continue
                target_tree = next(
                    (
                        other_tree for other_tree in planner_state["trees"]
                        if self._get_root_node(target_node) is other_tree.root_node
                    ),
                    None,
                )
                if target_tree is None:
                    continue
                target_td = float(
                    self._compute_raw_state_temporal_dist_np(
                        node.obs[None].astype(np.float32),
                        target_node.obs[None].astype(np.float32),
                    )[0]
                )
                candidate_pool.append(
                    {
                        "node": node,
                        "tree": tree,
                        "opposite_tree": target_tree,
                        "target_node": target_node,
                        "value": self._compute_selection_value_for_target(
                            source_node=node,
                            target_node=target_node,
                            t_curr=target_td,
                        ),
                    }
                )

        candidate_pool.sort(
            key=lambda item: (
                -float(item["value"]),
                -int(item["node"].depth),
                item["tree"].anchor_idx,
                item["node"].name,
            )
        )
        selected_parent_infos = candidate_pool[:selection_budget]
        for parent_info in selected_parent_infos:
            self.global_selection_count += 1
            selection_count = self.global_selection_count
            parent_info["selection_count"] = selection_count
            parent_info["node"].last_selection_count = selection_count
        return selected_parent_infos

    def _update_multi_tree_direct_bridges_from_expansions(
        self,
        planner_state: dict,
        tree_batches: dict[str, dict],
    ) -> None:
        for tree_batch in tree_batches.values():
            source_tree: MCTSTreeState = tree_batch["tree"]
            source_anchor_idx = int(source_tree.anchor_idx)
            for info in tree_batch["expanded_node_infos"].values():
                source_node = info.get("node")
                if source_node is None or source_node.obs is None:
                    continue
                for target_tree in planner_state["trees"]:
                    target_anchor_idx = int(target_tree.anchor_idx)
                    if target_anchor_idx == source_anchor_idx:
                        continue
                    target_nodes = [
                        node for node in target_tree.get_all_nodes()
                        if node.obs is not None
                    ]
                    if not target_nodes:
                        continue
                    target_obs = np.stack([node.obs for node in target_nodes], axis=0)
                    src_obs_rep = np.repeat(
                        source_node.obs[None].astype(np.float32),
                        len(target_nodes),
                        axis=0,
                    )
                    tds = self._compute_raw_state_temporal_dist_np(src_obs_rep, target_obs)
                    best_local_idx = int(np.argmin(tds))
                    target_node = target_nodes[best_local_idx]
                    bridge_cost = self._compute_source_target_total_cost(
                        source_node=source_node,
                        target_node=target_node,
                        t_curr=float(tds[best_local_idx]),
                    )
                    pair_key = self._anchor_pair_key(source_anchor_idx, target_anchor_idx)
                    if bridge_cost >= float(planner_state["best_direct_bridge_raw"][pair_key]):
                        continue
                    planner_state["best_direct_bridge_raw"][pair_key] = bridge_cost
                    planner_state["best_direct_bridge_raw"][pair_key[::-1]] = bridge_cost
                    planner_state["effective_pair_cost"][pair_key] = bridge_cost
                    planner_state["effective_pair_cost"][pair_key[::-1]] = bridge_cost
                    planner_state["best_direct_bridge_info"][pair_key] = {
                        "pair_key": pair_key,
                        "source_anchor_idx": source_anchor_idx,
                        "target_anchor_idx": target_anchor_idx,
                        "source_node": source_node,
                        "target_node": target_node,
                        "bridge_cost": float(bridge_cost),
                        "t_curr": float(tds[best_local_idx]),
                    }

    def _select_multi_tree_round_meetings(
        self,
        planner_state: dict,
        expanded_node_infos: dict[str, dict],
    ) -> dict[Tuple[int, int], dict]:
        accepted_candidates: dict[Tuple[int, int], dict] = {}
        plan_tokens = self._require_current_plan_tokens()
        forced_pairs = {
            self._anchor_pair_key(int(order[i]), int(order[i + 1]))
            for order in [planner_state["tentative_solver_result"].get("anchor_order", np.zeros((0,), dtype=np.int32))]
            for i in range(max(len(order) - 1, 0))
        }
        for info in expanded_node_infos.values():
            node = info.get("node")
            selected_tree = info.get("selected_tree")
            target_tree = info.get("meeting_target_tree")
            meeting_target_node = info.get("meeting_target_node")
            meeting_target_td = info.get("meeting_target_td")
            if node is None or selected_tree is None or target_tree is None:
                continue
            if meeting_target_node is None or meeting_target_node.obs is None:
                continue
            source_anchor_idx = int(selected_tree.anchor_idx)
            target_anchor_idx = int(target_tree.anchor_idx)
            pair_key = self._anchor_pair_key(source_anchor_idx, target_anchor_idx)
            if pair_key not in forced_pairs:
                continue
            if target_anchor_idx in planner_state["accepted_neighbors"][source_anchor_idx]:
                continue
            if self._is_anchor_satisfied(planner_state, source_anchor_idx):
                continue
            if self._is_anchor_satisfied(planner_state, target_anchor_idx):
                continue
            gap = self._compute_plan_gap_to_target(
                node,
                meeting_target_node,
                plan_tokens,
            )
            if gap is None or not np.isfinite(gap) or gap >= self.meeting_delta:
                continue
            t_curr = float(
                self._compute_raw_state_temporal_dist_np(
                    node.obs[None].astype(np.float32),
                    meeting_target_node.obs[None].astype(np.float32),
                )[0]
            )
            bridge_cost = self._compute_source_target_total_cost(
                source_node=node,
                target_node=meeting_target_node,
                t_curr=t_curr,
            )
            candidate = {
                "pair_key": pair_key,
                "source_anchor_idx": source_anchor_idx,
                "target_anchor_idx": target_anchor_idx,
                "source_node": node,
                "target_node": meeting_target_node,
                "selection_count": info.get(
                    "selection_count",
                    getattr(node, "last_selection_count", None),
                ),
                "bridge_cost": float(bridge_cost),
                "gap": float(gap),
                "meeting_target_td": (
                    float(meeting_target_td)
                    if meeting_target_td is not None
                    else float("inf")
                ),
                "selected_tree": selected_tree,
                "target_tree": target_tree,
            }
            incumbent = accepted_candidates.get(pair_key)
            if incumbent is None or (
                candidate["bridge_cost"],
                candidate["meeting_target_td"],
                candidate["gap"],
                candidate["source_node"].name,
            ) < (
                incumbent["bridge_cost"],
                incumbent["meeting_target_td"],
                incumbent["gap"],
                incumbent["source_node"].name,
            ):
                accepted_candidates[pair_key] = candidate
        return accepted_candidates

    def _log_anchor_meeting_accept(
        self,
        candidate: dict,
        loops: int,
        n_anchors: int,
    ) -> None:
        pair_text = self._anchor_pair_text_from_indices(
            candidate["pair_key"],
            n_anchors=int(n_anchors),
        )
        selection_count = candidate.get("selection_count")
        if selection_count is None:
            selection_text = "unknown"
        else:
            selection_text = f"{int(selection_count):03d}"
        print(
            "[meeting-accept] "
            f"round={int(loops):03d} "
            f"selection_count={selection_text} "
            f"pair={pair_text} "
            f"source_path={candidate['source_node'].name} "
            f"meeting_target_path={candidate['target_node'].name} "
            f"gap={float(candidate['gap']):.6f} "
            f"meeting_target_td={float(candidate['meeting_target_td']):.6f} "
            f"bridge_cost={float(candidate['bridge_cost']):.6f}",
            flush=True,
        )

    def _accept_multi_tree_round_meetings(
        self,
        planner_state: dict,
        accepted_candidates: dict[Tuple[int, int], dict],
        loops: int,
    ) -> None:
        for pair_key, candidate in accepted_candidates.items():
            if pair_key in planner_state["accepted_pair_edges"]:
                continue
            a, b = pair_key
            if self._is_anchor_satisfied(planner_state, a) or self._is_anchor_satisfied(planner_state, b):
                continue
            planner_state["accepted_pair_edges"][pair_key] = {
                **candidate,
                "accepted_round": int(loops),
            }
            planner_state["accepted_neighbors"][a].add(b)
            planner_state["accepted_neighbors"][b].add(a)
            self._log_anchor_meeting_accept(
                candidate,
                loops,
                n_anchors=len(planner_state["trees"]),
            )

    def _all_multi_tree_anchors_satisfied(self, planner_state: dict) -> bool:
        for anchor_idx in range(len(planner_state["trees"])):
            if not self._is_anchor_satisfied(planner_state, anchor_idx):
                return False
        return True

    def _get_connection_info_for_pair(
        self,
        planner_state: dict,
        pair_key: Tuple[int, int],
    ) -> Optional[dict]:
        if pair_key in planner_state["accepted_pair_edges"]:
            return planner_state["accepted_pair_edges"][pair_key]
        return planner_state["best_direct_bridge_info"].get(pair_key)

    def _find_restricted_anchor_walk(
        self,
        planner_state: dict,
        source_anchor_idx: int,
        target_anchor_idx: int,
        allowed_anchor_indices: set[int],
    ) -> Optional[list[int]]:
        source_anchor_idx = int(source_anchor_idx)
        target_anchor_idx = int(target_anchor_idx)
        allowed_anchor_indices = {int(idx) for idx in allowed_anchor_indices}
        allowed_anchor_indices.add(source_anchor_idx)
        allowed_anchor_indices.add(target_anchor_idx)

        n_anchors = len(planner_state["trees"])
        dist = {idx: np.inf for idx in allowed_anchor_indices}
        prev = {idx: None for idx in allowed_anchor_indices}
        visited: set[int] = set()
        dist[source_anchor_idx] = 0.0

        while len(visited) < len(allowed_anchor_indices):
            cur = min(
                (idx for idx in allowed_anchor_indices if idx not in visited),
                key=lambda idx: dist[idx],
                default=None,
            )
            if cur is None or not np.isfinite(dist[cur]):
                break
            if cur == target_anchor_idx:
                break
            visited.add(cur)
            for nxt in allowed_anchor_indices:
                if nxt == cur or nxt in visited:
                    continue
                pair_key = self._anchor_pair_key(cur, nxt)
                conn_info = self._get_connection_info_for_pair(planner_state, pair_key)
                if conn_info is None:
                    continue
                edge_cost = float(conn_info["bridge_cost"])
                cand = float(dist[cur]) + edge_cost
                if cand < dist[nxt]:
                    dist[nxt] = cand
                    prev[nxt] = cur

        if not np.isfinite(dist[target_anchor_idx]):
            return None

        walk = [target_anchor_idx]
        cur = target_anchor_idx
        while cur != source_anchor_idx:
            cur = prev[cur]
            if cur is None:
                return None
            walk.append(cur)
        walk.reverse()
        return walk

    def _repair_missing_pair_connection(
        self,
        planner_state: dict,
        source_anchor_idx: int,
        target_anchor_idx: int,
    ) -> Optional[dict]:
        source_anchor_idx = int(source_anchor_idx)
        target_anchor_idx = int(target_anchor_idx)
        pair_key = self._anchor_pair_key(source_anchor_idx, target_anchor_idx)
        existing = self._get_connection_info_for_pair(planner_state, pair_key)
        if existing is not None:
            return existing

        source_tree = planner_state["trees"][source_anchor_idx]
        target_tree = planner_state["trees"][target_anchor_idx]
        source_nodes = [
            node
            for node in source_tree.get_all_nodes()
            if node.obs is not None
            and len(node.plan_history) > 0
            and len(node.plan_history[-1]) > 0
        ]
        target_nodes = [
            node
            for node in target_tree.get_all_nodes()
            if node.obs is not None
        ]
        if not source_nodes or not target_nodes:
            return None

        target_obs = np.stack([node.obs for node in target_nodes], axis=0).astype(np.float32)
        best_candidate: Optional[dict] = None
        for source_node in source_nodes:
            src_obs_rep = np.repeat(
                source_node.obs[None].astype(np.float32),
                len(target_nodes),
                axis=0,
            )
            tds = self._compute_raw_state_temporal_dist_np(src_obs_rep, target_obs)
            best_local_idx = int(np.argmin(tds))
            target_node = target_nodes[best_local_idx]
            t_curr = float(tds[best_local_idx])
            bridge_cost = self._compute_source_target_total_cost(
                source_node=source_node,
                target_node=target_node,
                t_curr=t_curr,
            )
            candidate = {
                "pair_key": pair_key,
                "source_anchor_idx": source_anchor_idx,
                "target_anchor_idx": target_anchor_idx,
                "source_node": source_node,
                "target_node": target_node,
                "bridge_cost": float(bridge_cost),
                "t_curr": t_curr,
                "repaired": True,
            }
            if best_candidate is None or (
                candidate["bridge_cost"],
                candidate["t_curr"],
                candidate["source_node"].name,
                candidate["target_node"].name,
            ) < (
                best_candidate["bridge_cost"],
                best_candidate["t_curr"],
                best_candidate["source_node"].name,
                best_candidate["target_node"].name,
            ):
                best_candidate = candidate

        if best_candidate is None:
            return None

        planner_state["best_direct_bridge_raw"][pair_key] = float(best_candidate["bridge_cost"])
        planner_state["best_direct_bridge_raw"][pair_key[::-1]] = float(best_candidate["bridge_cost"])
        planner_state["effective_pair_cost"][pair_key] = float(best_candidate["bridge_cost"])
        planner_state["effective_pair_cost"][pair_key[::-1]] = float(best_candidate["bridge_cost"])
        planner_state["best_direct_bridge_info"][pair_key] = best_candidate
        print(
            "[edge-repair] "
            f"pair={self._anchor_pair_text_from_indices(pair_key, len(planner_state['trees']))} "
            f"source_path={best_candidate['source_node'].name} "
            f"target_path={best_candidate['target_node'].name} "
            f"bridge_cost={float(best_candidate['bridge_cost']):.6f}",
            flush=True,
        )
        return best_candidate

    def _materialize_connection_segment(
        self,
        planner_state: dict,
        connection_info: dict,
        desired_source_idx: int,
        desired_target_idx: int,
    ) -> torch.Tensor:
        plan_tokens = self._require_current_plan_tokens()
        stored_source_idx = int(connection_info["source_anchor_idx"])
        stored_target_idx = int(connection_info["target_anchor_idx"])
        target_root_obs = planner_state["trees"][stored_target_idx].root_node.obs
        target_root_obs_norm = self._normalize_obs_np(target_root_obs)
        edge_bundle = self._build_connection_plan_bundle(
            source_node=connection_info["source_node"],
            target_node=connection_info["target_node"],
            plan_tokens=plan_tokens,
            append_target_obs_normalized=target_root_obs_norm,
        )
        segment = edge_bundle["postprocessed_plan"]
        if stored_source_idx == int(desired_source_idx) and stored_target_idx == int(desired_target_idx):
            return segment
        if stored_source_idx == int(desired_target_idx) and stored_target_idx == int(desired_source_idx):
            return torch.flip(segment, [0])
        raise ValueError(
            f"Connection orientation mismatch: stored=({stored_source_idx},{stored_target_idx}) "
            f"desired=({desired_source_idx},{desired_target_idx})"
        )

    def _assemble_multi_tree_plan_bundle(self, planner_state: dict) -> Optional[dict]:
        solver_result = planner_state.get("tentative_solver_result") or self._compute_tentative_hamiltonian_solution(planner_state)
        if not solver_result.get("feasible", False):
            return None

        anchor_order = np.asarray(solver_result["anchor_order"], dtype=np.int32)
        if anchor_order.size < 2:
            return None

        n_anchors = len(planner_state["trees"])
        break_reason: Optional[str] = None

        def _pair_text(pair_key: tuple[int, int]) -> str:
            return (
                f"{anchor_short_label(int(pair_key[0]), n_anchors)}-"
                f"{anchor_short_label(int(pair_key[1]), n_anchors)}"
            )

        def _route_text(anchor_indices: np.ndarray | list[int]) -> str:
            return " -> ".join(
                anchor_short_label(int(idx), n_anchors)
                for idx in list(anchor_indices)
            )

        concat_segments: list[torch.Tensor] = []
        completed_anchor_order = [int(anchor_order[0])]

        for edge_idx in range(len(anchor_order) - 1):
            src_idx = int(anchor_order[edge_idx])
            dst_idx = int(anchor_order[edge_idx + 1])
            pair_key = self._anchor_pair_key(src_idx, dst_idx)
            connection_sequence: list[dict] = []

            direct_conn = self._get_connection_info_for_pair(planner_state, pair_key)
            if direct_conn is not None:
                connection_sequence = [direct_conn]
            else:
                allowed_anchor_indices = set(int(idx) for idx in anchor_order[: edge_idx + 2])
                walk = self._find_restricted_anchor_walk(
                    planner_state=planner_state,
                    source_anchor_idx=src_idx,
                    target_anchor_idx=dst_idx,
                    allowed_anchor_indices=allowed_anchor_indices,
                )
                if walk is None or len(walk) < 2:
                    repaired_conn = self._repair_missing_pair_connection(
                        planner_state=planner_state,
                        source_anchor_idx=src_idx,
                        target_anchor_idx=dst_idx,
                    )
                    if repaired_conn is None:
                        break_reason = (
                            f"missing_walk edge={anchor_short_label(src_idx, n_anchors)}->"
                            f"{anchor_short_label(dst_idx, n_anchors)} "
                            f"allowed={_route_text(sorted(allowed_anchor_indices))}"
                        )
                        break
                    connection_sequence = [repaired_conn]
                else:
                    walk_pairs = [
                        self._anchor_pair_key(int(walk[i]), int(walk[i + 1]))
                        for i in range(len(walk) - 1)
                    ]
                    connection_sequence = []
                    missing_conn = False
                    for walk_idx, walk_pair in enumerate(walk_pairs):
                        conn = self._get_connection_info_for_pair(planner_state, walk_pair)
                        if conn is None:
                            conn = self._repair_missing_pair_connection(
                                planner_state=planner_state,
                                source_anchor_idx=int(walk[walk_idx]),
                                target_anchor_idx=int(walk[walk_idx + 1]),
                            )
                        if conn is None:
                            missing_conn = True
                            break_reason = (
                                f"walk_missing_conn edge={anchor_short_label(src_idx, n_anchors)}->"
                                f"{anchor_short_label(dst_idx, n_anchors)} "
                                f"walk={_route_text(walk)} missing={_pair_text(walk_pair)}"
                            )
                            break
                        connection_sequence.append(conn)
                    if missing_conn:
                        break

            current_src = src_idx
            for conn in connection_sequence:
                stored_pair = {
                    int(conn["source_anchor_idx"]),
                    int(conn["target_anchor_idx"]),
                }
                if int(current_src) not in stored_pair:
                    return None
                current_dst = (
                    int(conn["target_anchor_idx"])
                    if int(conn["source_anchor_idx"]) == int(current_src)
                    else int(conn["source_anchor_idx"])
                )
                segment = self._materialize_connection_segment(
                    planner_state=planner_state,
                    connection_info=conn,
                    desired_source_idx=current_src,
                    desired_target_idx=current_dst,
                )
                if len(concat_segments) > 0 and segment.shape[0] > 0:
                    segment = segment[1:]
                concat_segments.append(segment)
                current_src = current_dst
            completed_anchor_order.append(int(current_src))
            if current_src != dst_idx:
                if break_reason is None:
                    break_reason = (
                        f"partial_edge edge={anchor_short_label(src_idx, n_anchors)}->"
                        f"{anchor_short_label(dst_idx, n_anchors)} "
                        f"stopped_at={anchor_short_label(current_src, n_anchors)}"
                    )
                break

        if not concat_segments:
            return None

        concatenated = torch.cat(concat_segments, dim=0)
        completed_anchor_order = np.asarray(completed_anchor_order, dtype=np.int32)
        if completed_anchor_order.size != anchor_order.size:
            edge_summaries: list[str] = []
            for i in range(len(anchor_order) - 1):
                pair_key = self._anchor_pair_key(int(anchor_order[i]), int(anchor_order[i + 1]))
                if pair_key in planner_state["accepted_pair_edges"]:
                    status = "accepted"
                    cost = float(planner_state["accepted_pair_edges"][pair_key]["bridge_cost"])
                elif pair_key in planner_state["best_direct_bridge_info"]:
                    status = "direct"
                    cost = float(planner_state["best_direct_bridge_info"][pair_key]["bridge_cost"])
                else:
                    status = "missing"
                    cost = float(planner_state["effective_pair_cost"][pair_key])
                cost_text = f"{cost:.1f}" if np.isfinite(cost) else "inf"
                edge_summaries.append(f"{_pair_text(pair_key)}:{status}@{cost_text}")
            accepted_pairs_text = ", ".join(
                _pair_text(pair_key)
                for pair_key in sorted(planner_state["accepted_pair_edges"].keys())
            ) or "none"
            print(
                "[MCTD debug] Incomplete multi-tree assembly | "
                f"solver={_route_text(anchor_order)} | "
                f"completed={_route_text(completed_anchor_order)} | "
                f"reason={break_reason or 'unknown'} | "
                f"accepted={accepted_pairs_text} | "
                f"order_edges={'; '.join(edge_summaries)}",
                flush=True,
            )
        return {
            "output_plan": concatenated,
            "plan_unnormalized": concatenated,
            "postprocessed_plan": concatenated,
            "postprocessed_len": int(concatenated.shape[0]),
            "anchor_order": anchor_order,
            "completed_anchor_order": completed_anchor_order,
            "route_text": " -> ".join(
                anchor_short_label(int(idx), len(planner_state["trees"]))
                for idx in completed_anchor_order.tolist()
            ),
        }

    def _run_anchor_policy_loop(
        self,
        *,
        pre_round_check: Optional[Callable[[], Optional[str]]] = None,
        post_increment_check: Optional[Callable[[], Optional[str]]] = None,
        run_round: Callable[[int], dict],
    ) -> dict:
        loops = 0
        termination_reason = "unknown"
        while loops < self.val_max_loops:
            if pre_round_check is not None:
                pre_round_reason = pre_round_check()
                if pre_round_reason is not None:
                    termination_reason = str(pre_round_reason)
                    break

            loops += 1

            if post_increment_check is not None:
                post_increment_reason = post_increment_check()
                if post_increment_reason is not None:
                    termination_reason = str(post_increment_reason)
                    break

            round_result = run_round(loops) or {}
            round_termination = round_result.get("termination_reason")
            if round_termination is not None:
                termination_reason = str(round_termination)
            if bool(round_result.get("stop", False)):
                break
        else:
            termination_reason = "val_max_loops"

        return {
            "loops": loops,
            "termination_reason": termination_reason,
        }

    def _close_anchor_search_progress_bars(
        self,
        trees: list[Optional["MCTSTreeState"]],
        global_search_pbar: Optional[tqdm] = None,
    ) -> None:
        for tree in trees:
            if tree is not None and tree.pbar is not None:
                tree.pbar.close()
        if global_search_pbar is not None:
            global_search_pbar.close()

    def _run_anchor_planner(
        self,
        horizon: int,
        conditions: Optional[Any],
        start: torch.Tensor,
        goal: torch.Tensor,
        goal_normalized: torch.Tensor,
        initial_sim_state: dict,
        waypoint_xys: Optional[np.ndarray],
        agent,
        envs,
        namespace: str,
    ) -> dict:
        waypoint_xys = (
            np.zeros((0, 2), dtype=np.float32)
            if waypoint_xys is None
            else np.asarray(waypoint_xys, dtype=np.float32).reshape(-1, 2)
        )

        fixed_solver_result = None
        if self.multi_tree_route_mode == "fixed_temporal":
            fixed_route_info = self._compute_fixed_temporal_route_solution(
                start_xy=np.asarray(self._current_task_start_xy[0], dtype=np.float32).reshape(2),
                goal_xy=np.asarray(self._current_task_goal_xy[0], dtype=np.float32).reshape(2),
                waypoints_xy=waypoint_xys,
            )
            fixed_solver_result = fixed_route_info["temporal_route_solution"]

        return self._run_multi_tree_online_hamiltonian_planner(
            horizon=horizon,
            conditions=conditions,
            start=start,
            goal=goal,
            initial_sim_state=initial_sim_state,
            waypoint_xys=waypoint_xys,
            fixed_solver_result=fixed_solver_result,
            agent=agent,
            envs=envs,
            namespace=namespace,
        )

    def _anchor_pair_text_from_indices(
        self,
        pair: tuple[int, int] | list[int],
        n_anchors: int,
    ) -> str:
        a, b = int(pair[0]), int(pair[1])
        return f"{anchor_short_label(a, n_anchors)}-{anchor_short_label(b, n_anchors)}"

    def _init_root_node_uncertainty(
        self,
        root_node: "TreeNode",
        opposite_tree: Optional["MCTSTreeState"],
        horizon: int,
        conditions: Optional[Any],
        target_node_override: Optional["TreeNode"] = None,
    ) -> dict:
        """Initialize root node uncertainty before MCTS search begins.

        Builds a noisy init plan from root_node via _build_plan_from_leaf (no full
        denoising expansion), then runs fast uncertainty sampling (G*K copies,
        fast_sampling_steps) to populate root_node.cluster_subplans and set
        root_node.value = -U.

        Called once per tree before the MCTS while-loop when both
        use_cluster_subplan_as_expansion and use_uncertainty_as_value are True.
        """
        plan_tokens = horizon // self.frame_stack
        seg_size = plan_tokens // self.sequence_dividing_factor
        obs_mean_np = np.array(self.observation_mean)
        obs_std_np = np.array(self.observation_std)

        # 1. Build noisy init plan (init token + noisy sequence, no full denoising)
        noisy_plan, _ = self._build_plan_from_leaf(
            parent_node=root_node,
            plan_tokens=plan_tokens,
            segment_size=seg_size,
        )  # (n_tokens, 1, fs*c); structure: [obs_parent(1) | noisy(plan_tokens) | pad]

        # Extract plan_tokens portion (skip obs_parent at index 0), rearrange to (plan_tokens*fs, c)
        noisy_plan_tokens = noisy_plan[1 : plan_tokens + 1, 0, :]  # (plan_tokens, fs*c)
        val_plan_last = rearrange(
            noisy_plan_tokens, "t (fs c) -> (t fs) c", fs=self.frame_stack
        )  # (plan_tokens*fs, c)

        # 2. Root uncertainty starts from the raw root-noise state: no denoised prefix yet.
        root_levels = root_node.current_levels  # (1, plan_tokens)
        root_prefix_len_per_batch = np.array([0], dtype=int)

        # 3. Normalize root obs
        obs_normalized = torch.tensor(
            (root_node.obs - obs_mean_np) / obs_std_np,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # (1, obs_dim)

        # 4. Run fast uncertainty sampling (G*K denoising passes from the noisy init plan)
        val_plan_last_batch = val_plan_last.unsqueeze(1)  # (plan_tokens*fs, 1, c)
        unc_sampling_result = self._run_fast_uncertainty_sampling(
            parent_nodes=[root_node],
            val_plan_last_batch=val_plan_last_batch,
            updated_levels=root_levels,
            current_prefix_len_per_batch=root_prefix_len_per_batch,
            seg_size=seg_size,
            horizon=horizon,
            conditions=conditions,
            obs_normalized=obs_normalized,
            opposite_tree=opposite_tree,
            obs_mean_np=obs_mean_np,
            obs_std_np=obs_std_np,
            target_nodes_override=[target_node_override] if target_node_override is not None else None,
        )

        # 5. Select target node for uncertainty computation (based on root obs)
        if target_node_override is not None:
            target_node = target_node_override
        else:
            assert opposite_tree is not None, "opposite_tree must be provided when target_node_override is None"
            all_opposite_nodes = opposite_tree.get_all_nodes()
            target_node = self._select_dynamic_goal(
                current_leaf_obs=root_node.obs,
                opposite_tree_all_nodes=all_opposite_nodes,
            )

        # 6. Compute uncertainty and cluster_subplans
        unc_compute_result = self._compute_uncertainty_and_clusters(
            unc_hists_per_cand=unc_sampling_result["unc_hists_per_cand"],
            curr_plan_last_batch=val_plan_last_batch,
            parent_nodes=[root_node],
            node_depths=[0],
            seg_size=seg_size,
            unc_noise_levels_per_cand=unc_sampling_result["unc_noise_levels_per_cand"],
            unc_guidance_scale_per_cand=unc_sampling_result["unc_guidance_scale_per_cand"],
            target_nodes=[target_node],
        )

        # 7. Set root node cluster_subplans and value
        root_node.cluster_subplans = unc_compute_result["cluster_subplans"][0]
        if root_node.cluster_subplans == []:
            root_node.reset_children_slots(0, [])
        root_node.set_value(unc_compute_result["values"][0])  # values[0] = -U
        root_vinfo = {
            "node": root_node,
            "depth": int(root_node.depth),
            "value": unc_compute_result["values"][0],
            "target_node": target_node,
            "selection_count": root_node.selection_count,
            "uncertainty_plan_hist_frame": unc_compute_result["filtered_unc_hists"][0],
            "unc_diagnostics": unc_compute_result["unc_results"][0],
        }
        setattr(root_node, "_root_uncertainty_vinfo", root_vinfo)
        return root_vinfo

    """DEPRECATED"""
    def _run_mcts_search(
        self,
        tree: MCTSTreeState,
        opposite_tree: MCTSTreeState,
        horizon: int,
        conditions: Optional[Any],
        start: np.ndarray,
        goal: np.ndarray,
        single_step: bool = False,
        selected_parent_nodes: Optional[List["TreeNode"]] = None,
    ) -> tuple[MCTSTreeState, dict[str, dict]]:
        """
        (B function) Run the MCTS search loop for a given tree state.

        When `single_step=False` (default), runs until the local parent-selection
        budget or time_limit is reached.
        When `single_step=True`, executes exactly one Selection→Expansion→Simulation→
        Backpropagation→EarlyTermination cycle and returns.

        In two-tree search, `opposite_tree` provides all nodes from the other tree
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
        local_round_idx: int = 0
        local_parent_expansions: int = 0
        local_child_expansions: int = 0
        
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
            "search_budget": self.mctd_max_search_num,
            "plan_tokens": self._require_current_plan_tokens(),
        }, depth=0)

        while True:
            if self.time_limit is not None:
                if time.time() - self.start_time > self.time_limit:
                    break
            else:
                if local_parent_expansions >= self.mctd_max_search_num:
                    break

            ## For checking the virtual visit count
            # root_node.check_virtual_visit_count()
            # [MEMORY DEBUG] Periodic memory logging
            if self.profiler and (local_round_idx > 0) and (local_round_idx % 10 == 0):
                self.profiler.snapshot(
                    f"mcts_search_iter_{local_round_idx}_{tree.tag}",
                    phase=f"mcts_iter_{local_round_idx}"
                )

            ###############################
            # Selection
            #  When leaf parallelization is True, then the selection is done in partially parallel (the children nodes from same parent node are selected at the same time)
            #  When leaf parallelization is False, then the selection is done in fully sequential (only one node is selected at a time)

            selection_start_time = time.time()
            selected_nodes, expanded_node_candidates = [], []

            if selected_parent_nodes is None:
                psn = self.parallel_search_num
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
                        for i in range(len(selected_node._children_nodes)):
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
            else:
                if not self.parallel_multiple_visits:
                    expandable_node_names = root_node.get_expandable_node_names()
                for selected_node in selected_parent_nodes:
                    is_term = selected_node.is_terminal()
                    is_exp = selected_node.is_expandable(
                        consider_virtually_visited=(not self.parallel_multiple_visits)
                    )
                    is_sel = selected_node.is_selectable()

                    if is_term or (not is_sel and not is_exp):
                        continue

                    if self.leaf_parallelization:
                        for i in range(len(selected_node._children_nodes)):
                            child_slot = selected_node._children_nodes[i]
                            if child_slot['node'] is not None:
                                continue
                            if child_slot['permanently_dead']:
                                continue
                            if (not self.parallel_multiple_visits) and child_slot['virtually_visited']:
                                continue

                            expanded_node_candidate = selected_node.get_expandable_candidate(
                                index=i,
                                consider_virtually_visited=(not self.parallel_multiple_visits),
                            )
                            selected_nodes.append(selected_node)
                            expanded_node_candidates.append(expanded_node_candidate)
                            if not self.parallel_multiple_visits:
                                if expanded_node_candidate["name"] not in expandable_node_names:
                                    raise ValueError(
                                        f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names"
                                    )
                                expandable_node_names.remove(expanded_node_candidate["name"])
                    else:
                        expanded_node_candidate = selected_node.get_expandable_candidate(
                            index=None,
                            consider_virtually_visited=(not self.parallel_multiple_visits),
                        )
                        selected_nodes.append(selected_node)
                        expanded_node_candidates.append(expanded_node_candidate)
                        if not self.parallel_multiple_visits:
                            if expanded_node_candidate["name"] not in expandable_node_names:
                                raise ValueError(
                                    f"Expanded node candidate {expanded_node_candidate['name']} is not in expandable node names"
                                )
                            expandable_node_names.remove(expanded_node_candidate["name"])
            if len(selected_nodes) == 0:
                break
            round_parent_count = len({id(node): node for node in selected_nodes})
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

            # ------------------------------------------------------------------
            # [CLUSTER REUSE] Pre-process valid_candidates.
            # When use_cluster_subplan_as_expansion=True and a parent node has
            # cluster_subplans stored from a previous uncertainty-sampling pass,
            # replace its single candidate slot with N_c slots (one per cluster).
            # The parent's _children_nodes array is reset to match N_c.
            # After this block, valid_candidates / expanded_node_candidates /
            # selected_nodes are all replaced to stay in sync.
            # ------------------------------------------------------------------
            _is_cluster_reuse_expansion = False
            _cluster_reuse_slot_map: dict = {}  # new cand name → cluster index
            if self.use_cluster_subplan_as_expansion:
                # Build a mapping from candidate name → selected_node (parent),
                # then expand any parent that already has cluster_subplans.
                _old_sel_map = {}
                for _oi, _info in enumerate(expanded_node_candidates):
                    _old_sel_map[_info["name"]] = selected_nodes[_oi]

                _new_valid: list = []
                _new_sel: list = []
                for _info in valid_candidates:
                    _parent = _info["parent_node"]
                    _sel = _old_sel_map[_info["name"]]
                    if _parent.cluster_subplans is not None:
                        _is_cluster_reuse_expansion = True
                        _N_c = len(_parent.cluster_subplans)
                        _gs_list = [float(_cs["guidance_scale"]) for _cs in _parent.cluster_subplans]
                        _parent.reset_children_slots(_N_c, _gs_list)
                        for _slot_i in range(_N_c):
                            _cand = _parent.get_expandable_candidate(index=_slot_i)
                            _new_valid.append(_cand)
                            _new_sel.append(_sel)
                            _cluster_reuse_slot_map[_cand["name"]] = _slot_i
                    else:
                        _new_valid.append(_info)
                        _new_sel.append(_sel)

                if _is_cluster_reuse_expansion:
                    valid_candidates = _new_valid
                    expanded_node_candidates = _new_valid   # keep indices in sync
                    selected_nodes = _new_sel

            plan_tokens = self._require_current_plan_tokens()
            assert plan_tokens % self.sequence_dividing_factor == 0, (
                f"plan_tokens {plan_tokens} is not divisible by sequence_dividing_factor {self.sequence_dividing_factor}"
            )
            seg_size = plan_tokens // self.sequence_dividing_factor

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
            # Per-candidate cluster subplans: populated after _compute_node_uncertainty.
            # Each entry is list[dict] (one per cluster) or None.
            # Passed to child TreeNode so that the next expansion can reuse these plans.
            cluster_subplans_per_candidate: list = [None] * len(valid_candidates)

            # Pre-compute non-terminal candidate indices (constant across retries).
            # Also allocate per-candidate noise-level/guidance-scale buffers for Helper 2.
            # These are populated on the first retry and reused (they are constant across retries
            # because they depend only on parent depths and self.mctd_guidance_scales).
            if self.use_uncertainty_as_value:
                _B_cands_pre = len(expanded_node_candidates)
                non_terminal_cand_indices = [
                    _i for _i in range(_B_cands_pre)
                    if expanded_node_candidates[_i]["parent_node"].depth + 1 < terminal_depth
                ]
                _unc_noise_levels_per_cand: list = [None] * len(valid_candidates)
                _unc_guidance_scale_per_cand: list = [None] * len(valid_candidates)
            else:
                non_terminal_cand_indices = []
                _unc_noise_levels_per_cand = []
                _unc_guidance_scale_per_cand = []

            for _ in range(
                self.num_tries_for_bad_plans
            ):  # resample when the generated plan is terrible (e.g., not moving plans)
                ###############################
                # Expansion
                expansion_start_time = time.time()

                if _is_cluster_reuse_expansion:
                    # ── Cluster-reuse path ──────────────────────────────────────
                    # Directly assemble expanded_node_plan_hists from stored
                    # cluster subplans instead of running parallel_plan.
                    # Each candidate maps to a cluster slot in its parent node.
                    _cr_plan_hist_list = []
                    _cr_updated_levels_list = []
                    for _cinfo in valid_candidates:
                        _cp = _cinfo["parent_node"]
                        assert _cp.cluster_subplans is not None, (
                            f"Cluster-reuse expansion expects parent '{_cp.name}' to have "
                            "cluster_subplans, but it is None."
                        )
                        _slot_i = _cluster_reuse_slot_map[_cinfo["name"]]
                        _cs = _cp.cluster_subplans[_slot_i]
                        # plan_hist: (fast_steps+1, plan_tokens*fs, c) → add batch dim
                        _ph = _cs["plan_hist"]
                        if not isinstance(_ph, torch.Tensor):
                            _ph = torch.tensor(_ph, dtype=torch.float32, device=self.device)
                        else:
                            _ph = _ph.to(self.device)
                        _cr_plan_hist_list.append(_ph.unsqueeze(2))  # (fast_steps+1, plan_tokens*fs, 1, c)
                        _cr_updated_levels_list.append(
                            np.asarray(_cs["current_levels"], dtype=np.int64)
                        )  # (1, plan_tokens)
                    # Stack along batch dim
                    expanded_node_plan_hists = torch.cat(_cr_plan_hist_list, dim=2)  # (fast_steps+1, plan_tokens*fs, B, c)
                    expanded_node_updated_levels = np.concatenate(_cr_updated_levels_list, axis=0)  # (B, plan_tokens)
                    expanded_node_noise_levels = None      # not used in cluster-reuse path
                    expanded_node_guidance_scales = None  # not used in cluster-reuse path
                    self._clear_parallel_plan_aux_state()
                    self._expansion_step_captures_by_name = {}  # no denoising captures
                else:
                    # ── Standard denoising path ──────────────────────────────────
                    expanded_node_plans = []
                    expanded_node_noise_levels = []
                    expanded_node_guidance_scales = []

                    prefix_len_list = []
                    for info in expanded_node_candidates:
                        # Build pre-built plan from leaf history (n_tokens(=t), 1, fs*c)
                        initial_plan, prefix_len = self._build_plan_from_leaf(
                            parent_node=info["parent_node"],
                            plan_tokens=plan_tokens,
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

                    # Generate the tree-conditioned denoising schedule for this batch.
                    expanded_node_noise_levels = self._generate_tree_conditioned_schedule(
                        parent_levels,
                        prefix_len_per_batch=np.array(prefix_len_list, dtype=int),
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
                                'guidance_xstart_displacements': {k: v[:, _sci] for k, v in step.get('guidance_xstart_displacements', {}).items()},
                                'pred_x_start_pos': step['pred_x_start_pos'][:, _sci] if step.get('pred_x_start_pos') is not None else None,
                                'pred_x_start_pos_after': step['pred_x_start_pos_after'][:, _sci] if step.get('pred_x_start_pos_after') is not None else None,
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
                        "round_idx": local_round_idx + 1,
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
                            "round_idx": local_round_idx + 1,
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


                if not self.mcts_use_replan:
                    # Skip replanning: use expansion results directly for value.
                    # val_plan_hists: plan used for value / filtering (no replan → same as expansion)
                    val_plan_hists = expanded_node_plan_hists

                    # assert prefix_len_list is not None
                    # batch_size = expanded_node_plan_hists.shape[2]
                    # plans_tokens = rearrange(expanded_node_plan_hists, "m (t fs) b c -> m t fs b c", fs=self.frame_stack)
                    # num_tokens_to_check = seg_size
                    # assert len(set(prefix_len_list)) == 1, \
                    #     f"Expected uniform prefix_len across batch, got {prefix_len_list}"
                    # plen = prefix_len_list[0]
                    # t_end = min(plen + num_tokens_to_check, plans_tokens.shape[1])
                    # sliced_hists = plans_tokens[:, plen:t_end, :, :, :]  # (m, T_check, fs, B, c)
                    # processed_hists = rearrange(sliced_hists, "m t fs b c -> m (t fs) b c")
                    # is_feasible = check_feasibility(processed_hists)

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
                    replan_prefix_len_per_batch = np.array([
                        (expanded_node_candidates[i]["parent_node"].depth + 1) * seg_size
                        for i in range(len(expanded_node_candidates))
                    ], dtype=int)
                    replan_noise_levels = (
                        self._generate_tree_conditioned_schedule(
                            replan_initial_levels,
                            prefix_len_per_batch=replan_prefix_len_per_batch,
                            is_replanning=True,
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

                    # val_plan_hists: plan used for value / filtering (replan → replanned)
                    val_plan_hists = replanned_plan_hists

                ##### NODE FEASIBILITY CHECK #####
                _parent_nodes_for_feasibility = [
                    expanded_node_candidates[i]["parent_node"]
                    for i in range(len(expanded_node_candidates))
                ]
                is_feasible = self._check_plan_batch_feasibility(
                    plan_hists=val_plan_hists,
                    root_obs_list=[
                        self._get_root_obs(parent_node)
                        for parent_node in _parent_nodes_for_feasibility
                    ],
                    progress_obs_list=[
                        parent_node.obs for parent_node in _parent_nodes_for_feasibility
                    ],
                    prefix_len_frames_list=[
                        self._get_prefix_len_frames_from_depth(parent_node.depth, seg_size)
                        for parent_node in _parent_nodes_for_feasibility
                    ],
                    subplan_tail_depths=[
                        expanded_node_candidates[i]["depth"]
                        for i in range(len(expanded_node_candidates))
                    ],
                    seg_size=seg_size,
                )
                
                ##### NODE UNCERTAINTY CHECK #####

                # uncertainty_plan_hists_per_candidate is initialized before the retry loop.
                # Here we populate it via _run_fast_uncertainty_sampling for non-terminal
                # candidates when use_uncertainty_as_value=True.
                # Input: val_plan_hists (replanned if mcts_use_replan, else expanded).
                if self.use_uncertainty_as_value and non_terminal_cand_indices:
                    assert expanded_node_updated_levels is not None, (
                        "expanded_node_updated_levels must be set for uncertainty sampling"
                    )
                    _nt_parent_nodes = [
                        expanded_node_candidates[_i]["parent_node"]
                        for _i in non_terminal_cand_indices
                    ]
                    _nt_val_plan_last = val_plan_hists[-1][:, non_terminal_cand_indices, :]
                    # (plan_tokens*fs, B_nt, c)
                    _nt_updated_levels = expanded_node_updated_levels[non_terminal_cand_indices]
                    # (B_nt, plan_tokens)
                    _nt_idx_tensor = torch.tensor(
                        non_terminal_cand_indices, device=effective_obs_normalized.device
                    )
                    _nt_obs_norm = effective_obs_normalized[_nt_idx_tensor]
                    # (B_nt, obs_dim)

                    _unc_fast_result = self._run_fast_uncertainty_sampling(
                        parent_nodes=_nt_parent_nodes,
                        val_plan_last_batch=_nt_val_plan_last,
                        updated_levels=_nt_updated_levels,
                        current_prefix_len_per_batch=np.array(
                            [
                                expanded_node_candidates[_i]["depth"] * seg_size
                                for _i in non_terminal_cand_indices
                            ],
                            dtype=int,
                        ),
                        seg_size=seg_size,
                        horizon=horizon,
                        conditions=conditions,
                        obs_normalized=_nt_obs_norm,
                        opposite_tree=opposite_tree,
                        obs_mean_np=obs_mean_np,
                        obs_std_np=obs_std_np,
                    )

                    for _ii, _i in enumerate(non_terminal_cand_indices):
                        _unc_h = _unc_fast_result["unc_hists_per_cand"][_ii]
                        _fallback_unc_plan_hists[_i] = _unc_h
                        uncertainty_plan_hists_per_candidate[_i] = _unc_h
                        # Store noise levels and guidance scales (constant across retries)
                        if _unc_noise_levels_per_cand[_i] is None:
                            _unc_noise_levels_per_cand[_i] = (
                                _unc_fast_result["unc_noise_levels_per_cand"][_ii]
                            )
                            _unc_guidance_scale_per_cand[_i] = (
                                _unc_fast_result["unc_guidance_scale_per_cand"][_ii]
                            )


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
                    expanded_node_candidates, final_best_plans
                )
                for i in range(len(achieved_infos)):
                    if achieved_infos[i] == "Achieved":
                        tree.achieved = True
                        achieved_indices.append(i)

                # Compute -uncertainty as value for each candidate via helper.
                values = np.zeros(len(expanded_node_candidates))

                # Terminal-depth candidates: assign -inf (no future segment).
                terminal_cand_indices = [
                    i for i, cand in enumerate(expanded_node_candidates)
                    if cand["depth"] >= terminal_depth
                ]
                for i in terminal_cand_indices:
                    values[i] = -np.inf
                    unc_results[i] = {"U": np.inf}

                # Non-terminal candidates: compute via _compute_uncertainty_and_clusters.
                if non_terminal_cand_indices:
                    _nt_unc_hists = [
                        uncertainty_plan_hists_per_candidate[_i]
                        for _i in non_terminal_cand_indices
                    ]
                    for _ii, _i in enumerate(non_terminal_cand_indices):
                        assert _nt_unc_hists[_ii] is not None, (
                            f"[Uncertainty] uncertainty_plan_hists_per_candidate[{_i}] is None "
                            "after fallback injection — this should not happen."
                        )
                    _nt_curr_plan_last = expanded_node_plan_hists[-1][:, non_terminal_cand_indices, :]
                    # (plan_tokens*fs, B_nt, c)
                    _nt_depths = [
                        expanded_node_candidates[_i]["depth"] for _i in non_terminal_cand_indices
                    ]
                    _nt_target_nodes = [
                        expanded_node_candidates[_i]["target_node"]
                        for _i in non_terminal_cand_indices
                    ]
                    _nt_noise_levels = [
                        _unc_noise_levels_per_cand[_i] for _i in non_terminal_cand_indices
                    ]
                    _nt_guidance_scales = [
                        _unc_guidance_scale_per_cand[_i] for _i in non_terminal_cand_indices
                    ]

                    _unc_compute_result = self._compute_uncertainty_and_clusters(
                        unc_hists_per_cand=_nt_unc_hists,
                        curr_plan_last_batch=_nt_curr_plan_last,
                        parent_nodes=_nt_parent_nodes,
                        node_depths=_nt_depths,
                        seg_size=seg_size,
                        unc_noise_levels_per_cand=_nt_noise_levels,
                        unc_guidance_scale_per_cand=_nt_guidance_scales,
                        target_nodes=_nt_target_nodes,
                    )

                    for _ii, _i in enumerate(non_terminal_cand_indices):
                        values[_i] = _unc_compute_result["values"][_ii]
                        unc_results[_i] = _unc_compute_result["unc_results"][_ii]
                        cluster_subplans_per_candidate[_i] = (
                            _unc_compute_result["cluster_subplans"][_ii]
                        )
                        uncertainty_plan_hists_per_candidate[_i] = (
                            _unc_compute_result["filtered_unc_hists"][_ii]
                        )
            else:
                values, achieved_infos, achieved_ts = self.calculate_values_bidir(
                    expanded_node_candidates, final_best_plans
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

            expanded_node_infos = {}
            for i in range(len(expanded_node_candidates)):  # B
                if not is_kept[i]:
                    continue  # slot remains empty, available for future expansion rounds
                name = expanded_node_candidates[i]["name"]
                if name not in expanded_node_infos:
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
                    expanded_node_infos[name]["cluster_subplans"] = cluster_subplans_per_candidate[i]
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
                        expanded_node_infos[name]["cluster_subplans"] = cluster_subplans_per_candidate[i]

            # Use parent_node directly from expanded_node_infos (selected_nodes_for_expansion
            # is no longer needed — parent_node is already stored in each candidate info dict).
            for name in expanded_node_infos:
                parent_node_for_expand = expanded_node_infos[name]["parent_node"]
                expand_kwargs = {k: v for k, v in expanded_node_infos[name].items()
                                 if k not in ("expanded_plan_hist_frame", "replanned_plan_hist_frame",
                                              "uncertainty_plan_hist_frame", "unc_diagnostics")}
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
                    selected_node.backpropagate(
                        preserve_value=self.use_uncertainty_as_value,
                    )

            backprop_end_time = time.time()
            tree.backprop_time.append(backprop_end_time - backprop_start_time)

            ######################
            # Early Termination
            early_termination_start_time = time.time()

            local_round_idx += 1
            local_parent_expansions += round_parent_count
            local_child_expansions += len(expanded_node_candidates)
            if tree.pbar is not None:
                tree.pbar.update(round_parent_count)
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

                    is_forward_tree = self._tree_uses_forward_prefix_semantics(tree)
                    visualize_valid_frame_bounds = [
                        self._get_plan_viz_valid_frame_bounds(
                            expanded_node_candidates[i]["depth"],
                            seg_size,
                            terminal_expanded_plans.shape[0],
                            is_forward_tree,
                        )
                        for i in visualize_indices
                    ]

                    # For goal tree visualization, flip the plans so they appear in start→goal direction
                    if not is_forward_tree:
                        terminal_expanded_plans = torch.flip(terminal_expanded_plans, [0])

                    _viz_node_t0 = time.time()
                    self.visualize_node_value_plans(
                        is_achieved_plan,
                        local_round_idx,
                        terminal_values,
                        terminal_names,
                        terminal_expanded_plans,
                        start,
                        goal,
                        tag=tree.tag,
                        valid_frame_bounds=visualize_valid_frame_bounds,
                    )
                    _viz_fp_node_ms = (time.time() - _viz_node_t0) * 1000

                    if self.viz_compare_expanded_to_value:
                        _viz_cmp_t0 = time.time()
                        terminal_value_plans = replanned_plan_hists[
                            -1, :, visualize_indices
                        ]  # (t fs) b c
                        if not is_forward_tree:
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
                        "expansion_idx": local_round_idx,
                        "expansion_ms": round(tree.expansion_time[-1] * 1000, 1),
                        "selection_ms": round(tree.selection_time[-1] * 1000, 1) if tree.selection_time else 0.0,
                        "replanning_ms": round(tree.replanning_time[-1] * 1000, 1) if tree.replanning_time else 0.0,
                        "backprop_ms": round(tree.backprop_time[-1] * 1000, 1) if tree.backprop_time else 0.0,
                        "early_term_ms": round(tree.early_termination_time[-1] * 1000, 1) if tree.early_termination_time else 0.0,
                    }, depth=0)
                break

        if not single_step and tree.pbar is not None:
            tree.pbar.close()

        # [LOGGING] Record search completion → guidance_anal (tree quality) + timing_anal (phase breakdown)
        terminal_depth_reached = tree.max_depth >= tree.terminal_depth
        self._glog("tree.search.complete", {
            "tree_tag": tree.tag,
            "final_selected_parents": local_parent_expansions,
            "n_rounds": local_round_idx,
            "final_max_depth": tree.max_depth,
            "terminal_depth": tree.terminal_depth,
            "terminal_depth_reached": terminal_depth_reached,
            "total_children_created": local_child_expansions,
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
            "n_iters": local_round_idx,
            "n_selected_parents": local_parent_expansions,
            "n_nodes": local_child_expansions,
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
    # Helper functions for alternating tree expansion and scheduling
    # =========================================================================

    def _sample_clamped_noise(self, n_tokens: int, batch_size: int = 1) -> torch.Tensor:
        """Sample (n_tokens, batch_size, *x_stacked_shape) randn clamped to clip_noise."""
        noise = torch.randn((n_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        return torch.clamp(noise, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)

    def _parent_selection_budget(self) -> int:
        """Return how many distinct parent nodes may be selected in one round."""
        if self.leaf_parallelization:
            return self.parallel_search_num // len(self.mctd_guidance_scales)
        return self.parallel_search_num

    def _run_global_uncertainty_expansion_round(
        self,
        selected_parent_infos: List[dict],
        horizon: int,
        conditions: Optional[Any],
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
    ) -> dict:
        """Expand globally selected parents in one mixed uncertainty batch."""
        del start, goal  # Mixed expansion does not require raw endpoints directly.

        if not selected_parent_infos:
            return {
                "expanded_node_infos": {},
                "tree_batches": {},
                "mixed_stats": {"n_selected_parents": 0, "n_candidates": 0},
            }

        selected_nodes: list = []
        expanded_node_candidates: list = []
        base_tree = selected_parent_infos[0]["tree"]
        terminal_depth = base_tree.terminal_depth
        plan_tokens = self._require_current_plan_tokens()
        seg_size = plan_tokens // self.sequence_dividing_factor

        for parent_info in selected_parent_infos:
            selected_tree: MCTSTreeState = parent_info["tree"]
            opposite_tree: MCTSTreeState = parent_info["opposite_tree"]
            selected_node: "TreeNode" = parent_info["node"]
            parent_key = f"{selected_tree.tag}:{selected_node.name}"

            if selected_node.obs is None:
                continue

            is_term = selected_node.is_terminal()
            is_exp = selected_node.is_expandable(
                consider_virtually_visited=(not self.parallel_multiple_visits)
            )
            is_sel = selected_node.is_selectable()
            if is_term or (not is_sel and not is_exp):
                continue

            if self.leaf_parallelization:
                # When cluster-reuse is active and this parent already has cluster_subplans,
                # we only need ONE sentinel candidate: the actual N_c child slots will be
                # created later by reset_children_slots in the cluster-reuse block.
                # For parents without cluster_subplans, generate one candidate per guidance scale
                # (standard leaf-parallelization behaviour).
                _is_cluster_parent = (
                    self.use_cluster_subplan_as_expansion
                    and selected_node.cluster_subplans is not None
                )
                _sentinel_done = False
                for i in range(len(selected_node._children_nodes)):
                    if _is_cluster_parent and _sentinel_done:
                        break
                    child_slot = selected_node._children_nodes[i]
                    if child_slot["node"] is not None:
                        continue
                    if child_slot["permanently_dead"]:
                        continue
                    if (not self.parallel_multiple_visits) and child_slot["virtually_visited"]:
                        continue
                    expanded_node_candidate = selected_node.get_expandable_candidate(
                        index=i,
                        consider_virtually_visited=(not self.parallel_multiple_visits),
                    )
                    expanded_node_candidate["selected_tree"] = selected_tree
                    expanded_node_candidate["opposite_tree"] = opposite_tree
                    expanded_node_candidate["target_node"] = parent_info.get("target_node")
                    expanded_node_candidate["parent_key"] = parent_key
                    selected_nodes.append(selected_node)
                    expanded_node_candidates.append(expanded_node_candidate)
                    if _is_cluster_parent:
                        _sentinel_done = True
            else:
                expanded_node_candidate = selected_node.get_expandable_candidate(
                    index=None,
                    consider_virtually_visited=(not self.parallel_multiple_visits),
                )
                expanded_node_candidate["selected_tree"] = selected_tree
                expanded_node_candidate["opposite_tree"] = opposite_tree
                expanded_node_candidate["target_node"] = parent_info.get("target_node")
                expanded_node_candidate["parent_key"] = parent_key
                selected_nodes.append(selected_node)
                expanded_node_candidates.append(expanded_node_candidate)

        if not expanded_node_candidates:
            return {
                "expanded_node_infos": {},
                "tree_batches": {},
                "mixed_stats": {
                    "n_selected_parents": len(selected_parent_infos),
                    "n_candidates": 0,
                },
            }

        valid_candidates = [
            info for info in expanded_node_candidates if info["parent_node"].obs is not None
        ]
        if not valid_candidates:
            return {
                "expanded_node_infos": {},
                "tree_batches": {},
                "mixed_stats": {
                    "n_selected_parents": len(selected_parent_infos),
                    "n_candidates": 0,
                },
            }

        _is_cluster_reuse_expansion = False
        _cluster_reuse_slot_map: dict[str, int] = {}
        if self.use_cluster_subplan_as_expansion:
            _all_valid_have_cluster_subplans = (
                len(valid_candidates) > 0
                and all(
                    _info["parent_node"].cluster_subplans is not None
                    for _info in valid_candidates
                )
            )
            _old_sel_map: dict[str, tuple] = {}
            for _oi, _info in enumerate(expanded_node_candidates):
                _cand_key = f"{_info['selected_tree'].tag}:{_info['name']}"
                _old_sel_map[_cand_key] = (
                    selected_nodes[_oi],
                    _info["selected_tree"],
                    _info["opposite_tree"],
                    _info["parent_key"],
                )

            _new_valid: list = []
            _new_sel: list = []
            if _all_valid_have_cluster_subplans:
                for _info in valid_candidates:
                    _cand_key = f"{_info['selected_tree'].tag}:{_info['name']}"
                    _sel, _sel_tree, _opp_tree, _parent_key = _old_sel_map[_cand_key]
                    _parent = _info["parent_node"]
                    _is_cluster_reuse_expansion = True
                    _n_clusters = len(_parent.cluster_subplans)
                    _gs_list = [
                        float(_cs["guidance_scale"]) for _cs in _parent.cluster_subplans
                    ]
                    _parent.reset_children_slots(_n_clusters, _gs_list)
                    for _slot_i in range(_n_clusters):
                        _cand = _parent.get_expandable_candidate(index=_slot_i)
                        _cand["selected_tree"] = _sel_tree
                        _cand["opposite_tree"] = _opp_tree
                        _cand["target_node"] = _info.get("target_node")
                        _cand["parent_key"] = _parent_key
                        _new_valid.append(_cand)
                        _new_sel.append(_sel)
                        _cluster_reuse_slot_map[
                            f"{_sel_tree.tag}:{_cand['name']}"
                        ] = _slot_i
                valid_candidates = _new_valid
                expanded_node_candidates = _new_valid
                selected_nodes = _new_sel
            else:
                expanded_node_candidates = valid_candidates
                selected_nodes = selected_nodes[: len(valid_candidates)]
        else:
            expanded_node_candidates = valid_candidates
            selected_nodes = selected_nodes[: len(valid_candidates)]

        obs_mean_np = np.array(self.observation_mean)
        obs_std_np = np.array(self.observation_std)
        eff_obs_norm_list, eff_goal_norm_list = [], []

        for info in expanded_node_candidates:
            parent_node = info["parent_node"]
            parent_obs = parent_node.obs
            p_norm = torch.tensor(
                (parent_obs - obs_mean_np) / obs_std_np,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            eff_obs_norm_list.append(p_norm)

            target_node = info.get("target_node")
            if target_node is None:
                all_opposite_nodes = info["opposite_tree"].get_all_nodes()
                assert len(all_opposite_nodes) > 0, "opposite_tree has no nodes"
                target_node = self._select_dynamic_goal(
                    current_leaf_obs=parent_obs,
                    opposite_tree_all_nodes=all_opposite_nodes,
                )
            info["target_node"] = target_node
            target_pos = target_node.obs
            g_norm = torch.tensor(
                (target_pos - obs_mean_np) / obs_std_np,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            eff_goal_norm_list.append(g_norm)

        effective_obs_normalized = torch.cat(eff_obs_norm_list, dim=0)
        effective_goal_normalized = torch.cat(eff_goal_norm_list, dim=0)

        filtered_expanded_node_plan_hists = [None] * len(expanded_node_candidates)
        filtered_replanned_plan_hists = [None] * len(expanded_node_candidates)
        uncertainty_plan_hists_per_candidate: list = [None] * len(expanded_node_candidates)
        fallback_unc_plan_hists: list = [None] * len(expanded_node_candidates)
        cluster_subplans_per_candidate: list = [None] * len(expanded_node_candidates)
        non_terminal_cand_indices = [
            i
            for i, cand in enumerate(expanded_node_candidates)
            if cand["parent_node"].depth + 1 < terminal_depth
        ]
        unc_noise_levels_per_cand: list = [None] * len(expanded_node_candidates)
        unc_guidance_scale_per_cand: list = [None] * len(expanded_node_candidates)
        expanded_node_plan_hists: Optional[torch.Tensor] = None
        replanned_plan_hists: Optional[torch.Tensor] = None
        expanded_node_updated_levels: Optional[np.ndarray] = None
        values = np.zeros(len(expanded_node_candidates))
        unc_results: dict = {}
        achieved_indices: list[int] = []
        mixed_stats = {
            "n_selected_parents": len(selected_parent_infos),
            "n_candidates": len(expanded_node_candidates),
            "cluster_reuse": bool(_is_cluster_reuse_expansion),
        }

        for _ in range(self.num_tries_for_bad_plans):
            expansion_start_time = time.time()

            if _is_cluster_reuse_expansion:
                cr_plan_hist_list = []
                cr_updated_levels_list = []
                for cinfo in expanded_node_candidates:
                    parent = cinfo["parent_node"]
                    assert parent.cluster_subplans is not None, (
                        f"Cluster-reuse expansion expects parent '{parent.name}' to have "
                        "cluster_subplans, but it is None."
                    )
                    cand_key = f"{cinfo['selected_tree'].tag}:{cinfo['name']}"
                    slot_i = _cluster_reuse_slot_map[cand_key]
                    cluster_subplan = parent.cluster_subplans[slot_i]
                    plan_hist = cluster_subplan["plan_hist"]
                    if not isinstance(plan_hist, torch.Tensor):
                        plan_hist = torch.tensor(
                            plan_hist, dtype=torch.float32, device=self.device
                        )
                    else:
                        plan_hist = plan_hist.to(self.device)
                    cr_plan_hist_list.append(plan_hist.unsqueeze(2))
                    cr_updated_levels_list.append(
                        np.asarray(cluster_subplan["current_levels"], dtype=np.int64)
                    )
                expanded_node_plan_hists = torch.cat(cr_plan_hist_list, dim=2)
                expanded_node_updated_levels = np.concatenate(cr_updated_levels_list, axis=0)
                expanded_node_guidance_scales = None
                self._clear_parallel_plan_aux_state()
                self._expansion_step_captures_by_name = {}
            else:
                expanded_node_plans = []
                expanded_node_guidance_scales = []
                prefix_len_list = []
                for info in expanded_node_candidates:
                    initial_plan, prefix_len = self._build_plan_from_leaf(
                        parent_node=info["parent_node"],
                        plan_tokens=plan_tokens,
                        segment_size=seg_size,
                    )
                    expanded_node_plans.append(initial_plan)
                    expanded_node_guidance_scales.append(info["guidance_scale"])
                    prefix_len_list.append(prefix_len)

                expanded_node_guidance_scales = torch.tensor(
                    expanded_node_guidance_scales, device=self.device
                )

                parent_levels_list = []
                for info in expanded_node_candidates:
                    parent_node = info["parent_node"]
                    if parent_node.current_levels is not None:
                        parent_levels_list.append(parent_node.current_levels)
                    else:
                        plan_tokens = horizon // self.frame_stack
                        parent_levels_list.append(
                            np.full(
                                (1, plan_tokens),
                                self.sampling_timesteps,
                                dtype=np.int64,
                            )
                        )

                parent_levels = np.concatenate(parent_levels_list, axis=0)
                expanded_node_noise_levels = self._generate_tree_conditioned_schedule(
                    parent_levels,
                    prefix_len_per_batch=np.array(prefix_len_list, dtype=int),
                )
                expanded_node_updated_levels = expanded_node_noise_levels[:, -1, :]
                self._current_tree_tag = "mixed"

                parent_to_gid: dict[str, int] = {}
                group_ids: list[int] = []
                for cand in expanded_node_candidates:
                    parent_key = cand["parent_key"]
                    if parent_key not in parent_to_gid:
                        parent_to_gid[parent_key] = len(parent_to_gid)
                    group_ids.append(parent_to_gid[parent_key])

                expanded_node_plan_hists = self.parallel_plan(
                    start=effective_obs_normalized,
                    goal=effective_goal_normalized,
                    horizon=horizon,
                    conditions=conditions,
                    guidance_scale=expanded_node_guidance_scales,
                    noise_level=expanded_node_noise_levels,
                    plans=expanded_node_plans,
                    prefix_len_list=prefix_len_list,
                    particle_guidance_scale=self.particle_guidance_scale,
                    group_ids=group_ids,
                    call_type="expansion",
                )

            raw_captures = getattr(self, "_parallel_plan_step_captures", None)
            exp_sc_by_global_key: dict[str, list] = {}
            if raw_captures:
                for sci, cand in enumerate(expanded_node_candidates):
                    exp_sc_by_global_key[f"{cand['selected_tree'].tag}:{cand['name']}"] = [
                        {
                            "prior_pred_noise": step["prior_pred_noise"][:, sci]
                            if step["prior_pred_noise"] is not None
                            else None,
                            "guidance_grads": {
                                k: v[:, sci] for k, v in step["guidance_grads"].items()
                            },
                            "guidance_grads_clean": {
                                k: v[:, sci]
                                for k, v in step.get("guidance_grads_clean", {}).items()
                            },
                            "guidance_xstart_displacements": {
                                k: v[:, sci]
                                for k, v in step.get(
                                    "guidance_xstart_displacements", {}
                                ).items()
                            },
                            "pred_x_start_pos": step["pred_x_start_pos"][:, sci]
                            if step.get("pred_x_start_pos") is not None
                            else None,
                            "pred_x_start_pos_after": step["pred_x_start_pos_after"][:, sci]
                            if step.get("pred_x_start_pos_after") is not None
                            else None,
                            "noise_level": step["noise_level"][:, sci]
                            if step.get("noise_level") is not None
                            else None,
                        }
                        for step in raw_captures
                    ]

            g_losses = getattr(self, "_last_guidance_losses", {})
            if g_losses or expanded_node_guidance_scales is not None:
                final_plan = self._unnormalize_x(expanded_node_plan_hists[-1])
                goal_unnorm = self._unnormalize_x(effective_goal_normalized)
                final_pos = (
                    final_plan[-1, :, self.pos_dim_indices].detach().cpu().numpy()
                )
                goal_pos = goal_unnorm[:, self.pos_dim_indices].detach().cpu().numpy()
                dist_per_batch = np.linalg.norm(final_pos - goal_pos, axis=-1).tolist()
                scales = (
                    expanded_node_guidance_scales.tolist()
                    if hasattr(expanded_node_guidance_scales, "tolist")
                    else list(expanded_node_guidance_scales)
                    if expanded_node_guidance_scales is not None
                    else []
                )
                eff_scale = float(np.mean(scales)) if scales else 0.0
                self._glog(
                    "guidance.combined",
                    {
                        "tree_tag": "mixed",
                        "global_parent_count": self.global_search_num,
                        "eff_goal_scale": eff_scale,
                        "batch_size": len(expanded_node_candidates),
                        "dist_per_batch": [round(d, 4) for d in dist_per_batch],
                        "final_token_dist": round(float(np.mean(dist_per_batch)), 4),
                        "anchor_loss": round(g_losses.get("anchor", 0.0), 6),
                        "goal_loss": round(g_losses.get("goal", 0.0), 6),
                        "rdf_loss": round(g_losses.get("rdf", 0.0), 6),
                        "particle_loss": round(g_losses.get("particle", 0.0), 6),
                        "goal_anchor_ratio": round(
                            abs(g_losses.get("goal", 0.0))
                            / (abs(g_losses.get("anchor", 1e-9)) + 1e-9),
                            4,
                        ),
                    },
                    depth=1,
                )

            if raw_captures and len(raw_captures) > 1:
                last_cap = raw_captures[-1]
                prior_noise = last_cap.get("prior_pred_noise")
                guidance_grads = last_cap.get("guidance_grads", {})
                if prior_noise is not None and guidance_grads:
                    prior_norm = float(prior_noise.norm().item())
                    guidance_norms = {
                        k: round(float(v.norm().item()), 6)
                        for k, v in guidance_grads.items()
                    }
                    guidance_total = sum(guidance_norms.values())
                    self._glog(
                        "diffusion.grad_comparison",
                        {
                            "tree_tag": "mixed",
                            "global_parent_count": self.global_search_num,
                            "prior_norm": round(prior_norm, 6),
                            "guidance_total_norm": round(guidance_total, 6),
                            "ratio": round(guidance_total / (prior_norm + 1e-9), 4),
                            "per_guidance_norms": guidance_norms,
                        },
                        depth=1,
                    )

            assert expanded_node_plan_hists.ndim == 4
            assert expanded_node_plan_hists.shape[2] == len(expanded_node_candidates)
            mixed_stats["expansion_ms"] = round(
                (time.time() - expansion_start_time) * 1000, 1
            )

            replanning_start_time = time.time()

            if not self.mcts_use_replan:
                replanned_plan_hists = expanded_node_plan_hists
                val_plan_hists = expanded_node_plan_hists
            else:
                replan_init_plans = []
                for i in range(len(expanded_node_candidates)):
                    plan_t_fs = expanded_node_plan_hists[-1, :, i].unsqueeze(1)
                    plan_tokens = horizon // self.frame_stack
                    plan_rearranged = rearrange(
                        plan_t_fs, "(t fs) b c -> t b (fs c)", fs=self.frame_stack
                    )
                    parent_node = expanded_node_candidates[i]["parent_node"]
                    sim_plan, _ = self._build_plan_from_leaf(
                        parent_node, plan_tokens, seg_size, expanded_plan=plan_rearranged
                    )
                    replan_init_plans.append(sim_plan)

                assert expanded_node_updated_levels is not None
                replan_initial_levels = np.concatenate(
                    [expanded_node_updated_levels[i : i + 1] for i in range(len(expanded_node_candidates))],
                    axis=0,
                )
                replan_prefix_len_per_batch = np.array(
                    [
                        (expanded_node_candidates[i]["parent_node"].depth + 1) * seg_size
                        for i in range(len(expanded_node_candidates))
                    ],
                    dtype=int,
                )
                replan_noise_levels = self._generate_tree_conditioned_schedule(
                    replan_initial_levels,
                    prefix_len_per_batch=replan_prefix_len_per_batch,
                    is_replanning=True,
                )
                replanned_plan_hists = self.parallel_plan(
                    effective_obs_normalized,
                    effective_goal_normalized,
                    horizon,
                    conditions,
                    guidance_scale=torch.zeros(
                        len(expanded_node_candidates), device=self.device
                    ),
                    noise_level=replan_noise_levels,
                    plans=replan_init_plans,
                    prefix_len_list=[
                        (expanded_node_candidates[i]["parent_node"].depth) * seg_size
                        for i in range(len(expanded_node_candidates))
                    ],
                    call_type="replan",
                )
                assert replanned_plan_hists.ndim == 4
                assert replanned_plan_hists.shape[2] == len(expanded_node_candidates)
                val_plan_hists = replanned_plan_hists

            parent_nodes_for_feasibility = [
                expanded_node_candidates[i]["parent_node"]
                for i in range(len(expanded_node_candidates))
            ]
            is_feasible = self._check_plan_batch_feasibility(
                plan_hists=val_plan_hists,
                root_obs_list=[
                    self._get_root_obs(parent_node)
                    for parent_node in parent_nodes_for_feasibility
                ],
                progress_obs_list=[
                    parent_node.obs for parent_node in parent_nodes_for_feasibility
                ],
                prefix_len_frames_list=[
                    self._get_prefix_len_frames_from_depth(parent_node.depth, seg_size)
                    for parent_node in parent_nodes_for_feasibility
                ],
                subplan_tail_depths=[
                    expanded_node_candidates[i]["depth"]
                    for i in range(len(expanded_node_candidates))
                ],
                seg_size=seg_size,
            )

            if non_terminal_cand_indices:
                assert expanded_node_updated_levels is not None, (
                    "expanded_node_updated_levels must be set for uncertainty sampling"
                )
                nt_parent_nodes = [
                    expanded_node_candidates[i]["parent_node"]
                    for i in non_terminal_cand_indices
                ]
                nt_val_plan_last = val_plan_hists[-1][:, non_terminal_cand_indices, :]
                nt_updated_levels = expanded_node_updated_levels[non_terminal_cand_indices]
                nt_idx_tensor = torch.tensor(
                    non_terminal_cand_indices, device=effective_obs_normalized.device
                )
                nt_obs_norm = effective_obs_normalized[nt_idx_tensor]

                unc_fast_result = self._run_fast_uncertainty_sampling(
                    parent_nodes=nt_parent_nodes,
                    val_plan_last_batch=nt_val_plan_last,
                    updated_levels=nt_updated_levels,
                    current_prefix_len_per_batch=np.array(
                        [
                            expanded_node_candidates[i]["depth"] * seg_size
                            for i in non_terminal_cand_indices
                        ],
                        dtype=int,
                    ),
                    seg_size=seg_size,
                    horizon=horizon,
                    conditions=conditions,
                    obs_normalized=nt_obs_norm,
                    opposite_tree=None,
                    obs_mean_np=obs_mean_np,
                    obs_std_np=obs_std_np,
                    opposite_trees=[
                        expanded_node_candidates[i]["opposite_tree"]
                        for i in non_terminal_cand_indices
                    ],
                    target_nodes_override=[
                        expanded_node_candidates[i]["target_node"]
                        for i in non_terminal_cand_indices
                    ],
                )

                for ii, i in enumerate(non_terminal_cand_indices):
                    unc_hist = unc_fast_result["unc_hists_per_cand"][ii]
                    fallback_unc_plan_hists[i] = unc_hist
                    uncertainty_plan_hists_per_candidate[i] = unc_hist
                    if unc_noise_levels_per_cand[i] is None:
                        unc_noise_levels_per_cand[i] = (
                            unc_fast_result["unc_noise_levels_per_cand"][ii]
                        )
                        unc_guidance_scale_per_cand[i] = (
                            unc_fast_result["unc_guidance_scale_per_cand"][ii]
                        )

            for i in range(len(is_feasible)):
                if is_feasible[i] and filtered_expanded_node_plan_hists[i] is None:
                    filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                    filtered_replanned_plan_hists[i] = val_plan_hists[:, :, i]

            for i in range(len(expanded_node_candidates)):
                if filtered_expanded_node_plan_hists[i] is None:
                    uncertainty_plan_hists_per_candidate[i] = None

            if None in filtered_expanded_node_plan_hists:
                mixed_stats["replanning_ms"] = round(
                    (time.time() - replanning_start_time) * 1000, 1
                )
                continue
            break

        assert expanded_node_plan_hists is not None
        assert replanned_plan_hists is not None

        final_is_feasible = [
            fh is not None for fh in filtered_expanded_node_plan_hists
        ]
        for i in range(len(filtered_expanded_node_plan_hists)):
            if filtered_expanded_node_plan_hists[i] is None:
                filtered_expanded_node_plan_hists[i] = expanded_node_plan_hists[:, :, i]
                filtered_replanned_plan_hists[i] = replanned_plan_hists[:, :, i]
                if uncertainty_plan_hists_per_candidate[i] is None:
                    uncertainty_plan_hists_per_candidate[i] = fallback_unc_plan_hists[i]

        expanded_node_plan_hists = torch.stack(filtered_expanded_node_plan_hists, dim=2)
        replanned_plan_hists = torch.stack(filtered_replanned_plan_hists, dim=2)

        final_best_plans = replanned_plan_hists[-1]
        achieved_infos, achieved_ts = self._check_achieved_bidir(
            expanded_node_candidates,
            final_best_plans,
        )
        tree_achieved_flags: dict[str, bool] = {}
        for i in range(len(achieved_infos)):
            if achieved_infos[i] == "Achieved":
                achieved_indices.append(i)
                tree_tag = expanded_node_candidates[i]["selected_tree"].tag
                tree_achieved_flags[tree_tag] = True

        terminal_cand_indices = [
            i
            for i, cand in enumerate(expanded_node_candidates)
            if cand["depth"] >= terminal_depth
        ]
        for i in terminal_cand_indices:
            values[i] = -np.inf
            unc_results[i] = {"U": np.inf}

        if non_terminal_cand_indices:
            nt_unc_hists = [
                uncertainty_plan_hists_per_candidate[i] for i in non_terminal_cand_indices
            ]
            for ii, i in enumerate(non_terminal_cand_indices):
                assert nt_unc_hists[ii] is not None, (
                    f"[Uncertainty] uncertainty_plan_hists_per_candidate[{i}] is None "
                    "after fallback injection — this should not happen."
                )
            nt_curr_plan_last = expanded_node_plan_hists[-1][:, non_terminal_cand_indices, :]
            nt_depths = [
                expanded_node_candidates[i]["depth"] for i in non_terminal_cand_indices
            ]
            nt_target_nodes = [
                expanded_node_candidates[i]["target_node"]
                for i in non_terminal_cand_indices
            ]
            nt_noise_levels = [
                unc_noise_levels_per_cand[i] for i in non_terminal_cand_indices
            ]
            nt_guidance_scales = [
                unc_guidance_scale_per_cand[i] for i in non_terminal_cand_indices
            ]

            unc_compute_result = self._compute_uncertainty_and_clusters(
                unc_hists_per_cand=nt_unc_hists,
                curr_plan_last_batch=nt_curr_plan_last,
                parent_nodes=nt_parent_nodes,
                node_depths=nt_depths,
                seg_size=seg_size,
                unc_noise_levels_per_cand=nt_noise_levels,
                unc_guidance_scale_per_cand=nt_guidance_scales,
                target_nodes=nt_target_nodes,
            )
            for ii, i in enumerate(non_terminal_cand_indices):
                values[i] = unc_compute_result["values"][ii]
                unc_results[i] = unc_compute_result["unc_results"][ii]
                cluster_subplans_per_candidate[i] = unc_compute_result["cluster_subplans"][ii]
                uncertainty_plan_hists_per_candidate[i] = (
                    unc_compute_result["filtered_unc_hists"][ii]
                )

        candidate_obses = []
        for i in range(len(expanded_node_candidates)):
            child_depth = expanded_node_candidates[i]["depth"]
            plan_hists_last_i = expanded_node_plan_hists[-1, :, i]
            obs_i = self._extract_obs_at_boundary(
                plan_hists_last_i.unsqueeze(1),
                depth=child_depth,
                seg_size=seg_size,
            )[0]
            candidate_obses.append(obs_i)

        is_kept = self._deduplicate_by_endpoint(
            expanded_node_candidates,
            candidate_obses,
            final_is_feasible,
        )

        expanded_node_infos: dict[str, dict] = {}
        for i in range(len(expanded_node_candidates)):
            candidate = expanded_node_candidates[i]
            name = candidate["name"]
            selected_tree = candidate["selected_tree"]
            global_name = f"{selected_tree.tag}:{name}"
            slot_index = int(name.split("-")[-1])
            parent_node = candidate["parent_node"]
            if not is_kept[i]:
                parent_node.mark_slot_permanently_dead(slot_index)
                continue
            parent_node._children_nodes[slot_index]["virtually_visited"] = False

            if global_name not in expanded_node_infos:
                expanded_node_infos[global_name] = candidate
                expanded_node_infos[global_name]["plan_history"].append([])

            value = values[i]
            plan_hist = (
                expanded_node_plan_hists[:, :, i]
                if not self.mcts_use_replan
                else replanned_plan_hists[:, :, i]
            )
            replanned_plan = replanned_plan_hists[-1, :, i]
            updated_level = (
                expanded_node_updated_levels[i : i + 1]
                if expanded_node_updated_levels is not None
                else None
            )

            if expanded_node_infos[global_name]["value"] is None or value > expanded_node_infos[global_name]["value"]:
                expanded_node_infos[global_name]["value"] = value
                expanded_node_infos[global_name]["replanned_plan"] = replanned_plan
                expanded_node_infos[global_name]["plan_history"][-1] = plan_hist
                expanded_node_infos[global_name]["expanded_plan_hist_frame"] = (
                    expanded_node_plan_hists[:, :, i]
                )
                expanded_node_infos[global_name]["replanned_plan_hist_frame"] = (
                    replanned_plan_hists[:, :, i] if self.mcts_use_replan else None
                )
                expanded_node_infos[global_name]["uncertainty_plan_hist_frame"] = (
                    uncertainty_plan_hists_per_candidate[i]
                )
                expanded_node_infos[global_name]["unc_diagnostics"] = unc_results.get(i)
                expanded_node_infos[global_name]["current_levels"] = updated_level
                expanded_node_infos[global_name]["cluster_subplans"] = (
                    cluster_subplans_per_candidate[i]
                )

        for global_name, info in expanded_node_infos.items():
            parent_node_for_expand = info["parent_node"]
            expand_kwargs = {
                k: v
                for k, v in info.items()
                if k
                not in (
                    "expanded_plan_hist_frame",
                    "replanned_plan_hist_frame",
                    "uncertainty_plan_hist_frame",
                    "unc_diagnostics",
                    "selected_tree",
                    "opposite_tree",
                    "parent_key",
                    "node",
                )
            }
            child_node = parent_node_for_expand.expand(**expand_kwargs)
            info["node"] = child_node

        tree_batches: dict[str, dict] = {}
        for global_idx, candidate in enumerate(expanded_node_candidates):
            selected_tree = candidate["selected_tree"]
            tree_tag = selected_tree.tag
            batch = tree_batches.setdefault(
                tree_tag,
                {
                    "tree": selected_tree,
                    "tree_indices": [],
                    "selected_nodes": [],
                    "expanded_node_infos": {},
                    "step_captures_by_name": {},
                    "had_achieved": False,
                },
            )
            batch["tree_indices"].append(global_idx)
            batch["selected_nodes"].append(selected_nodes[global_idx])
            if tree_achieved_flags.get(tree_tag):
                batch["had_achieved"] = True

        for global_name, info in expanded_node_infos.items():
            tree_tag = info["selected_tree"].tag
            tree_batches[tree_tag]["expanded_node_infos"][info["name"]] = info

        for tree_tag, batch in tree_batches.items():
            indices = batch.pop("tree_indices")
            batch["expanded_node_candidates"] = [
                expanded_node_candidates[i] for i in indices
            ]
            batch["values"] = np.asarray([values[i] for i in indices], dtype=np.float32)
            batch["achieved_indices"] = [
                local_i for local_i, global_i in enumerate(indices) if global_i in achieved_indices
            ]
            batch["expanded_node_plan_hists"] = expanded_node_plan_hists[:, :, indices, :]
            batch["replanned_plan_hists"] = replanned_plan_hists[:, :, indices, :]
            for global_i in indices:
                candidate = expanded_node_candidates[global_i]
                global_cand_key = f"{tree_tag}:{candidate['name']}"
                if global_cand_key in exp_sc_by_global_key:
                    batch["step_captures_by_name"][candidate["name"]] = exp_sc_by_global_key[
                        global_cand_key
                    ]

        self._expansion_step_captures_by_name = {}
        return {
            "expanded_node_infos": expanded_node_infos,
            "tree_batches": tree_batches,
            "mixed_stats": mixed_stats,
        }

    def _visualize_tree_final_plans(
        self,
        tree: MCTSTreeState,
        tree_batch: dict,
        start: torch.Tensor,
        goal: torch.Tensor,
        viz_step: int,
    ) -> None:
        """Visualize terminal candidate plans for a single tree batch."""
        viz_fp_t0 = time.time()
        viz_fp_node_ms = 0.0
        viz_fp_compare_ms = 0.0
        if self.viz_final_plans and tree_batch["expanded_node_candidates"]:
            expanded_node_candidates = tree_batch["expanded_node_candidates"]
            values = tree_batch["values"]
            achieved_indices = tree_batch["achieved_indices"]
            expanded_node_plan_hists = tree_batch["expanded_node_plan_hists"]
            replanned_plan_hists = tree_batch["replanned_plan_hists"]
            seg_size = self._require_current_plan_tokens() // self.sequence_dividing_factor
            terminal_depth_indices = [
                i
                for i, info in enumerate(expanded_node_candidates)
                if info["depth"] == tree.terminal_depth
            ]

            if tree_batch.get("had_achieved", False):
                visualize_indices = sorted(
                    set(terminal_depth_indices) | set(achieved_indices)
                )
            else:
                visualize_indices = sorted(terminal_depth_indices)

            if visualize_indices:
                terminal_values = values[visualize_indices]
                terminal_names = [
                    expanded_node_candidates[i]["name"] for i in visualize_indices
                ]
                terminal_expanded_plans = expanded_node_plan_hists[
                    -1, :, visualize_indices
                ]
                is_achieved_plan = [
                    i in achieved_indices for i in visualize_indices
                ]
                is_forward_tree = self._tree_uses_forward_prefix_semantics(tree)
                visualize_valid_frame_bounds = [
                    self._get_plan_viz_valid_frame_bounds(
                        expanded_node_candidates[i]["depth"],
                        seg_size,
                        terminal_expanded_plans.shape[0],
                        is_forward_tree,
                    )
                    for i in visualize_indices
                ]

                if not is_forward_tree:
                    terminal_expanded_plans = torch.flip(terminal_expanded_plans, [0])

                viz_node_t0 = time.time()
                self.visualize_node_value_plans(
                    is_achieved_plan,
                    viz_step,
                    terminal_values,
                    terminal_names,
                    terminal_expanded_plans,
                    start,
                    goal,
                    tag=tree.tag,
                    valid_frame_bounds=visualize_valid_frame_bounds,
                )
                viz_fp_node_ms = (time.time() - viz_node_t0) * 1000

                if self.viz_compare_expanded_to_value:
                    viz_cmp_t0 = time.time()
                    terminal_value_plans = replanned_plan_hists[
                        -1, :, visualize_indices
                    ]
                    if not is_forward_tree:
                        terminal_value_plans = torch.flip(terminal_value_plans, [0])
                    self.visualize_expanded_vs_value_plans(
                        is_achieved_plan,
                        terminal_names,
                        terminal_expanded_plans,
                        terminal_value_plans,
                        start,
                        goal,
                    )
                    viz_fp_compare_ms = (time.time() - viz_cmp_t0) * 1000

        viz_fp_total_ms = (time.time() - viz_fp_t0) * 1000
        self._tlog(
            "timing.viz_final_plans",
            {
                "tree_tag": tree.tag,
                "node_value_ms": round(viz_fp_node_ms, 1),
                "compare_ms": round(viz_fp_compare_ms, 1),
                "total_ms": round(viz_fp_total_ms, 1),
            },
            depth=1,
        )

    def _log_tree_batch_visualizations(
        self,
        tree_batch: dict,
        start: torch.Tensor,
        goal: torch.Tensor,
        loops: int,
        namespace: Optional[str] = None,
    ) -> None:
        tree: MCTSTreeState = tree_batch["tree"]
        expanded_node_infos = tree_batch["expanded_node_infos"]
        self._expansion_step_captures_by_name = tree_batch.get(
            "step_captures_by_name", {}
        )
        if expanded_node_infos:
            self._log_expanded_node_videos(
                expanded_node_infos,
                tree,
                start,
                goal,
                loops,
                namespace=namespace,
            )
        self._visualize_tree_final_plans(
            tree,
            tree_batch,
            start,
            goal,
            viz_step=self.global_search_num,
        )

    def _postprocess_tree_local_expansions(
        self,
        tree_batches: dict[str, dict],
        agent,
        envs,
        start: torch.Tensor,
        goal: torch.Tensor,
        loops: int,
        namespace: Optional[str] = None,
        log_videos_and_viz: bool = True,
    ) -> None:
        """Run tree-local backprop, state rollout, and visualization after mixed expansion."""
        original_step_captures = getattr(self, "_expansion_step_captures_by_name", {})
        try:
            for tree_batch in tree_batches.values():
                tree: MCTSTreeState = tree_batch["tree"]
                expanded_node_infos = tree_batch["expanded_node_infos"]

                backprop_start_time = time.time()
                created_parent_nodes = {}
                for info in expanded_node_infos.values():
                    created_parent_nodes[id(info["parent_node"])] = info["parent_node"]
                for selected_node in created_parent_nodes.values():
                    selected_node.backpropagate(
                        preserve_value=self.use_uncertainty_as_value,
                    )
                tree.backprop_time.append(time.time() - backprop_start_time)

                if tree_batch.get("had_achieved", False):
                    tree.achieved = True
                if expanded_node_infos:
                    tree.max_depth = max(
                        tree.max_depth,
                        max(info["depth"] for info in expanded_node_infos.values()),
                    )

                self._expansion_step_captures_by_name = tree_batch.get(
                    "step_captures_by_name", {}
                )
                if expanded_node_infos:
                    self._update_expanded_children_state(
                        tree,
                        expanded_node_infos,
                        agent,
                        envs,
                    )
                if log_videos_and_viz:
                    self._log_tree_batch_visualizations(
                        tree_batch,
                        start,
                        goal,
                        loops,
                        namespace=namespace,
                    )
        finally:
            self._expansion_step_captures_by_name = original_step_captures

    def _update_expanded_children_state(
        self,
        active_tree: MCTSTreeState,
        expanded_node_infos: dict[str, dict],
        agent,
        envs,
    ) -> None:
        """Populate obs/sim_state for freshly created child nodes."""
        if self.use_rollout:
            for info in expanded_node_infos.values():
                parent_node: "TreeNode" = info["parent_node"]
                _child: Optional["TreeNode"] = info.get("node")
                if _child is None:
                    continue

                plan_hist_last: torch.Tensor = info["plan_history"][-1][-1]
                plan_unnormalized: torch.Tensor = self._unnormalize_x(
                    plan_hist_last.unsqueeze(1)
                )

                seg_size: int = self._require_current_plan_tokens() // self.sequence_dividing_factor
                new_denoised_start: int = self._get_prefix_len_frames_from_depth(
                    parent_node.depth, seg_size
                )
                new_denoised_end: int = self._get_prefix_len_frames_from_depth(
                    parent_node.depth + 1, seg_size
                )

                _new_sim_state = self._rollout_leaf_plan(
                    leaf_plan_unnormalized=plan_unnormalized,
                    new_denoised_start_idx=new_denoised_start,
                    new_denoised_end_idx=new_denoised_end,
                    agent=agent,
                    envs=envs,
                    parent_sim_state=parent_node.sim_state,
                    is_backward=(not self._tree_uses_forward_prefix_semantics(active_tree)),
                )
                assert _new_sim_state is not None, "_new_sim_state is None"
                _child.sim_state = _new_sim_state
                _child.obs = np.concatenate(
                    [_new_sim_state["qpos"], _new_sim_state["qvel"]]
                )[self.obs_dim_indices]
                _parent_td = float(
                    self._compute_raw_state_temporal_dist_np(
                        parent_node.obs[None].astype(np.float32),
                        _child.obs[None].astype(np.float32),
                    )[0]
                )
                _child.cum_temporal_dist_from_root = (
                    float(getattr(parent_node, "cum_temporal_dist_from_root", 0.0))
                    + _parent_td
                )
        else:
            seg_size: int = self._require_current_plan_tokens() // self.sequence_dividing_factor
            for info in expanded_node_infos.values():
                parent_node: "TreeNode" = info["parent_node"]
                _child: Optional["TreeNode"] = info.get("node")
                if _child is None:
                    continue

                plan_hist_last: torch.Tensor = info["plan_history"][-1][-1]
                _child.obs = self._extract_obs_at_boundary(
                    plan_hist_last.unsqueeze(1),
                    depth=parent_node.depth + 1,
                    seg_size=seg_size,
                )[0]

                _child.sim_state = {}
                for k, v in parent_node.sim_state.items():
                    if isinstance(v, np.ndarray):
                        _child.sim_state[k] = v.copy()
                    else:
                        _child.sim_state[k] = v
                _child.sim_state["qpos"][:2] = _child.obs[self.pos_dim_indices]
                _parent_td = float(
                    self._compute_raw_state_temporal_dist_np(
                        parent_node.obs[None].astype(np.float32),
                        _child.obs[None].astype(np.float32),
                    )[0]
                )
                _child.cum_temporal_dist_from_root = (
                    float(getattr(parent_node, "cum_temporal_dist_from_root", 0.0))
                    + _parent_td
                )

    def _log_root_uncertainty_videos(
        self,
        root_uncertainty_infos: dict[str, dict],
        trees: List[MCTSTreeState],
        start: torch.Tensor,
        goal: torch.Tensor,
        loops: int,
        namespace: Optional[str] = None,
    ) -> None:
        """Log root fast-sampling uncertainty videos before the first expansion round."""
        if not self.viz_uncertain_next_subplan_last_obs or not root_uncertainty_infos:
            return

        _viz_namespace = (
            namespace.split("/", 1)[1]
            if namespace and namespace.startswith("validation/")
            else namespace
        )
        trees_by_tag = {tree.tag: tree for tree in trees}
        _v_start_np = start.cpu().numpy()[:, self.pos_dim_indices]
        _v_goal_np = goal.cpu().numpy()[:, self.pos_dim_indices]
        _v_hilp_fn = getattr(self, "_hilp_value_fn_instance", None)
        _v_tgt_vis_cache: dict = {}
        _v_obs_std_np = (
            self.data_std.cpu().numpy() if isinstance(self.data_std, torch.Tensor)
            else np.array(self.data_std)
        )[self.pos_dim_indices]

        for tree_tag, root_vinfo in root_uncertainty_infos.items():
            active_tree = trees_by_tag.get(tree_tag)
            if active_tree is None:
                continue
            _unc_hist = root_vinfo.get("uncertainty_plan_hist_frame")
            if _unc_hist is None:
                continue
            self._log_candidate_plan_video(
                f"{tree_tag}:root",
                root_vinfo,
                active_tree,
                _v_start_np,
                _v_goal_np,
                _v_hilp_fn,
                _v_tgt_vis_cache,
                {},
                _v_obs_std_np,
                loops,
                log_prefix="uncertainty_estimate",
                log_namespace=_viz_namespace,
                plan_hist_override=_unc_hist,
                is_uncertainty_viz=True,
            )

    def _log_expanded_node_videos(
        self,
        expanded_node_infos: dict[str, dict],
        active_tree: MCTSTreeState,
        start: torch.Tensor,
        goal: torch.Tensor,
        loops: int,
        namespace: Optional[str] = None,
    ) -> None:
        """Log per-candidate denoising videos for one expansion batch."""
        _viz_namespace = (
            namespace.split("/", 1)[1]
            if namespace and namespace.startswith("validation/")
            else namespace
        )
        _v_start_np = start.cpu().numpy()[:, self.pos_dim_indices]
        _v_goal_np = goal.cpu().numpy()[:, self.pos_dim_indices]
        _v_hilp_fn = getattr(self, "_hilp_value_fn_instance", None)
        _v_tgt_vis_cache: dict = {}
        _v_sc_by_name: dict = getattr(self, "_expansion_step_captures_by_name", {})
        _v_obs_std_np = (
            self.data_std.cpu().numpy() if isinstance(self.data_std, torch.Tensor)
            else np.array(self.data_std)
        )[self.pos_dim_indices]

        _viz_subplan_expand_ms = 0.0
        _viz_subplan_replan_ms = 0.0
        _viz_subplan_unc_ms = 0.0
        _viz_subplan_n = 0
        for _vname, _vinfo in (expanded_node_infos.items()):
            _viz_subplan_n += 1
            
            if self.viz_subplan_denoising:
                _viz_t0 = time.time()
                self._log_candidate_plan_video(
                    _vname,
                    _vinfo,
                    active_tree,
                    _v_start_np,
                    _v_goal_np,
                    _v_hilp_fn,
                    _v_tgt_vis_cache,
                    _v_sc_by_name,
                    _v_obs_std_np,
                    loops,
                    log_prefix="expanded",
                    log_namespace=_viz_namespace,
                    plan_hist_override=_vinfo.get("expanded_plan_hist_frame"),
                )
                _viz_subplan_expand_ms += (time.time() - _viz_t0) * 1000

            if self.mcts_use_replan and self.viz_replanning and _vinfo.get("replanned_plan_hist_frame") is not None:
                _viz_t0 = time.time()
                self._log_candidate_plan_video(
                    _vname,
                    _vinfo,
                    active_tree,
                    _v_start_np,
                    _v_goal_np,
                    _v_hilp_fn,
                    _v_tgt_vis_cache,
                    _v_sc_by_name,
                    _v_obs_std_np,
                    loops,
                    log_prefix="replanned",
                    log_namespace=_viz_namespace,
                    plan_hist_override=_vinfo.get("replanned_plan_hist_frame"),
                )
                _viz_subplan_replan_ms += (time.time() - _viz_t0) * 1000

            _unc_hist = _vinfo.get("uncertainty_plan_hist_frame")
            _should_log_unc = (
                self.viz_uncertain_next_subplan_last_obs
                and _unc_hist is not None
                and _vinfo.get("depth", 0) < active_tree.terminal_depth
            )
            if _should_log_unc:
                _viz_t0 = time.time()
                self._log_candidate_plan_video(
                    _vname,
                    _vinfo,
                    active_tree,
                    _v_start_np,
                    _v_goal_np,
                    _v_hilp_fn,
                    _v_tgt_vis_cache,
                    _v_sc_by_name,
                    _v_obs_std_np,
                    loops,
                    log_prefix="uncertainty_estimate",
                    log_namespace=_viz_namespace,
                    plan_hist_override=_vinfo["uncertainty_plan_hist_frame"],
                    is_uncertainty_viz=True,
                )
                _viz_subplan_unc_ms += (time.time() - _viz_t0) * 1000

        if _viz_subplan_n > 0 or self.viz_subplan_denoising:
            self._tlog(
                "timing.viz_subplan_denoising",
                {
                    "tree_tag": active_tree.tag,
                    "n_candidates": _viz_subplan_n,
                    "expand_ms": round(_viz_subplan_expand_ms, 1),
                    "replan_ms": round(_viz_subplan_replan_ms, 1),
                    "uncertainty_ms": round(_viz_subplan_unc_ms, 1),
                    "total_ms": round(
                        _viz_subplan_expand_ms
                        + _viz_subplan_replan_ms
                        + _viz_subplan_unc_ms,
                        1,
                    ),
                },
                depth=1,
            )

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

    def _run_multi_tree_online_hamiltonian_planner(
        self,
        horizon: int,
        conditions: Optional[Any],
        start: torch.Tensor,
        goal: torch.Tensor,
        initial_sim_state: dict,
        waypoint_xys: Optional[np.ndarray],
        fixed_solver_result: Optional[dict[str, Any]],
        agent,
        envs,
        namespace: str,
    ) -> dict:
        if not self.use_uncertainty_as_value:
            raise NotImplementedError("use_anchor_planner currently requires use_uncertainty_as_value=True")

        start_obs = start.detach().cpu().numpy()[0, self.obs_dim_indices]
        goal_obs = goal.detach().cpu().numpy()[0]
        planner_state = self._init_multi_tree_planner_state(
            horizon=horizon,
            start_obs=start_obs,
            goal_obs=goal_obs,
            initial_sim_state=initial_sim_state,
            waypoint_xys=waypoint_xys,
            fixed_solver_result=fixed_solver_result,
        )

        global_search_pbar = tqdm(
            total=self.mctd_max_search_num,
            desc="MCTS (multi-tree)",
            leave=True,
            dynamic_ncols=True,
        )
        root_uncertainty_infos: dict[str, dict] = {}
        if self.use_cluster_subplan_as_expansion:
            root_uncertainty_infos = self._ensure_uncertainty_roots_initialized_multi(
                planner_state=planner_state,
                horizon=horizon,
                conditions=conditions,
            )
            self._log_root_uncertainty_videos(
                root_uncertainty_infos,
                planner_state["trees"],
                start,
                goal,
                loops=0,
                namespace=namespace,
            )

        loops = 0
        termination_reason = "unknown"

        def _multi_tree_pre_round_check() -> Optional[str]:
            if self.time_limit is not None and (time.time() - self.start_time > self.time_limit):
                return "time_limit"
            if self.global_search_num >= self.mctd_max_search_num:
                return "search_budget"
            if self._all_multi_tree_anchors_satisfied(planner_state):
                return "all_anchors_satisfied"
            return None

        def _multi_tree_run_round(current_loop: int) -> dict:
            selected_parent_infos = self._select_multi_tree_expansion_parents(planner_state)
            if not selected_parent_infos:
                return {"stop": True, "termination_reason": "no_selected_parents"}

            self.global_search_num += len(selected_parent_infos)
            global_search_pbar.update(len(selected_parent_infos))

            mixed_round_result = self._run_global_uncertainty_expansion_round(
                selected_parent_infos=selected_parent_infos,
                horizon=horizon,
                conditions=conditions,
                start=None,
                goal=None,
            )
            self._postprocess_tree_local_expansions(
                mixed_round_result["tree_batches"],
                agent,
                envs,
                start,
                goal,
                current_loop,
                namespace=namespace,
                log_videos_and_viz=False,
            )
            self._update_multi_tree_direct_bridges_from_expansions(
                planner_state=planner_state,
                tree_batches=mixed_round_result["tree_batches"],
            )
            self._update_multi_tree_meeting_targets_from_expansions(
                planner_state=planner_state,
                tree_batches=mixed_round_result["tree_batches"],
            )
            accepted_candidates = self._select_multi_tree_round_meetings(
                planner_state=planner_state,
                expanded_node_infos=mixed_round_result["expanded_node_infos"],
            )
            self._accept_multi_tree_round_meetings(
                planner_state=planner_state,
                accepted_candidates=accepted_candidates,
                loops=current_loop,
            )
            self._compute_tentative_hamiltonian_solution(planner_state)
            original_step_captures = getattr(self, "_expansion_step_captures_by_name", {})
            try:
                for tree_batch in mixed_round_result["tree_batches"].values():
                    self._log_tree_batch_visualizations(
                        tree_batch,
                        start,
                        goal,
                        current_loop,
                        namespace=namespace,
                    )
            finally:
                self._expansion_step_captures_by_name = original_step_captures
            return {"stop": False}

        try:
            loop_result = self._run_anchor_policy_loop(
                pre_round_check=_multi_tree_pre_round_check,
                run_round=_multi_tree_run_round,
            )
            loops = int(loop_result["loops"])
            termination_reason = str(loop_result["termination_reason"])
        finally:
            self._close_anchor_search_progress_bars(
                list(planner_state["trees"]),
                global_search_pbar=global_search_pbar,
            )

        selected_plan_bundle = self._assemble_multi_tree_plan_bundle(planner_state)
        accepted_pairs_text = ", ".join(
            f"{anchor_short_label(int(a), len(planner_state['trees']))}-"
            f"{anchor_short_label(int(b), len(planner_state['trees']))}"
            for a, b in sorted(planner_state["accepted_pair_edges"].keys())
        ) or "none"
        solver_route_text = planner_state.get("tentative_solver_result", {}).get(
            "route_text",
            "unknown",
        )
        print(
            "[MCTD debug] Multi-tree termination | "
            f"reason={termination_reason} | loops={loops} | "
            f"parents={self.global_search_num}/{self.mctd_max_search_num} | "
            f"accepted={accepted_pairs_text} | tentative={solver_route_text}",
            flush=True,
        )
        return {
            "planner_kind": "anchor_multi_tree",
            "planner_state": planner_state,
            "trees": planner_state["trees"],
            "loops": loops,
            "selected_plan_bundle": selected_plan_bundle,
            "best_node": None,
            "active_tree": None,
            "last_reverse_output": False,
        }

    def _node_path_label(
        self,
        expanding_node: "TreeNode",
        active_tree: Optional["MCTSTreeState"],
        target_node: Optional["TreeNode"],
    ) -> str:
        """Build a human-readable path label for plan visualization.

        All active planning paths now use anchor-name-prefixed labels.

        Args:
            expanding_node: The TreeNode just expanded.
            active_tree: Tree owning `expanding_node`.
            target_node: The node in the opposite tree being targeted (may be None).

        Returns:
            Label string for video/log keys.
        """
        exp_parts = expanding_node.name.split("-")
        if exp_parts:
            exp_parts[-1] = f"({exp_parts[-1]})"
        expanding_label = "-".join(exp_parts) if exp_parts else "(?)"
        source_anchor = self._node_anchor_name(
            expanding_node,
            getattr(active_tree, "anchor_name", "?") if active_tree is not None else "?",
        )
        if target_node is None:
            return f"{source_anchor}:{expanding_label}"
        target_anchor = self._node_anchor_name(target_node)
        target_label = "-".join(target_node.name.split("-"))
        return f"{source_anchor}:{expanding_label}__{target_anchor}:{target_label}"
