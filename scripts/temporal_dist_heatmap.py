#!/usr/bin/env python3
"""Temporal-distance heatmap + sampled-graph waypoint-route visualizer.

This script builds the planner config from checkpoint metadata, initializes a
single OGBench maze env to obtain an in-distribution HILP reference state, and
renders one PNG with two panels:

1. Temporal-distance heatmap computed from HILP embedding distance via
   ``DiffusionForcingPlanning.emb_dist_to_temporal_dist`` with the existing
   guidance gradient field overlay.
2. A sampled-graph panel colored by shortest distance from the snapped goal
   node, with task override start/goal/waypoints and a constrained Hamiltonian
   route overlay on the cached graph.

The diffusion checkpoint weights themselves are not loaded because this
visualization only depends on the planner config and HILP utilities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Circle, Polygon, Rectangle
from omegaconf import OmegaConf, open_dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from algorithms.diffusion_forcing.sampled_graph_estimator import (
    build_or_load_sampled_graph_cache_from_npz,
    extract_shortest_path_submatrix,
    query_distinct_nearest_nodes,
    query_shortest_path_between_node_indices,
)
from experiments.exp_planning import PlanningExperiment
from utils.route_metric_utils import (
    anchor_short_label as _shared_anchor_short_label,
    compute_pairwise_temporal_distance_matrix as _shared_compute_pairwise_temporal_distance_matrix,
    solve_fixed_endpoint_hamiltonian_path as _shared_solve_fixed_endpoint_hamiltonian_path,
)


OGBENCH_ENVS = {
    "pointmaze-medium-v0",
    "pointmaze-large-v0",
    "pointmaze-giant-v0",
    "pointmaze-teleport-v0",
    "antmaze-medium-v0",
    "antmaze-large-v0",
    "antmaze-giant-v0",
    "antmaze-teleport-v0",
}


def _normalize_dataset_config_name(name: str) -> str:
    name = str(name)
    if not name.startswith("og_"):
        return name
    stripped = name[3:]
    stripped_yaml = _REPO_ROOT / "configurations" / "dataset" / f"{stripped}.yaml"
    return stripped if stripped_yaml.is_file() else name


def _load_training_metadata(ckpt_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    training_cfg_path = ckpt_path.parent / "training_config.yaml"
    algo_overrides: dict[str, Any]
    dataset_meta: dict[str, Any]

    if training_cfg_path.is_file():
        loaded = OmegaConf.to_container(OmegaConf.load(training_cfg_path), resolve=True)
        loaded = loaded or {}
        algo_overrides = dict(loaded.get("algorithm", {}) or {})
        dataset_meta = dict(loaded.get("dataset", {}) or {})
    else:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        algo_overrides = dict(ckpt.get("training_hparams", {}) or {})
        dataset_meta = {}

    dataset_config_name = dataset_meta.get("config")
    if dataset_config_name is None:
        dataset_config_name = algo_overrides.get("dataset_config")
    if dataset_config_name is None:
        dataset_config_name = algo_overrides.get("train_dataset_config")
    if dataset_config_name is None:
        raise ValueError(
            f"Could not determine dataset config from {training_cfg_path} or checkpoint training_hparams."
        )

    dataset_config_name = _normalize_dataset_config_name(str(dataset_config_name))
    return algo_overrides, dataset_config_name, dataset_meta


def _remove_eval_only_training_keys(algo_overrides: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(algo_overrides)
    diffusion_cfg = dict(cleaned.get("diffusion", {}) or {})
    diffusion_cfg.pop("sampling_timesteps", None)
    if diffusion_cfg:
        cleaned["diffusion"] = diffusion_cfg
    return cleaned


def _build_root_cfg(
    dataset_config_name: str,
    algo_overrides: dict[str, Any],
    dataset_meta: dict[str, Any],
    task_override_path: Optional[str],
    task_override_waypoint_group_idx: Optional[int],
) -> Any:
    root_cfg = OmegaConf.load(_REPO_ROOT / "configurations" / "config.yaml")
    experiment_cfg = OmegaConf.load(_REPO_ROOT / "configurations" / "experiment" / "base_pytorch.yaml")
    dataset_cfg = OmegaConf.load(_REPO_ROOT / "configurations" / "dataset" / f"{dataset_config_name}.yaml")
    algorithm_cfg = OmegaConf.load(_REPO_ROOT / "configurations" / "algorithm" / "df_planning.yaml")
    algorithm_cfg = OmegaConf.merge(algorithm_cfg, OmegaConf.create(_remove_eval_only_training_keys(algo_overrides)))

    with open_dict(root_cfg):
        root_cfg.name = "temporal_distance_viz"
        root_cfg.debug = False
        root_cfg.cluster = None
        root_cfg.experiment = experiment_cfg
        root_cfg.dataset = dataset_cfg
        root_cfg.algorithm = algorithm_cfg
        root_cfg.wandb.mode = "disabled"

    with open_dict(root_cfg.experiment):
        root_cfg.experiment._name = "base_pytorch"

    with open_dict(root_cfg.dataset):
        root_cfg.dataset._name = dataset_config_name
        if dataset_meta.get("episode_len") is not None:
            root_cfg.dataset.episode_len = int(dataset_meta["episode_len"])
        if dataset_meta.get("jump") is not None:
            root_cfg.dataset.jump = int(dataset_meta["jump"])

    with open_dict(root_cfg.algorithm):
        root_cfg.algorithm._name = "df_planning"
        root_cfg.algorithm.dataset_config = dataset_config_name
        root_cfg.algorithm.train_dataset_config = dataset_config_name
        root_cfg.algorithm.task_override_path = task_override_path
        root_cfg.algorithm.task_override_waypoint_group_idx = task_override_waypoint_group_idx

    root_jump = (
        algo_overrides.get("jump")
        if algo_overrides.get("jump") is not None
        else dataset_meta.get("jump", 1)
    )
    with open_dict(root_cfg):
        root_cfg.jump = int(root_jump)

    return root_cfg


def _build_algo(root_cfg: Any):
    exp = PlanningExperiment(root_cfg, logger=None, ckpt_path=None)
    exp._ensure_algo_cfg_fallbacks()
    exp.algo = exp._build_algo()
    return exp.algo


def _is_grid_env(env_id: str) -> bool:
    return (
        "maze2d" in env_id
        or "diagonal2d" in env_id
        or "pointmaze" in env_id
        or "antmaze" in env_id
    )


def _get_maze_grid(env_id: str) -> list[str]:
    if "medium" in env_id:
        maze_string = "########\\#OO#OOO#\\#OOOO#O#\\###O#OO#\\##OOOO##\\#OO#O#O#\\#OO#OOO#\\########"
    elif "large" in env_id:
        maze_string = "#########\\#OOOOO#O#\\#O#O#OOO#\\#O#O#####\\#OOO#OOO#\\###O###O#\\#OOOOOOO#\\#O###O###\\#OOO#OOO#\\#O#O#O#O#\\#OOOOO#O#\\#########"
    elif "giant" in env_id:
        maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    elif "teleport" in env_id:
        maze_string = "#########\\#O##OOOO#\\#OOOO##O#\\#O##O##O#\\#OO#OOOO#\\#OO######\\##OOOOOO#\\#O#O###O#\\##OOOOOO#\\#OOO#####\\#O#OOOOO#\\#########"
    else:
        raise ValueError(f"Unsupported maze env for layout: {env_id}")
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def _to_plot_coords(env_id: str, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if env_id in OGBENCH_ENVS:
        return pts / 4.0 + 1.0
    return pts


def _configure_maze_axes(ax, maze_grid: Optional[list[str]]) -> None:
    ax.set_aspect("equal")
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(4)
    if maze_grid is not None:
        ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
        ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
        ax.set_xlim(0.5, len(maze_grid) + 0.5)
        ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
        ax.grid(True, color="white", linewidth=4)
        ax.set_axisbelow(True)


def _draw_maze_walls(ax, maze_grid: Optional[list[str]], alpha: float = 1.0, zorder: int = 5) -> None:
    if maze_grid is None:
        return
    for i, row in enumerate(maze_grid):
        for j, cell in enumerate(row):
            if cell == "#":
                ax.add_patch(
                    Rectangle(
                        (i + 0.5, j + 0.5),
                        1.0,
                        1.0,
                        facecolor="black",
                        edgecolor="black",
                        alpha=alpha,
                        zorder=zorder,
                    )
                )


def _normalize_vector_field(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mag = np.sqrt(u ** 2 + v ** 2)
    safe_mag = np.where(mag > eps, mag, 1.0)
    u_n = np.where(mag > eps, u / safe_mag, 0.0)
    v_n = np.where(mag > eps, v / safe_mag, 0.0)
    return u_n, v_n


def _build_single_goal_obs(algo, goal_xy: np.ndarray) -> np.ndarray:
    obs_dim = len(algo.obs_dim_indices)
    goal_obs = np.zeros(obs_dim, dtype=np.float32)

    ref_obs = getattr(algo, "_hilp_ref_obs", None)
    if ref_obs is not None and len(ref_obs) > max(algo.obs_dim_indices):
        goal_obs = np.asarray(ref_obs, dtype=np.float32)[algo.obs_dim_indices].copy()

    goal_obs[np.asarray(algo.pos_dim_indices, dtype=np.int64)] = np.asarray(goal_xy, dtype=np.float32)
    return goal_obs


def _compute_pairwise_temporal_distance_matrix(
    algo,
    src_xys: np.ndarray,
    dst_xys: np.ndarray,
    gamma: float,
) -> np.ndarray:
    return _shared_compute_pairwise_temporal_distance_matrix(algo, src_xys, dst_xys, gamma)


def _get_world_extent(algo) -> tuple[float, float, float, float]:
    if _is_grid_env(algo.env_id):
        maze_grid = _get_maze_grid(algo.env_id)
        height = len(maze_grid)
        width = len(maze_grid[0])
        x_min = (0.5 - 1.0) * 4.0
        x_max = (height + 0.5 - 1.0) * 4.0
        y_min = (0.5 - 1.0) * 4.0
        y_max = (width + 0.5 - 1.0) * 4.0
        return x_min, x_max, y_min, y_max

    data_mean = algo.data_mean.cpu().numpy() if isinstance(algo.data_mean, torch.Tensor) else np.asarray(algo.data_mean)
    data_std = algo.data_std.cpu().numpy() if isinstance(algo.data_std, torch.Tensor) else np.asarray(algo.data_std)
    pos_mean = data_mean[algo.pos_dim_indices]
    pos_std = data_std[algo.pos_dim_indices]
    return (
        float(pos_mean[0] - 3.0 * pos_std[0]),
        float(pos_mean[0] + 3.0 * pos_std[0]),
        float(pos_mean[1] - 3.0 * pos_std[1]),
        float(pos_mean[1] + 3.0 * pos_std[1]),
    )


def _compute_temporal_distance_heatmap(
    algo,
    goal_xy: np.ndarray,
    grid_res: int,
    gamma: float,
) -> dict[str, np.ndarray]:
    x_min, x_max, y_min, y_max = _get_world_extent(algo)
    xs = np.linspace(x_min, x_max, grid_res, dtype=np.float32)
    ys = np.linspace(y_min, y_max, grid_res, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)
    grid_xy = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)

    temporal_dist = _compute_pairwise_temporal_distance_matrix(
        algo,
        src_xys=grid_xy,
        dst_xys=np.asarray(goal_xy, dtype=np.float32).reshape(1, 2),
        gamma=gamma,
    ).reshape(x_grid.shape).astype(np.float32)
    return {"X": x_grid, "Y": y_grid, "values": temporal_dist}


def _initialize_hilp_reference_obs(algo, task_id: int) -> Optional[dict[str, np.ndarray]]:
    if algo.env_id not in OGBENCH_ENVS:
        return None

    from stable_baselines3.common.vec_env import DummyVecEnv

    maze_type = algo.env_id.split("-")[1]
    make_maze_env = algo._get_ogbench_make_maze_env()

    if "pointmaze" in algo.env_id:
        env_fn = lambda: make_maze_env("point", "maze", maze_type=maze_type, width=200, height=200)
    elif "antmaze" in algo.env_id:
        env_fn = lambda: make_maze_env("ant", "maze", maze_type=maze_type, width=200, height=200)
    else:
        raise RuntimeError(f"Unsupported OGBench env: {algo.env_id}")

    envs = DummyVecEnv([env_fn])
    try:
        algo._prepare_ogbench_env(envs.envs[0])
        envs.envs[0].set_task(int(task_id))
        envs.reset()
        sim_state = algo._get_sim_state(envs)
        if sim_state is None:
            raise RuntimeError("Failed to extract qpos/qvel from environment after reset.")
        ref_obs = np.concatenate([sim_state["qpos"], sim_state["qvel"]], axis=0)[: algo.hilp_obs_dim].astype(np.float32)
        algo._hilp_ref_obs = ref_obs

        reset_info = {}
        if hasattr(envs, "reset_infos") and len(envs.reset_infos) > 0:
            reset_info = envs.reset_infos[0] or {}
        start_xy = np.asarray(
            reset_info.get("start_xy", sim_state["qpos"][:2]),
            dtype=np.float32,
        ).reshape(-1)[:2]
        goal_xy = reset_info.get("goal_xy", reset_info.get("goal"))
        if goal_xy is None:
            goal_xy_out = None
        else:
            goal_xy_out = np.asarray(goal_xy, dtype=np.float32).reshape(-1)[:2]
        waypoints_xy = np.asarray(
            reset_info.get("waypoints_xy", np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        )
        if waypoints_xy.size == 0:
            waypoints_xy = np.zeros((0, 2), dtype=np.float32)
        else:
            waypoints_xy = waypoints_xy.reshape(-1, 2)
        waypoint_xy_groups = []
        for waypoint_xy_group in reset_info.get("waypoint_xy_groups", []):
            waypoint_xy_group = np.asarray(waypoint_xy_group, dtype=np.float32)
            if waypoint_xy_group.size == 0:
                waypoint_xy_groups.append(np.zeros((0, 2), dtype=np.float32))
            else:
                waypoint_xy_groups.append(waypoint_xy_group.reshape(-1, 2))
        return {
            "start_xy": start_xy,
            "goal_xy": goal_xy_out,
            "waypoints_xy": waypoints_xy,
            "waypoint_xy_groups": waypoint_xy_groups,
            "active_waypoint_group_idx": reset_info.get("active_waypoint_group_idx"),
            "task_name": reset_info.get("task_name"),
        }
    finally:
        envs.close()


def _resolve_sampled_graph_cfg(algo, args) -> dict[str, float | int | Path]:
    cfg = algo.cfg
    sample_ratio = float(
        args.sample_ratio
        if args.sample_ratio is not None
        else cfg.get("sampled_graph_sample_ratio", 0.001)
    )
    edge_radius = float(
        args.edge_radius
        if args.edge_radius is not None
        else cfg.get("sampled_graph_edge_radius", 3.0)
    )
    graph_seed = int(
        args.graph_seed
        if args.graph_seed is not None
        else cfg.get("sampled_graph_seed", 42)
    )
    cache_dir_raw = cfg.get("sampled_graph_save_dir", cfg.get("kde_save_dir", "~/.ogbench/data"))
    cache_dir = Path(str(cache_dir_raw)).expanduser()
    return {
        "sample_ratio": sample_ratio,
        "edge_radius": edge_radius,
        "graph_seed": graph_seed,
        "cache_dir": cache_dir,
    }


def _draw_star(center: np.ndarray, radius: float, *, color: str = "black", zorder: int = 13) -> Polygon:
    angles = np.linspace(0.0, 2 * np.pi, 5, endpoint=False) + 5 * np.pi / 10
    inner_radius = radius / 2.0
    points = []
    for angle in angles:
        points.append(
            [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
            ]
        )
        points.append(
            [
                center[0] + inner_radius * np.cos(angle + np.pi / 5),
                center[1] + inner_radius * np.sin(angle + np.pi / 5),
            ]
        )
    return Polygon(np.asarray(points, dtype=np.float32), closed=True, facecolor=color, edgecolor=color, zorder=zorder)


def _plot_task_icons(
    ax,
    env_id: str,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    waypoints_xy: np.ndarray,
) -> None:
    start_plot = _to_plot_coords(env_id, np.asarray(start_xy, dtype=np.float32).reshape(1, 2))[0]
    goal_plot = _to_plot_coords(env_id, np.asarray(goal_xy, dtype=np.float32).reshape(1, 2))[0]
    waypoint_plot = _to_plot_coords(env_id, np.asarray(waypoints_xy, dtype=np.float32).reshape(-1, 2))

    ax.add_patch(Circle((start_plot[0], start_plot[1]), 0.16, facecolor="white", edgecolor="black", linewidth=1.2, zorder=12))
    ax.add_patch(Circle((start_plot[0], start_plot[1]), 0.08, facecolor="black", edgecolor="black", zorder=13))
    ax.add_patch(Circle((goal_plot[0], goal_plot[1]), 0.16, facecolor="white", edgecolor="black", linewidth=1.2, zorder=12))
    ax.add_patch(_draw_star(goal_plot, radius=0.08, color="black", zorder=13))

    if waypoint_plot.size == 0:
        return
    for idx, waypoint in enumerate(waypoint_plot, start=1):
        ax.scatter(
            waypoint[0],
            waypoint[1],
            marker="D",
            s=110,
            facecolors="white",
            edgecolors="seagreen",
            linewidths=1.5,
            zorder=14,
            label="Waypoint" if idx == 1 else "_nolegend_",
        )
        ax.scatter(
            waypoint[0],
            waypoint[1],
            marker="o",
            s=20,
            c="seagreen",
            zorder=15,
            label="_nolegend_",
        )
        ax.text(
            waypoint[0] + 0.14,
            waypoint[1] + 0.14,
            f"W{idx}",
            fontsize=8,
            color="seagreen",
            weight="bold",
            zorder=16,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )


def _build_anchor_queries(
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
        "n_waypoints": n_waypoints,
    }


def _anchor_short_label(anchor_idx: int, n_anchors: int) -> str:
    return _shared_anchor_short_label(anchor_idx, n_anchors)


def _solve_fixed_endpoint_hamiltonian_path(anchor_shortest_dists: np.ndarray) -> dict[str, Any]:
    return _shared_solve_fixed_endpoint_hamiltonian_path(anchor_shortest_dists)


def _expand_anchor_route_segments(
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


def _build_straight_anchor_route_segments(
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
        step_distance = float(
            np.linalg.norm(path_xy[1] - path_xy[0], ord=2)
        )
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


def _plot_route_segments(
    ax,
    env_id: str,
    route_segments: dict[str, Any],
    *,
    label: str = "Hamiltonian route",
) -> None:
    if not bool(route_segments.get("reachable", False)):
        return

    first_segment = True
    for segment in route_segments.get("segments", []):
        path_xy = np.asarray(segment.get("path_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if len(path_xy) == 0:
            continue
        path_plot = _to_plot_coords(env_id, path_xy)
        ax.plot(
            path_plot[:, 0],
            path_plot[:, 1],
            color="black",
            linewidth=3.6,
            alpha=0.82,
            zorder=6,
            label="_nolegend_",
        )
        ax.plot(
            path_plot[:, 0],
            path_plot[:, 1],
            color="white",
            linewidth=2.2,
            alpha=0.96,
            zorder=7,
            label=label if first_segment else "_nolegend_",
        )
        first_segment = False


def _plot_temporal_panel(
    ax,
    fig,
    env_id: str,
    heatmap: dict[str, np.ndarray],
    grad_field: Optional[dict[str, np.ndarray]],
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    waypoints_xy: np.ndarray,
    route_solution: dict[str, Any],
    route_segments: dict[str, Any],
    task_name: Optional[str],
) -> None:
    maze_grid = _get_maze_grid(env_id) if _is_grid_env(env_id) else None
    _configure_maze_axes(ax, maze_grid)

    x_plot = _to_plot_coords(env_id, heatmap["X"])
    y_plot = _to_plot_coords(env_id, heatmap["Y"])
    mesh = ax.pcolormesh(
        x_plot,
        y_plot,
        heatmap["values"],
        shading="auto",
        cmap="viridis_r",
        alpha=0.85,
        zorder=1,
        norm=Normalize(vmin=float(heatmap["values"].min()), vmax=float(heatmap["values"].max())),
    )
    fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02, label="temporal distance")

    if grad_field is not None:
        x_g = _to_plot_coords(env_id, grad_field["x_grid"])
        y_g = _to_plot_coords(env_id, grad_field["y_grid"])
        grads = grad_field["hilp_grads"]
        far_mask = grad_field.get("far_mask_grid")
        if far_mask is None:
            far_mask = np.zeros(grads.shape[:2], dtype=bool)
        u = grads[:, :, 0]
        v = grads[:, :, 1]
        u_n, v_n = _normalize_vector_field(u, v)

        flat_hilp = (~far_mask).reshape(-1)
        flat_far = far_mask.reshape(-1)
        flat_colors = np.empty((u_n.size, 4), dtype=float)
        flat_colors[flat_hilp] = to_rgba("crimson", alpha=0.6)
        flat_colors[flat_far] = to_rgba("steelblue", alpha=0.6)

        ax.quiver(
            x_g.reshape(-1),
            y_g.reshape(-1),
            u_n.reshape(-1),
            v_n.reshape(-1),
            color=flat_colors,
            angles="xy",
            scale_units="xy",
            scale=2.0,
            pivot="mid",
            width=0.004,
            zorder=4,
        )
        if flat_hilp.any():
            ax.scatter([], [], c="crimson", marker=r"$\rightarrow$", s=120, alpha=0.6, label="HILP grad")
        if flat_far.any():
            ax.scatter([], [], c="steelblue", marker=r"$\rightarrow$", s=120, alpha=0.6, label="RMSE grad (far)")

    _plot_route_segments(ax, env_id, route_segments, label="Temporal Hamiltonian route")
    _draw_maze_walls(ax, maze_grid, alpha=1.0, zorder=5)
    _plot_task_icons(ax, env_id, start_xy, goal_xy, waypoints_xy)
    task_prefix = f"{task_name}  " if task_name else ""
    route_text = str(route_solution.get("route_text", ""))
    if bool(route_solution.get("feasible", False)) and bool(route_segments.get("reachable", False)):
        route_summary = f"route={route_text}  total temporal={float(route_solution['total_cost']):.3f}"
    else:
        route_summary = "route=no feasible Hamiltonian path"
    ax.set_title(
        f"{task_prefix}Temporal Distance Heatmap + Grad Field\n"
        f"{route_summary}"
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def _plot_sample_panel(
    ax,
    fig,
    env_id: str,
    sampled_xy: np.ndarray,
    shortest_dists: np.ndarray,
    anchor_xys: np.ndarray,
    anchor_node_xys: np.ndarray,
    anchor_node_dists: np.ndarray,
    sample_ratio: float,
    edge_radius: float,
    n_total: int,
    n_reachable: int,
    route_solution: dict[str, Any],
    route_segments: dict[str, Any],
    task_name: Optional[str],
) -> None:
    maze_grid = _get_maze_grid(env_id) if _is_grid_env(env_id) else None
    _configure_maze_axes(ax, maze_grid)
    sampled_plot = _to_plot_coords(env_id, sampled_xy)
    shortest_dists = np.asarray(shortest_dists, dtype=np.float32)
    finite_mask = np.isfinite(shortest_dists)
    unreachable_mask = ~finite_mask

    color_handle = None
    if finite_mask.any():
        finite_vals = shortest_dists[finite_mask]
        vmin = float(finite_vals.min())
        vmax = float(finite_vals.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        color_handle = ax.scatter(
            sampled_plot[finite_mask, 0],
            sampled_plot[finite_mask, 1],
            s=48,
            c=finite_vals,
            cmap="viridis_r",
            norm=Normalize(vmin=vmin, vmax=vmax),
            alpha=0.8,
            linewidths=0.0,
            zorder=3,
        )
        fig.colorbar(
            color_handle,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            label="shortest distance from goal node",
        )
    if unreachable_mask.any():
        ax.scatter(
            sampled_plot[unreachable_mask, 0],
            sampled_plot[unreachable_mask, 1],
            s=48,
            c="dimgray",
            alpha=0.75,
            linewidths=0.0,
            zorder=2,
            label="Unreachable",
        )
    anchor_plot = _to_plot_coords(env_id, np.asarray(anchor_xys, dtype=np.float32))
    anchor_node_plot = _to_plot_coords(env_id, np.asarray(anchor_node_xys, dtype=np.float32))
    goal_anchor_idx = int(len(anchor_xys) - 1)

    _plot_route_segments(ax, env_id, route_segments, label="Hamiltonian route")
    waypoint_connector_labeled = False
    waypoint_node_labeled = False
    for anchor_idx, (actual_plot, node_plot) in enumerate(zip(anchor_plot, anchor_node_plot)):
        if anchor_idx == 0:
            connector_color = "springgreen"
            node_edge_color = "springgreen"
            node_marker = "o"
            connector_label = "Start connector"
            node_label = "Snapped start node"
            node_size = 150
        elif anchor_idx == goal_anchor_idx:
            connector_color = "deepskyblue"
            node_edge_color = "deepskyblue"
            node_marker = "o"
            connector_label = "Goal connector"
            node_label = "Snapped goal node"
            node_size = 150
        else:
            connector_color = "seagreen"
            node_edge_color = "seagreen"
            node_marker = "D"
            connector_label = "Waypoint connector" if not waypoint_connector_labeled else "_nolegend_"
            node_label = "Snapped waypoint node" if not waypoint_node_labeled else "_nolegend_"
            node_size = 120
            waypoint_connector_labeled = True
            waypoint_node_labeled = True

        ax.plot(
            [actual_plot[0], node_plot[0]],
            [actual_plot[1], node_plot[1]],
            linestyle="--",
            color=connector_color,
            linewidth=1.6,
            alpha=0.92,
            zorder=8,
            label=connector_label,
        )
        ax.scatter(
            node_plot[0],
            node_plot[1],
            c="none",
            marker=node_marker,
            s=node_size,
            edgecolors=node_edge_color,
            linewidths=2.0,
            zorder=10,
            label=node_label,
        )

    _draw_maze_walls(ax, maze_grid, alpha=1.0, zorder=5)
    _plot_task_icons(
        ax,
        env_id,
        start_xy=np.asarray(anchor_xys[0], dtype=np.float32),
        goal_xy=np.asarray(anchor_xys[-1], dtype=np.float32),
        waypoints_xy=np.asarray(anchor_xys[1:-1], dtype=np.float32),
    )

    route_text = str(route_solution.get("route_text", ""))
    if bool(route_solution.get("feasible", False)) and bool(route_segments.get("reachable", False)):
        path_summary = f"route={route_text}  total={float(route_segments['total_distance']):.3f}"
    else:
        path_summary = "route=no feasible Hamiltonian path"
    snap_tokens = []
    for anchor_idx, snap_dist in enumerate(np.asarray(anchor_node_dists, dtype=np.float32).tolist()):
        snap_tokens.append(f"{_anchor_short_label(anchor_idx, len(anchor_xys))}={float(snap_dist):.3f}")
    task_prefix = f"{task_name}  " if task_name else ""
    ax.set_title(
        f"{task_prefix}Sampled Graph Goal Distances + Hamiltonian Route\n"
        f"ratio={sample_ratio:.4f}  n={len(sampled_xy):,}/{n_total:,}  radius<={edge_radius:.2f}\n"
        f"snap dists: {'  '.join(snap_tokens)}  reachable={n_reachable}/{len(sampled_xy):,}\n"
        f"{path_summary}"
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def _default_output_path(
    dataset_name: str,
    task_id: int,
    goal_xy: np.ndarray,
    explicit_goal: bool,
    task_override_path: Optional[str],
    waypoint_group_idx: Optional[int],
) -> Path:
    if task_override_path not in (None, ""):
        goal_tag = Path(str(task_override_path)).stem
        if waypoint_group_idx is not None:
            goal_tag = f"{goal_tag}_g{int(waypoint_group_idx)}"
    else:
        goal_tag = (
            f"gx{goal_xy[0]:.2f}_gy{goal_xy[1]:.2f}"
            if explicit_goal
            else "task_goal"
        )
    safe_goal_tag = goal_tag.replace("/", "_").replace(" ", "_")
    return _REPO_ROOT / "visualizations" / f"{dataset_name}_task{task_id}_{safe_goal_tag}_temporal_dist_viz.png"


def _parse_goal_pos(raw_goal: Optional[str]) -> Optional[np.ndarray]:
    if raw_goal is None:
        return None
    parts = [part.strip() for part in raw_goal.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--goal_pos must look like 'x,y', got: {raw_goal}")
    return np.asarray([float(parts[0]), float(parts[1])], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal-distance heatmap visualizer")
    parser.add_argument("--ckpt", required=True, help="Path to model.ckpt")
    parser.add_argument("--task_id", type=int, default=1, help="OGBench task id used for reference observation")
    parser.add_argument(
        "--task_override_path",
        default=None,
        help="Optional repo-relative or absolute task override YAML/JSON path",
    )
    parser.add_argument(
        "--waypoint_group_idx",
        type=int,
        default=None,
        help="Optional 0-based waypoint group index override for multi-group task overrides",
    )
    parser.add_argument("--goal_pos", default=None, help="Destination position as 'x,y'. If omitted, uses task goal.")
    parser.add_argument("--grid_res", type=int, default=100, help="Heatmap grid resolution per axis")
    parser.add_argument("--grad_grid_step", type=float, default=2.0, help="World-coordinate spacing for grad field")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Dataset state sampling ratio for scatter panel")
    parser.add_argument("--edge_radius", type=float, default=None, help="Undirected graph edge radius in world coords")
    parser.add_argument("--graph_seed", type=int, default=None, help="Sampling RNG seed for sampled graph cache")
    parser.add_argument("--gamma", type=float, default=0.995, help="Gamma for emb_dist_to_temporal_dist")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--no_show", action="store_true", help="Skip plt.show()")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    task_override_path = None if args.task_override_path in (None, "", "none", "None") else str(args.task_override_path)
    algo_overrides, dataset_config_name, dataset_meta = _load_training_metadata(ckpt_path)
    root_cfg = _build_root_cfg(
        dataset_config_name,
        algo_overrides,
        dataset_meta,
        task_override_path=task_override_path,
        task_override_waypoint_group_idx=args.waypoint_group_idx,
    )
    algo = _build_algo(root_cfg)

    task_ref_info = _initialize_hilp_reference_obs(algo, args.task_id)
    if task_ref_info is None or task_ref_info.get("start_xy") is None:
        raise ValueError("Could not infer task start from env reset.")
    task_start_xy = np.asarray(task_ref_info["start_xy"], dtype=np.float32)
    waypoints_xy = np.asarray(
        task_ref_info.get("waypoints_xy", np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 2)
    task_name = task_ref_info.get("task_name")

    explicit_goal = args.goal_pos is not None and task_override_path is None
    goal_xy = None
    if task_override_path is not None:
        goal_xy = task_ref_info.get("goal_xy")
    else:
        goal_xy = _parse_goal_pos(args.goal_pos)
        if goal_xy is None:
            goal_xy = task_ref_info.get("goal_xy")
    if goal_xy is None:
        raise ValueError("Could not infer task goal from env reset. Please provide --goal_pos x,y.")
    goal_xy = np.asarray(goal_xy, dtype=np.float32)

    dataset_name = str(algo.dataset)
    sampled_graph_cfg = _resolve_sampled_graph_cfg(algo, args)
    sample_ratio = float(sampled_graph_cfg["sample_ratio"])
    edge_radius = float(sampled_graph_cfg["edge_radius"])
    graph_seed = int(sampled_graph_cfg["graph_seed"])
    graph_cache_dir = Path(sampled_graph_cfg["cache_dir"])
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

    npz_path = Path(algo._kde_save_dir).expanduser() / f"{dataset_name}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset npz not found: {npz_path}")
    sampled_graph_cache = build_or_load_sampled_graph_cache_from_npz(
        npz_path=str(npz_path),
        dataset=dataset_name,
        save_dir=str(graph_cache_dir),
        sample_ratio=sample_ratio,
        edge_radius=edge_radius,
        seed=graph_seed,
    )
    anchor_bundle = _build_anchor_queries(
        start_xy=task_start_xy,
        goal_xy=goal_xy,
        waypoints_xy=waypoints_xy,
    )
    anchor_assignment = query_distinct_nearest_nodes(
        sampled_graph_cache,
        query_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
        priority_order=np.asarray(anchor_bundle["priority_order"], dtype=np.int32),
        query_labels=list(anchor_bundle["anchor_labels"]),
    )
    anchor_node_indices = np.asarray(anchor_assignment["node_indices"], dtype=np.int32)
    anchor_node_xys = np.asarray(anchor_assignment["node_xys"], dtype=np.float32)
    anchor_node_dists = np.asarray(anchor_assignment["node_euclidean_dists"], dtype=np.float32)
    temporal_anchor_dists = _compute_pairwise_temporal_distance_matrix(
        algo,
        src_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
        dst_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
        gamma=args.gamma,
    )
    temporal_route_solution = _solve_fixed_endpoint_hamiltonian_path(temporal_anchor_dists)
    if bool(temporal_route_solution["feasible"]):
        temporal_route_segments = _build_straight_anchor_route_segments(
            anchor_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
            anchor_order=np.asarray(temporal_route_solution["anchor_order"], dtype=np.int32),
        )
    else:
        temporal_route_segments = {
            "reachable": False,
            "segments": [],
            "total_distance": np.inf,
        }

    graph_anchor_dists = extract_shortest_path_submatrix(sampled_graph_cache, anchor_node_indices)
    graph_route_solution = _solve_fixed_endpoint_hamiltonian_path(graph_anchor_dists)
    if bool(graph_route_solution["feasible"]):
        graph_route_segments = _expand_anchor_route_segments(
            sampled_graph_cache,
            anchor_node_indices=anchor_node_indices,
            anchor_order=np.asarray(graph_route_solution["anchor_order"], dtype=np.int32),
        )
    else:
        graph_route_segments = {
            "reachable": False,
            "segments": [],
            "total_distance": np.inf,
        }

    sampled_xy = np.asarray(sampled_graph_cache["points_xy"], dtype=np.float32)
    n_total = int(sampled_graph_cache["n_total"])
    goal_node_index = int(anchor_node_indices[-1])
    goal_shortest_dists = np.asarray(
        sampled_graph_cache["shortest_dists"][goal_node_index],
        dtype=np.float32,
    )
    n_reachable = int(np.isfinite(goal_shortest_dists).sum())
    print(
        f"[SampledGraph] goal node idx={goal_node_index} "
        f"goal->node euclidean={float(anchor_node_dists[-1]):.4f} "
        f"reachable={n_reachable}/{len(sampled_xy)}",
        flush=True,
    )
    for assignment in anchor_assignment["assignments"]:
        print(
            f"[SampledGraph] anchor={assignment['query_label']} "
            f"node_idx={assignment['node_index']} "
            f"rank={assignment['node_rank']} "
            f"snap_dist={assignment['node_euclidean_dist']:.4f}",
            flush=True,
        )
    print(
        f"[TemporalRoute] route_feasible={temporal_route_solution['feasible']} "
        f"route_reachable={temporal_route_segments['reachable']} "
        f"route={temporal_route_solution['route_text']} "
        f"temporal_total={float(temporal_route_solution['total_cost']):.4f}",
        flush=True,
    )
    print(
        f"[SampledGraph] route_feasible={graph_route_solution['feasible']} "
        f"route_reachable={graph_route_segments['reachable']} "
        f"route={graph_route_solution['route_text']} "
        f"graph_total={float(graph_route_segments['total_distance']):.4f}",
        flush=True,
    )

    heatmap = _compute_temporal_distance_heatmap(algo, goal_xy, grid_res=args.grid_res, gamma=args.gamma)
    grad_field = algo._compute_guidance_grad_fields(goal_xy.astype(np.float32), grid_step=args.grad_grid_step)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    title_lines = [
        "Temporal Distance Visualization",
        f"dataset={dataset_name}  task_id={args.task_id}  goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})",
    ]
    if task_name:
        title_lines.append(f"task={task_name}")
    if len(waypoints_xy) > 0:
        title_lines.append(f"waypoints={len(waypoints_xy)}")
    fig.suptitle("\n".join(title_lines), fontsize=12)
    _plot_temporal_panel(
        axes[0],
        fig,
        algo.env_id,
        heatmap,
        grad_field,
        start_xy=task_start_xy,
        goal_xy=goal_xy,
        waypoints_xy=waypoints_xy,
        route_solution=temporal_route_solution,
        route_segments=temporal_route_segments,
        task_name=task_name,
    )
    _plot_sample_panel(
        axes[1],
        fig,
        algo.env_id,
        sampled_xy,
        shortest_dists=goal_shortest_dists,
        anchor_xys=np.asarray(anchor_bundle["anchor_xys"], dtype=np.float32),
        anchor_node_xys=anchor_node_xys,
        anchor_node_dists=anchor_node_dists,
        sample_ratio=sample_ratio,
        edge_radius=edge_radius,
        n_total=n_total,
        n_reachable=n_reachable,
        route_solution=graph_route_solution,
        route_segments=graph_route_segments,
        task_name=task_name,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out_path = Path(args.out).expanduser() if args.out else _default_output_path(
        dataset_name=dataset_name,
        task_id=args.task_id,
        goal_xy=goal_xy,
        explicit_goal=explicit_goal,
        task_override_path=task_override_path,
        waypoint_group_idx=args.waypoint_group_idx,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"[SAVED] {out_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
