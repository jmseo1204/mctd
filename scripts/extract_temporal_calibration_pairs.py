#!/usr/bin/env python3
"""Extract per-task paired (T, G) edge observations for temporal-distance
calibration. For each task, computes the temporal distance matrix and the
graph shortest-distance matrix over the same anchor set (start + all
candidates + goal), then emits the upper-triangular pairs as a CSV.

This decouples calibration from the row-level summaries (T_min/T_mean/...)
in features.csv where T_min and G_min come from independently sorted edges
and therefore cannot be paired at the edge level.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.diffusion_forcing.sampled_graph_estimator import (
    build_or_load_sampled_graph_cache_from_npz,
    precompute_nearest_node_rankings,
    assign_distinct_nodes_from_rankings,
)
from scripts.analyze_waypoint_disagreement_gap import _load_feasible_points_payload
from scripts.temporal_dist_heatmap import (
    _build_algo,
    _build_root_cfg,
    _initialize_hilp_reference_obs,
    _load_training_metadata,
    _resolve_sampled_graph_cfg,
)
from utils.route_metric_utils import compute_pairwise_temporal_distance_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--task-override",
        default="configurations/task_overrides/antmaze_giant_waypoints.yaml",
    )
    parser.add_argument("--graph-cache-path", default=None)
    parser.add_argument(
        "--output", default="outputs/waypoint_gap_analysis/temporal_pairs.csv"
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--edge_radius", type=float, default=None)
    parser.add_argument("--graph_seed", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    task_override_path = Path(args.task_override).expanduser()
    if not task_override_path.is_absolute():
        task_override_path = (REPO_ROOT / task_override_path).resolve()
    with open(task_override_path, "r", encoding="utf-8") as f:
        override_payload = yaml.safe_load(f) or {}

    feasible_points_path = (
        Path(override_payload["source_feasible_points_path"]).expanduser()
    )
    if not feasible_points_path.is_absolute():
        feasible_points_path = (REPO_ROOT / feasible_points_path).resolve()
    feasible_env_id, feasible_ijs = _load_feasible_points_payload(feasible_points_path)

    algo_overrides, dataset_config_name, dataset_meta = _load_training_metadata(ckpt_path)
    algo_overrides["ogbench_enable_reset_perturb"] = bool(
        override_payload.get("ogbench_enable_reset_perturb", True)
    )
    root_cfg = _build_root_cfg(
        dataset_config_name,
        algo_overrides,
        dataset_meta,
        task_override_path=None,
        task_override_waypoint_group_idx=None,
    )
    algo = _build_algo(root_cfg)

    sampled_graph_cfg = _resolve_sampled_graph_cfg(algo, args)
    dataset_name = str(algo.dataset)
    graph_cache_path = Path(
        args.graph_cache_path
        if args.graph_cache_path
        else override_payload["sampled_graph_cache_path"]
    ).expanduser()
    if not graph_cache_path.is_absolute():
        graph_cache_path = (REPO_ROOT / graph_cache_path).resolve()
    npz_path = Path(algo._kde_save_dir).expanduser() / f"{dataset_name}.npz"
    sampled_graph_cache = build_or_load_sampled_graph_cache_from_npz(
        npz_path=str(npz_path),
        dataset=dataset_name,
        save_dir=str(sampled_graph_cfg["cache_dir"]),
        sample_ratio=float(sampled_graph_cfg["sample_ratio"]),
        edge_radius=float(sampled_graph_cfg["edge_radius"]),
        seed=int(sampled_graph_cfg["graph_seed"]),
        cache_path=str(graph_cache_path),
    )
    graph_shortest_dists = np.asarray(
        sampled_graph_cache["shortest_dists"], dtype=np.float32
    )

    maze_type = algo.env_id.split("-")[1]
    env = algo._make_single_ogbench_env(maze_type)

    rows = []
    try:
        for task_id in sorted(int(k) for k in override_payload.get("tasks", {}).keys()):
            task_info = env.task_infos[task_id - 1]
            start_ij = np.asarray(task_info["init_ij"], dtype=np.int32).reshape(2)
            goal_ij = np.asarray(task_info["goal_ij"], dtype=np.int32).reshape(2)
            start_goal_mask = ~(
                np.all(feasible_ijs == start_ij[None, :], axis=1)
                | np.all(feasible_ijs == goal_ij[None, :], axis=1)
            )
            candidate_ijs = feasible_ijs[start_goal_mask].astype(np.int32, copy=True)
            candidate_xys = np.asarray(
                [env.ij_to_xy((int(ij[0]), int(ij[1]))) for ij in candidate_ijs],
                dtype=np.float32,
            ).reshape(-1, 2)
            task_ref_info = _initialize_hilp_reference_obs(algo, task_id)
            start_xy = np.asarray(task_ref_info["start_xy"], dtype=np.float32).reshape(2)
            goal_xy = np.asarray(task_ref_info["goal_xy"], dtype=np.float32).reshape(2)
            full_anchor_xys = np.concatenate(
                [start_xy.reshape(1, 2), candidate_xys, goal_xy.reshape(1, 2)],
                axis=0,
            ).astype(np.float32, copy=False)
            n = len(full_anchor_xys)

            print(f"[Task {task_id}] anchors={n}, computing T full matrix", flush=True)
            T_full = compute_pairwise_temporal_distance_matrix(
                algo,
                src_xys=full_anchor_xys,
                dst_xys=full_anchor_xys,
                gamma=float(args.gamma),
            )
            ranking = precompute_nearest_node_rankings(
                sampled_graph_cache, full_anchor_xys
            )
            assignment = assign_distinct_nodes_from_rankings(
                sampled_graph_cache,
                ranking,
                query_indices=np.arange(n, dtype=np.int32),
                priority_order=np.arange(n, dtype=np.int32),
            )
            node_indices = np.asarray(assignment["node_indices"], dtype=np.int32)
            G_full = graph_shortest_dists[
                np.ix_(node_indices, node_indices)
            ].astype(np.float64, copy=False)

            iu = np.triu_indices(n, k=1)
            T_pairs = T_full[iu].astype(np.float64)
            G_pairs = G_full[iu].astype(np.float64)
            mask = np.isfinite(T_pairs) & np.isfinite(G_pairs) & (T_pairs > 0) & (G_pairs > 0)
            i_idx = iu[0][mask]
            j_idx = iu[1][mask]
            for ii, jj, tt, gg in zip(i_idx, j_idx, T_pairs[mask], G_pairs[mask]):
                rows.append(
                    {
                        "task_id": int(task_id),
                        "i": int(ii),
                        "j": int(jj),
                        "T": float(tt),
                        "G": float(gg),
                    }
                )
            print(
                f"  [Task {task_id}] kept {int(mask.sum())} / {len(mask)} pairs",
                flush=True,
            )
    finally:
        env.close()

    df = pd.DataFrame(rows)
    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[SAVED] {out}  pairs={len(df)}", flush=True)


if __name__ == "__main__":
    main()
