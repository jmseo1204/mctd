#!/usr/bin/env python3
"""Analysis #2 (extraction): enumerate ALL waypoint combinations (not just
mismatches) for each task and compute features + a binary `mismatch` label.

The feature set mirrors analyze_waypoint_disagreement_gap.py so the
downstream classification report can compare features across the
mismatch vs non-mismatch boundary.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.diffusion_forcing.sampled_graph_estimator import (
    assign_distinct_nodes_from_rankings,
    build_or_load_sampled_graph_cache_from_npz,
    precompute_nearest_node_rankings,
)
from scripts.analyze_waypoint_disagreement_gap import (
    _compute_features,
    _load_feasible_points_payload,
)
from scripts.temporal_dist_heatmap import (
    _build_algo,
    _build_root_cfg,
    _initialize_hilp_reference_obs,
    _load_training_metadata,
    _resolve_sampled_graph_cfg,
)
from utils.route_metric_utils import (
    compute_pairwise_temporal_distance_matrix,
    solve_fixed_endpoint_hamiltonian_path_with_second_best,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--task-override",
        default="configurations/task_overrides/antmaze_giant_waypoints.yaml",
    )
    parser.add_argument("--feasible-points-path", default=None)
    parser.add_argument("--graph-cache-path", default=None)
    parser.add_argument(
        "--output-dir", default="outputs/waypoint_gap_analysis_full"
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--edge_radius", type=float, default=None)
    parser.add_argument("--graph_seed", type=int, default=None)
    parser.add_argument(
        "--limit-per-task", type=int, default=0, help="Debug subset"
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    task_override_path = Path(args.task_override).expanduser()
    if not task_override_path.is_absolute():
        task_override_path = (REPO_ROOT / task_override_path).resolve()
    with open(task_override_path, "r", encoding="utf-8") as f:
        override_payload = yaml.safe_load(f) or {}

    feasible_points_path_raw = (
        args.feasible_points_path
        if args.feasible_points_path
        else override_payload.get("source_feasible_points_path")
    )
    feasible_points_path = Path(feasible_points_path_raw).expanduser()
    if not feasible_points_path.is_absolute():
        feasible_points_path = (REPO_ROOT / feasible_points_path).resolve()
    feasible_env_id, feasible_ijs = _load_feasible_points_payload(feasible_points_path)
    num_waypoints = int(override_payload.get("num_waypoints", 3))
    if num_waypoints != 3:
        raise ValueError("This analysis assumes num_waypoints=3")

    algo_overrides, dataset_config_name, dataset_meta = _load_training_metadata(ckpt_path)
    ogbench_enable_reset_perturb = bool(
        override_payload.get(
            "ogbench_enable_reset_perturb",
            algo_overrides.get("ogbench_enable_reset_perturb", True),
        )
    )
    algo_overrides["ogbench_enable_reset_perturb"] = ogbench_enable_reset_perturb
    root_cfg = _build_root_cfg(
        dataset_config_name,
        algo_overrides,
        dataset_meta,
        task_override_path=None,
        task_override_waypoint_group_idx=None,
    )
    algo = _build_algo(root_cfg)
    if str(algo.env_id) != feasible_env_id:
        raise ValueError(
            f"feasible env_id {feasible_env_id} != ckpt env_id {algo.env_id}"
        )

    sampled_graph_cfg = _resolve_sampled_graph_cfg(algo, args)
    dataset_name = str(algo.dataset)
    graph_cache_path_raw = (
        args.graph_cache_path
        if args.graph_cache_path
        else override_payload.get("sampled_graph_cache_path")
    )
    graph_cache_path = Path(graph_cache_path_raw).expanduser()
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
    local_priority_order = np.asarray(
        [0, num_waypoints + 1] + list(range(1, num_waypoints + 1)), dtype=np.int32
    )

    maze_type = algo.env_id.split("-")[1]
    env = algo._make_single_ogbench_env(maze_type)

    rows: list[dict[str, Any]] = []
    try:
        tasks_payload = override_payload.get("tasks", {}) or {}
        for task_id_raw in sorted(tasks_payload.keys(), key=lambda x: int(x)):
            task_id = int(task_id_raw)
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
            if len(candidate_ijs) < num_waypoints:
                continue

            task_ref_info = _initialize_hilp_reference_obs(algo, task_id)
            start_xy = np.asarray(task_ref_info["start_xy"], dtype=np.float32).reshape(2)
            goal_xy = np.asarray(task_ref_info["goal_xy"], dtype=np.float32).reshape(2)
            full_anchor_xys = np.concatenate(
                [start_xy.reshape(1, 2), candidate_xys, goal_xy.reshape(1, 2)],
                axis=0,
            ).astype(np.float32, copy=False)

            print(f"[Task {task_id}] computing temporal full matrix", flush=True)
            temporal_full_matrix = compute_pairwise_temporal_distance_matrix(
                algo,
                src_xys=full_anchor_xys,
                dst_xys=full_anchor_xys,
                gamma=float(args.gamma),
            )
            graph_ranking_cache = precompute_nearest_node_rankings(
                sampled_graph_cache, full_anchor_xys
            )

            combo_local_indices = list(
                combinations(range(len(candidate_ijs)), num_waypoints)
            )
            if args.limit_per_task and args.limit_per_task > 0:
                combo_local_indices = combo_local_indices[: args.limit_per_task]
            n_combos = len(combo_local_indices)
            print(f"[Task {task_id}] processing {n_combos} combos", flush=True)

            for combo_idx, combo_local in enumerate(combo_local_indices):
                combo_local_arr = np.asarray(combo_local, dtype=np.int32)
                combo_full = combo_local_arr + 1
                full_indices = np.concatenate(
                    [
                        np.asarray([0], dtype=np.int32),
                        combo_full,
                        np.asarray(
                            [len(full_anchor_xys) - 1], dtype=np.int32
                        ),
                    ]
                )
                T_sub = temporal_full_matrix[
                    np.ix_(full_indices, full_indices)
                ].astype(np.float64, copy=False)

                assignment = assign_distinct_nodes_from_rankings(
                    sampled_graph_cache,
                    graph_ranking_cache,
                    query_indices=full_indices,
                    priority_order=local_priority_order,
                )
                node_indices = np.asarray(
                    assignment["node_indices"], dtype=np.int32
                )
                G_sub = graph_shortest_dists[
                    np.ix_(node_indices, node_indices)
                ].astype(np.float64, copy=False)

                if not (np.all(np.isfinite(G_sub)) and np.all(np.isfinite(T_sub))):
                    continue

                # Determine mismatch by comparing waypoint orderings (excluding endpoints).
                t_route = solve_fixed_endpoint_hamiltonian_path_with_second_best(T_sub)
                g_route = solve_fixed_endpoint_hamiltonian_path_with_second_best(G_sub)
                if not (t_route["feasible"] and g_route["feasible"]):
                    continue
                t_order = np.asarray(t_route["anchor_order"], dtype=np.int32)[1:-1]
                g_order = np.asarray(g_route["anchor_order"], dtype=np.int32)[1:-1]
                mismatch = int(not np.array_equal(t_order, g_order))

                feats = _compute_features(T_sub, G_sub)
                feats["task_id"] = task_id
                feats["combo_idx_in_task"] = combo_idx
                feats["mismatch"] = mismatch
                ijs = candidate_ijs[combo_local_arr].tolist()
                feats["w1_i"], feats["w1_j"] = int(ijs[0][0]), int(ijs[0][1])
                feats["w2_i"], feats["w2_j"] = int(ijs[1][0]), int(ijs[1][1])
                feats["w3_i"], feats["w3_j"] = int(ijs[2][0]), int(ijs[2][1])
                rows.append(feats)

                if (combo_idx + 1) % 5000 == 0:
                    print(
                        f"  [Task {task_id}] {combo_idx + 1}/{n_combos}", flush=True
                    )
    finally:
        env.close()

    if not rows:
        raise RuntimeError("No rows produced")
    df = pd.DataFrame(rows)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "features_full.csv"
    df.to_csv(csv_path, index=False)
    pkl_path = output_dir / "features_full.pkl"
    df.to_pickle(pkl_path)
    print(
        f"[SAVED] {csv_path}  rows={len(df)}  cols={len(df.columns)}  "
        f"mismatch_rate={df['mismatch'].mean():.4f}",
        flush=True,
    )
    print(f"[SAVED] {pkl_path}", flush=True)


if __name__ == "__main__":
    main()
