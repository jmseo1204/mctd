#!/usr/bin/env python3
"""Extract per-combo features and Hamiltonian-route gap targets from the
mismatch waypoint groups produced by ``find_waypoint_metric_disagreements.py``.

For each (task, waypoint_group) row in the task-override yaml we compute:
  - 5x5 temporal-distance submatrix  T  (HILP-derived)
  - 5x5 graph shortest-distance submatrix  G  (sampled-graph snapped)
  - graph_optimal_order  : Hamiltonian path under G
  - temporal_order       : Hamiltonian path under T
  - gap_abs = G_len(temporal_order) - G_len(graph_optimal_order)   [>= 0]
  - gap_rel = gap_abs / G_len(graph_optimal_order)

A bunch of feature columns are derived from T and G under several hypotheses
(temporal-vs-graph scale/accuracy, point spread, ordering robustness, tour
characteristics). The full row-level dataframe is saved as parquet for the
downstream report script.
"""

from __future__ import annotations

import argparse
import os
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


def _load_feasible_points_payload(path: Path) -> tuple[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    env_id = payload.get("env_id")
    if not env_id:
        raise ValueError(f"'env_id' is required in {path}")
    feasible_points = payload.get("feasible_points", [])
    feasible_ijs = np.asarray(feasible_points, dtype=np.int32)
    if feasible_ijs.size == 0:
        feasible_ijs = np.zeros((0, 2), dtype=np.int32)
    else:
        feasible_ijs = feasible_ijs.reshape(-1, 2)
    return str(env_id), feasible_ijs


def _hamiltonian_total_under_metric(
    metric_5x5: np.ndarray, order: np.ndarray
) -> float:
    """Sum step costs along the given anchor order using the supplied metric."""
    order = np.asarray(order, dtype=np.int32)
    step_costs = metric_5x5[order[:-1], order[1:]]
    if not np.all(np.isfinite(step_costs)):
        return float("inf")
    return float(np.sum(step_costs, dtype=np.float64))


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return float("nan")
    ar = pd.Series(a).rank().to_numpy()
    br = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _pairwise_inversions(a: np.ndarray, b: np.ndarray) -> int:
    """Count pairs (i,j) where the ranking of a vs b disagrees in sign."""
    n = len(a)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                inv += 1
    return int(inv)


def _offdiag_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def _compute_features(T: np.ndarray, G: np.ndarray) -> dict[str, float]:
    """Derive feature columns from the 5x5 temporal (T) and graph (G) matrices.

    Both matrices are assumed symmetric in expectation; we use the upper-tri
    off-diagonal (10 edges) for most aggregates.
    """
    feats: dict[str, float] = {}
    T_off = _offdiag_values(T).astype(np.float64)
    G_off = _offdiag_values(G).astype(np.float64)

    # ---- Hypothesis B: spread / scale of points ----
    feats["G_max"] = float(np.max(G_off))
    feats["G_min"] = float(np.min(G_off))
    feats["G_mean"] = float(np.mean(G_off))
    feats["G_std"] = float(np.std(G_off))
    feats["G_median"] = float(np.median(G_off))
    feats["G_range"] = feats["G_max"] - feats["G_min"]
    feats["G_compactness"] = (
        float(feats["G_min"] / feats["G_max"]) if feats["G_max"] > 0 else float("nan")
    )
    feats["T_max"] = float(np.max(T_off))
    feats["T_min"] = float(np.min(T_off))
    feats["T_mean"] = float(np.mean(T_off))
    feats["T_std"] = float(np.std(T_off))
    feats["T_median"] = float(np.median(T_off))

    # ---- Hypothesis A: temporal scale / accuracy vs graph ----
    # Scale-match using median ratio (robust); compare T*scale to G.
    ratios = T_off / np.where(G_off > 0, G_off, np.nan)
    feats["ratio_TG_median"] = float(np.nanmedian(ratios))
    feats["ratio_TG_std"] = float(np.nanstd(ratios))
    feats["ratio_TG_max"] = float(np.nanmax(ratios))
    feats["ratio_TG_min"] = float(np.nanmin(ratios))
    feats["ratio_TG_iqr"] = float(
        np.nanpercentile(ratios, 75) - np.nanpercentile(ratios, 25)
    )

    inv_scale = (
        1.0 / feats["ratio_TG_median"]
        if feats["ratio_TG_median"] not in (0.0, float("inf"), float("nan"))
        else float("nan")
    )
    if np.isfinite(inv_scale):
        T_scaled = T_off * inv_scale
        rel_err = (T_scaled - G_off) / np.where(G_off > 0, G_off, np.nan)
        feats["scaled_relerr_max"] = float(np.nanmax(np.abs(rel_err)))
        feats["scaled_relerr_mean"] = float(np.nanmean(np.abs(rel_err)))
        feats["scaled_relerr_std"] = float(np.nanstd(rel_err))
        log_ratio = np.log(np.where(T_scaled > 0, T_scaled, np.nan)) - np.log(
            np.where(G_off > 0, G_off, np.nan)
        )
        feats["log_ratio_abs_mean"] = float(np.nanmean(np.abs(log_ratio)))
        feats["log_ratio_abs_max"] = float(np.nanmax(np.abs(log_ratio)))
    else:
        feats["scaled_relerr_max"] = float("nan")
        feats["scaled_relerr_mean"] = float("nan")
        feats["scaled_relerr_std"] = float("nan")
        feats["log_ratio_abs_mean"] = float("nan")
        feats["log_ratio_abs_max"] = float("nan")

    feats["spearman_TG"] = _spearman_corr(T_off, G_off)
    feats["pearson_TG"] = _pearson_corr(T_off, G_off)
    feats["edge_inversions"] = float(_pairwise_inversions(T_off, G_off))

    # ---- Hypothesis C: ordering robustness ----
    g_route = solve_fixed_endpoint_hamiltonian_path_with_second_best(G)
    t_route = solve_fixed_endpoint_hamiltonian_path_with_second_best(T)
    feats["graph_second_best_gap"] = float(g_route["second_best_gap"])
    feats["temporal_second_best_gap"] = float(t_route["second_best_gap"])
    feats["graph_second_best_gap_rel"] = (
        float(g_route["second_best_gap"] / g_route["total_cost"])
        if g_route["total_cost"] > 0
        else float("nan")
    )
    feats["temporal_second_best_gap_rel"] = (
        float(t_route["second_best_gap"] / t_route["total_cost"])
        if t_route["total_cost"] > 0
        else float("nan")
    )

    # ---- Hypothesis D: tour characteristics ----
    feats["graph_optimal_tour_len"] = float(g_route["total_cost"])
    feats["temporal_optimal_tour_len"] = float(t_route["total_cost"])
    feats["graph_SG_direct"] = float(G[0, -1])
    feats["graph_SG_shortcut_ratio"] = (
        float(G[0, -1] / g_route["total_cost"])
        if g_route["total_cost"] > 0
        else float("nan")
    )
    g_step_costs = np.asarray(g_route["step_costs"], dtype=np.float64)
    feats["graph_max_edge_in_tour"] = float(np.max(g_step_costs))
    feats["graph_max_edge_in_tour_ratio"] = (
        float(np.max(g_step_costs) / g_route["total_cost"])
        if g_route["total_cost"] > 0
        else float("nan")
    )
    feats["graph_step_std_in_tour"] = float(np.std(g_step_costs))

    # Targets: gap = G(temporal_order) - G(graph_optimal_order)
    temporal_order = np.asarray(t_route["anchor_order"], dtype=np.int32)
    g_under_temporal = _hamiltonian_total_under_metric(G, temporal_order)
    feats["graph_under_temporal_order_len"] = float(g_under_temporal)
    feats["gap_abs"] = float(g_under_temporal - g_route["total_cost"])
    feats["gap_rel"] = (
        float(feats["gap_abs"] / g_route["total_cost"])
        if g_route["total_cost"] > 0
        else float("nan")
    )

    return feats


def _build_ij_to_candidate_index(candidate_ijs: np.ndarray) -> dict[tuple[int, int], int]:
    return {
        (int(ij[0]), int(ij[1])): idx for idx, ij in enumerate(candidate_ijs)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute features and Hamiltonian gap targets per mismatch waypoint group."
    )
    parser.add_argument("--ckpt", required=True, help="Path to model.ckpt")
    parser.add_argument(
        "--task-override",
        default="configurations/task_overrides/antmaze_giant_waypoints.yaml",
        help="Repo-relative or absolute YAML path containing mismatch waypoint groups",
    )
    parser.add_argument(
        "--feasible-points-path",
        default=None,
        help="Override feasible-points yaml. Defaults to value stored in the task-override yaml.",
    )
    parser.add_argument(
        "--graph-cache-path",
        default=None,
        help="Override sampled-graph .pkl path. Defaults to value stored in task-override yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/waypoint_gap_analysis",
        help="Repo-relative or absolute directory to save features parquet",
    )
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--edge_radius", type=float, default=None)
    parser.add_argument("--graph_seed", type=int, default=None)
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="If >0, only process this many waypoint groups per task (debug)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    task_override_path = Path(args.task_override).expanduser()
    if not task_override_path.is_absolute():
        task_override_path = (REPO_ROOT / task_override_path).resolve()
    if not task_override_path.is_file():
        raise FileNotFoundError(f"Task override yaml not found: {task_override_path}")

    with open(task_override_path, "r", encoding="utf-8") as f:
        override_payload = yaml.safe_load(f) or {}

    feasible_points_path_raw = (
        args.feasible_points_path
        if args.feasible_points_path
        else override_payload.get("source_feasible_points_path")
    )
    if not feasible_points_path_raw:
        raise ValueError(
            "feasible-points path not provided and not found in task-override yaml"
        )
    feasible_points_path = Path(feasible_points_path_raw).expanduser()
    if not feasible_points_path.is_absolute():
        feasible_points_path = (REPO_ROOT / feasible_points_path).resolve()

    feasible_env_id, feasible_ijs = _load_feasible_points_payload(feasible_points_path)
    num_waypoints = int(override_payload.get("num_waypoints", 3))
    if num_waypoints != 3:
        raise ValueError(
            f"This analysis assumes num_waypoints=3, got {num_waypoints}"
        )

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
    if not graph_cache_path_raw:
        raise ValueError("graph cache path not provided and not in task-override yaml")
    graph_cache_path = Path(graph_cache_path_raw).expanduser()
    if not graph_cache_path.is_absolute():
        graph_cache_path = (REPO_ROOT / graph_cache_path).resolve()

    npz_path = Path(algo._kde_save_dir).expanduser() / f"{dataset_name}.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Dataset npz not found: {npz_path}")
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
        [0, num_waypoints + 1] + list(range(1, num_waypoints + 1)),
        dtype=np.int32,
    )

    maze_type = algo.env_id.split("-")[1]
    env = algo._make_single_ogbench_env(maze_type)

    rows: list[dict[str, Any]] = []
    try:
        tasks_payload = override_payload.get("tasks", {}) or {}
        for task_id_raw, task_block in tasks_payload.items():
            task_id = int(task_id_raw)
            waypoint_ij_groups = task_block.get("waypoint_ij_groups", []) or []
            if not waypoint_ij_groups:
                print(f"[Task {task_id}] no waypoint groups; skipping", flush=True)
                continue

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
            ij_to_local = _build_ij_to_candidate_index(candidate_ijs)

            task_ref_info = _initialize_hilp_reference_obs(algo, task_id)
            if (
                task_ref_info is None
                or task_ref_info.get("start_xy") is None
                or task_ref_info.get("goal_xy") is None
            ):
                raise RuntimeError(
                    f"Failed to init HILP reference observation for task {task_id}"
                )
            start_xy = np.asarray(task_ref_info["start_xy"], dtype=np.float32).reshape(2)
            goal_xy = np.asarray(task_ref_info["goal_xy"], dtype=np.float32).reshape(2)

            full_anchor_xys = np.concatenate(
                [start_xy.reshape(1, 2), candidate_xys, goal_xy.reshape(1, 2)],
                axis=0,
            ).astype(np.float32, copy=False)

            print(
                f"[Task {task_id}] computing temporal full matrix "
                f"({len(full_anchor_xys)}x{len(full_anchor_xys)})",
                flush=True,
            )
            temporal_full_matrix = compute_pairwise_temporal_distance_matrix(
                algo,
                src_xys=full_anchor_xys,
                dst_xys=full_anchor_xys,
                gamma=float(args.gamma),
            )
            graph_ranking_cache = precompute_nearest_node_rankings(
                sampled_graph_cache, full_anchor_xys
            )

            iter_groups = waypoint_ij_groups
            if args.limit_per_task and args.limit_per_task > 0:
                iter_groups = iter_groups[: args.limit_per_task]

            n_groups = len(iter_groups)
            print(
                f"[Task {task_id}] processing {n_groups} mismatch waypoint groups",
                flush=True,
            )

            for combo_idx, group in enumerate(iter_groups):
                ijs = [(int(p[0]), int(p[1])) for p in group]
                try:
                    combo_local = np.asarray(
                        [ij_to_local[ij] for ij in ijs], dtype=np.int32
                    )
                except KeyError as e:
                    raise KeyError(
                        f"[Task {task_id}] waypoint ij {e} not in candidate_ijs"
                    )
                combo_full = combo_local + 1
                full_indices = np.concatenate(
                    [
                        np.asarray([0], dtype=np.int32),
                        combo_full,
                        np.asarray(
                            [len(full_anchor_xys) - 1], dtype=np.int32
                        ),
                    ]
                )
                T_sub = temporal_full_matrix[np.ix_(full_indices, full_indices)].astype(
                    np.float64, copy=False
                )

                assignment = assign_distinct_nodes_from_rankings(
                    sampled_graph_cache,
                    graph_ranking_cache,
                    query_indices=full_indices,
                    priority_order=local_priority_order,
                )
                node_indices = np.asarray(
                    assignment["node_indices"], dtype=np.int32
                )
                G_sub = graph_shortest_dists[np.ix_(node_indices, node_indices)].astype(
                    np.float64, copy=False
                )

                if not np.all(np.isfinite(G_sub)) or not np.all(np.isfinite(T_sub)):
                    continue

                feats = _compute_features(T_sub, G_sub)
                feats["task_id"] = task_id
                feats["combo_idx_in_task"] = combo_idx
                feats["w1_i"], feats["w1_j"] = ijs[0]
                feats["w2_i"], feats["w2_j"] = ijs[1]
                feats["w3_i"], feats["w3_j"] = ijs[2]
                rows.append(feats)

                if (combo_idx + 1) % 1000 == 0:
                    print(
                        f"  [Task {task_id}] {combo_idx + 1}/{n_groups} processed",
                        flush=True,
                    )
    finally:
        env.close()

    if not rows:
        raise RuntimeError("No feature rows were produced; aborting")

    df = pd.DataFrame(rows)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "features.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}  rows={len(df)}  cols={len(df.columns)}", flush=True)
    pickle_path = output_dir / "features.pkl"
    df.to_pickle(pickle_path)
    print(f"[SAVED] {pickle_path}", flush=True)
    try:
        parquet_path = output_dir / "features.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"[SAVED] {parquet_path}", flush=True)
    except Exception as e:
        print(f"[WARN] parquet save skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
