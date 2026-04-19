#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.diffusion_forcing.sampled_graph_estimator import (
    assign_distinct_nodes_from_rankings,
    build_sampled_graph_cache_path,
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
    batch_solve_fixed_endpoint_hamiltonian_paths,
    compute_pairwise_temporal_distance_matrix,
    solve_fixed_endpoint_hamiltonian_path_with_second_best,
)


class _FlowStyleList(list):
    """YAML sequence wrapper rendered in flow style."""


class _ReadableYamlDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _represent_flow_style_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_ReadableYamlDumper.add_representer(_FlowStyleList, _represent_flow_style_list)


def _parse_task_ids(raw_value: Optional[str], num_tasks: int) -> list[int]:
    if raw_value in (None, ""):
        return list(range(1, num_tasks + 1))
    task_ids = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        task_id = int(token)
        if not 1 <= task_id <= num_tasks:
            raise ValueError(f"task_id {task_id} is out of range [1, {num_tasks}]")
        task_ids.append(task_id)
    return task_ids


def _parse_bool_arg(raw_value: Optional[str], *, default: bool) -> bool:
    if raw_value in (None, ""):
        return bool(default)
    value = str(raw_value).strip().lower()
    if value in ("1", "true", "t", "yes", "y", "on"):
        return True
    if value in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {raw_value!r}")


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


def _to_repo_relative_or_abs(path: Path) -> str:
    path = path.resolve()
    return os.path.relpath(path, REPO_ROOT)


def _combo_waypoint_ij_groups(
    combo_local_indices: np.ndarray,
    candidate_ijs: np.ndarray,
) -> list[list[list[int]]]:
    groups = []
    for combo_local in combo_local_indices:
        point_list = []
        for point in candidate_ijs[np.asarray(combo_local, dtype=np.int32)].astype(int).tolist():
            point_list.append(_FlowStyleList([int(point[0]), int(point[1])]))
        groups.append(_FlowStyleList(point_list))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Find waypoint groups where temporal and graph Hamiltonian routes disagree.")
    parser.add_argument("--ckpt", required=True, help="Path to model.ckpt")
    parser.add_argument(
        "--feasible-points-path",
        default="configurations/task_overrides/antmaze_giant_feasible_points.yaml",
        help="Repo-relative or absolute YAML path containing feasible waypoint ij candidates",
    )
    parser.add_argument(
        "--out",
        default="configurations/task_overrides/antmaze_giant_waypoints.yaml",
        help="Repo-relative or absolute YAML path to save mismatch waypoint groups",
    )
    parser.add_argument(
        "--graph-cache-path",
        default=None,
        help="Optional repo-relative or absolute sampled-graph cache .pkl path to reuse/create",
    )
    parser.add_argument("--num-waypoints", type=int, default=3, help="Number of waypoint candidates per combination")
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids to search. Default: all tasks")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Optional sampled-graph state sampling ratio override")
    parser.add_argument("--edge_radius", type=float, default=None, help="Optional sampled-graph edge radius override")
    parser.add_argument("--graph_seed", type=int, default=None, help="Optional sampled-graph RNG seed override")
    parser.add_argument("--gamma", type=float, default=0.995, help="Gamma for emb_dist_to_temporal_dist")
    parser.add_argument(
        "--ogbench-enable-reset-perturb",
        default=None,
        help="Whether OGBench resets should keep start/goal perturbation during waypoint search (true/false).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    feasible_points_path = Path(args.feasible_points_path).expanduser()
    if not feasible_points_path.is_absolute():
        feasible_points_path = (REPO_ROOT / feasible_points_path).resolve()
    if not feasible_points_path.is_file():
        raise FileNotFoundError(f"Feasible points file not found: {feasible_points_path}")

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    feasible_env_id, feasible_ijs = _load_feasible_points_payload(feasible_points_path)
    if args.num_waypoints < 1:
        raise ValueError(f"--num-waypoints must be >= 1, got {args.num_waypoints}")

    algo_overrides, dataset_config_name, dataset_meta = _load_training_metadata(ckpt_path)
    ogbench_enable_reset_perturb = _parse_bool_arg(
        args.ogbench_enable_reset_perturb,
        default=bool(algo_overrides.get("ogbench_enable_reset_perturb", True)),
    )
    algo_overrides["ogbench_enable_reset_perturb"] = bool(ogbench_enable_reset_perturb)
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
            f"feasible points env_id {feasible_env_id} does not match checkpoint env_id {algo.env_id}"
        )

    sampled_graph_cfg = _resolve_sampled_graph_cfg(algo, args)
    dataset_name = str(algo.dataset)
    if args.graph_cache_path in (None, ""):
        graph_cache_path_for_yaml = build_sampled_graph_cache_path(
            dataset=dataset_name,
            save_dir=str(sampled_graph_cfg["cache_dir_raw"]),
            sample_ratio=float(sampled_graph_cfg["sample_ratio"]),
            edge_radius=float(sampled_graph_cfg["edge_radius"]),
            seed=int(sampled_graph_cfg["graph_seed"]),
            timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        graph_cache_path = Path(graph_cache_path_for_yaml).expanduser()
    else:
        graph_cache_path_for_yaml = str(args.graph_cache_path)
        graph_cache_path = Path(graph_cache_path_for_yaml).expanduser()
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

    maze_type = algo.env_id.split("-")[1]
    env = algo._make_single_ogbench_env(maze_type)
    try:
        task_ids = _parse_task_ids(args.task_ids, env.num_tasks)
        output_payload = {
            "env_id": str(feasible_env_id),
            "source_feasible_points_path": _to_repo_relative_or_abs(feasible_points_path),
            "sampled_graph_cache_path": str(graph_cache_path_for_yaml),
            "ogbench_enable_reset_perturb": bool(ogbench_enable_reset_perturb),
            "sampled_graph_cache_params": {
                "dataset": dataset_name,
                "sample_ratio": float(sampled_graph_cfg["sample_ratio"]),
                "edge_radius": float(sampled_graph_cfg["edge_radius"]),
                "seed": int(sampled_graph_cfg["graph_seed"]),
            },
            "num_waypoints": int(args.num_waypoints),
            "active_waypoint_group_idx": None,
            "total_stats": {
                "num_tasks_searched": int(len(task_ids)),
                "total_waypoint_combinations": 0,
                "mismatch_combination_count": 0,
            },
            "tasks": {},
        }

        for task_id in task_ids:
            task_info = env.task_infos[int(task_id) - 1]
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

            if len(candidate_ijs) < args.num_waypoints:
                output_payload["tasks"][int(task_id)] = {
                    "task_stats": {
                        "candidate_point_count": int(len(candidate_ijs)),
                        "total_waypoint_combinations": 0,
                        "mismatch_combination_count": 0,
                    },
                    "waypoint_ij_groups": [],
                }
                print(
                    f"[Task {task_id}] skipped: only {len(candidate_ijs)} candidate points after excluding start/goal",
                    flush=True,
                )
                continue

            task_ref_info = _initialize_hilp_reference_obs(algo, int(task_id))
            if task_ref_info is None or task_ref_info.get("start_xy") is None or task_ref_info.get("goal_xy") is None:
                raise RuntimeError(f"Failed to initialize HILP reference observation for task {task_id}")
            start_xy = np.asarray(task_ref_info["start_xy"], dtype=np.float32).reshape(2)
            goal_xy = np.asarray(task_ref_info["goal_xy"], dtype=np.float32).reshape(2)

            full_anchor_xys = np.concatenate(
                [
                    start_xy.reshape(1, 2),
                    candidate_xys,
                    goal_xy.reshape(1, 2),
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            combo_local_indices = np.asarray(
                list(combinations(range(len(candidate_ijs)), int(args.num_waypoints))),
                dtype=np.int32,
            )
            combo_full_anchor_indices = combo_local_indices + 1

            temporal_full_matrix = compute_pairwise_temporal_distance_matrix(
                algo,
                src_xys=full_anchor_xys,
                dst_xys=full_anchor_xys,
                gamma=float(args.gamma),
            )
            temporal_batch = batch_solve_fixed_endpoint_hamiltonian_paths(
                temporal_full_matrix,
                combo_indices=combo_full_anchor_indices,
            )

            graph_ranking_cache = precompute_nearest_node_rankings(sampled_graph_cache, full_anchor_xys)
            graph_shortest_dists = np.asarray(sampled_graph_cache["shortest_dists"], dtype=np.float32)
            local_priority_order = np.asarray(
                [0, int(args.num_waypoints) + 1] + list(range(1, int(args.num_waypoints) + 1)),
                dtype=np.int32,
            )

            mismatch_combo_with_keys: list[tuple[float, np.ndarray]] = []
            graph_feasible_count = 0
            temporal_feasible_count = int(np.sum(temporal_batch["feasible_mask"]))
            for combo_idx, combo_local in enumerate(combo_local_indices):
                if not bool(temporal_batch["feasible_mask"][combo_idx]):
                    continue

                selected_query_indices = np.concatenate(
                    [
                        np.asarray([0], dtype=np.int32),
                        combo_full_anchor_indices[combo_idx],
                        np.asarray([len(full_anchor_xys) - 1], dtype=np.int32),
                    ]
                )
                assignment = assign_distinct_nodes_from_rankings(
                    sampled_graph_cache,
                    graph_ranking_cache,
                    query_indices=selected_query_indices,
                    priority_order=local_priority_order,
                )
                selected_node_indices = np.asarray(assignment["node_indices"], dtype=np.int32)
                graph_submatrix = graph_shortest_dists[np.ix_(selected_node_indices, selected_node_indices)]
                graph_route = solve_fixed_endpoint_hamiltonian_path_with_second_best(graph_submatrix)
                if not bool(graph_route["feasible"]):
                    continue
                graph_feasible_count += 1

                temporal_candidate_order = (
                    np.asarray(temporal_batch["route_orders"][combo_idx, 1:-1], dtype=np.int32) - 1
                )
                graph_candidate_order = combo_local[
                    np.asarray(graph_route["anchor_order"], dtype=np.int32)[1:-1] - 1
                ]
                if not np.array_equal(temporal_candidate_order, graph_candidate_order):
                    # Sort mismatches by how strongly the graph metric prefers its
                    # best Hamiltonian ordering over the runner-up ordering.
                    sort_key = float(graph_route["second_best_gap"])
                    mismatch_combo_with_keys.append((sort_key, combo_local.copy()))

            mismatch_combo_with_keys.sort(key=lambda item: item[0], reverse=True)
            mismatch_combo_locals = [combo_local for _, combo_local in mismatch_combo_with_keys]
            waypoint_ij_groups = _combo_waypoint_ij_groups(
                np.asarray(mismatch_combo_locals, dtype=np.int32).reshape(-1, int(args.num_waypoints))
                if mismatch_combo_locals else np.zeros((0, int(args.num_waypoints)), dtype=np.int32),
                candidate_ijs,
            )
            mismatch_count = len(waypoint_ij_groups)
            total_waypoint_combinations = int(len(combo_local_indices))
            output_payload["tasks"][int(task_id)] = {
                "task_stats": {
                    "candidate_point_count": int(len(candidate_ijs)),
                    "total_waypoint_combinations": total_waypoint_combinations,
                    "mismatch_combination_count": int(mismatch_count),
                },
                "waypoint_ij_groups": waypoint_ij_groups,
            }
            output_payload["total_stats"]["total_waypoint_combinations"] += total_waypoint_combinations
            output_payload["total_stats"]["mismatch_combination_count"] += int(mismatch_count)
            if mismatch_count > 0 and output_payload["active_waypoint_group_idx"] is None:
                output_payload["active_waypoint_group_idx"] = 0
            print(
                f"[Task {task_id}] candidates={len(candidate_ijs)} "
                f"combos={len(combo_local_indices):,} temporal_feasible={temporal_feasible_count:,} "
                f"graph_feasible={graph_feasible_count:,} mismatches={mismatch_count:,}",
                flush=True,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(
                output_payload,
                f,
                Dumper=_ReadableYamlDumper,
                sort_keys=False,
                allow_unicode=False,
                default_flow_style=False,
            )
        print(f"[SAVED] {out_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
