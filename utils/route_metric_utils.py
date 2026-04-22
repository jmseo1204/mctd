from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np


def compute_pairwise_temporal_distance_matrix(
    algo,
    src_xys: np.ndarray,
    dst_xys: np.ndarray,
    gamma: float,
) -> np.ndarray:
    src_xys = np.asarray(src_xys, dtype=np.float32).reshape(-1, 2)
    dst_xys = np.asarray(dst_xys, dtype=np.float32).reshape(-1, 2)
    if src_xys.size == 0 or dst_xys.size == 0:
        return np.zeros((len(src_xys), len(dst_xys)), dtype=np.float32)

    obs_dim = len(algo.obs_dim_indices)
    if getattr(algo, "_hilp_ref_obs", None) is not None and len(algo._hilp_ref_obs) > max(algo.obs_dim_indices):
        base_obs = np.asarray(algo._hilp_ref_obs, dtype=np.float32)[algo.obs_dim_indices].copy()
    else:
        base_obs = np.zeros(obs_dim, dtype=np.float32)

    n_src = int(len(src_xys))
    n_dst = int(len(dst_xys))
    obs_batch = np.tile(base_obs[None, :], (n_src * n_dst, 1)).astype(np.float32)
    goal_batch = np.tile(base_obs[None, :], (n_src * n_dst, 1)).astype(np.float32)
    obs_batch[:, algo.pos_dim_indices] = np.repeat(src_xys, n_dst, axis=0)
    goal_batch[:, algo.pos_dim_indices] = np.tile(dst_xys, (n_src, 1))

    hilp_values = algo._compute_hilp_values(obs_batch, goal_batch).cpu().numpy()
    temporal_dist = algo.emb_dist_to_temporal_dist((-hilp_values), gamma=gamma)
    return np.asarray(temporal_dist, dtype=np.float32).reshape(n_src, n_dst)


def anchor_short_label(anchor_idx: int, n_anchors: int) -> str:
    if anchor_idx == 0:
        return "S"
    if anchor_idx == n_anchors - 1:
        return "G"
    return f"W{anchor_idx}"


def solve_fixed_endpoint_hamiltonian_path(anchor_shortest_dists: np.ndarray) -> dict[str, Any]:
    anchor_shortest_dists = np.asarray(anchor_shortest_dists, dtype=np.float32)
    n_anchors = int(anchor_shortest_dists.shape[0])
    if anchor_shortest_dists.shape != (n_anchors, n_anchors):
        raise ValueError("anchor_shortest_dists must be a square matrix")
    if n_anchors < 2:
        raise ValueError("At least start and goal anchors are required")

    waypoint_indices = list(range(1, n_anchors - 1))
    best_order = None
    best_cost = np.inf
    best_step_costs = None

    for waypoint_order in permutations(waypoint_indices):
        order = np.asarray((0,) + waypoint_order + (n_anchors - 1,), dtype=np.int32)
        step_costs = anchor_shortest_dists[order[:-1], order[1:]]
        if not np.all(np.isfinite(step_costs)):
            continue
        total_cost = float(np.sum(step_costs, dtype=np.float64))
        if total_cost < best_cost:
            best_cost = total_cost
            best_order = order
            best_step_costs = np.asarray(step_costs, dtype=np.float32)

    if best_order is None or best_step_costs is None:
        return {
            "feasible": False,
            "anchor_order": np.zeros((0,), dtype=np.int32),
            "step_costs": np.zeros((0,), dtype=np.float32),
            "total_cost": np.inf,
            "route_text": "no feasible Hamiltonian path",
        }

    route_text = " -> ".join(anchor_short_label(int(idx), n_anchors) for idx in best_order.tolist())
    return {
        "feasible": True,
        "anchor_order": best_order,
        "step_costs": best_step_costs,
        "total_cost": float(best_cost),
        "route_text": route_text,
    }


def solve_fixed_endpoint_hamiltonian_path_with_second_best(
    anchor_shortest_dists: np.ndarray,
) -> dict[str, Any]:
    anchor_shortest_dists = np.asarray(anchor_shortest_dists, dtype=np.float32)
    n_anchors = int(anchor_shortest_dists.shape[0])
    if anchor_shortest_dists.shape != (n_anchors, n_anchors):
        raise ValueError("anchor_shortest_dists must be a square matrix")
    if n_anchors < 2:
        raise ValueError("At least start and goal anchors are required")

    waypoint_indices = list(range(1, n_anchors - 1))
    best_order = None
    best_cost = np.inf
    best_step_costs = None
    second_order = None
    second_cost = np.inf
    second_step_costs = None
    feasible_count = 0

    for waypoint_order in permutations(waypoint_indices):
        order = np.asarray((0,) + waypoint_order + (n_anchors - 1,), dtype=np.int32)
        step_costs = anchor_shortest_dists[order[:-1], order[1:]]
        if not np.all(np.isfinite(step_costs)):
            continue
        feasible_count += 1
        total_cost = float(np.sum(step_costs, dtype=np.float64))
        step_costs = np.asarray(step_costs, dtype=np.float32)

        if total_cost < best_cost:
            second_cost = best_cost
            second_order = best_order
            second_step_costs = best_step_costs
            best_cost = total_cost
            best_order = order
            best_step_costs = step_costs
        elif total_cost < second_cost:
            second_cost = total_cost
            second_order = order
            second_step_costs = step_costs

    if best_order is None or best_step_costs is None:
        return {
            "feasible": False,
            "anchor_order": np.zeros((0,), dtype=np.int32),
            "step_costs": np.zeros((0,), dtype=np.float32),
            "total_cost": np.inf,
            "route_text": "no feasible Hamiltonian path",
            "second_anchor_order": np.zeros((0,), dtype=np.int32),
            "second_step_costs": np.zeros((0,), dtype=np.float32),
            "second_total_cost": np.inf,
            "second_route_text": "no feasible Hamiltonian path",
            "second_best_gap": np.inf,
            "num_feasible_paths": 0,
        }

    route_text = " -> ".join(anchor_short_label(int(idx), n_anchors) for idx in best_order.tolist())
    if second_order is None or second_step_costs is None:
        second_route_text = "no second feasible Hamiltonian path"
        second_best_gap = np.inf
        second_anchor_order = np.zeros((0,), dtype=np.int32)
        second_step_costs_out = np.zeros((0,), dtype=np.float32)
        second_total_cost = np.inf
    else:
        second_route_text = " -> ".join(
            anchor_short_label(int(idx), n_anchors) for idx in second_order.tolist()
        )
        second_best_gap = float(second_cost - best_cost)
        second_anchor_order = second_order
        second_step_costs_out = second_step_costs
        second_total_cost = float(second_cost)

    return {
        "feasible": True,
        "anchor_order": best_order,
        "step_costs": best_step_costs,
        "total_cost": float(best_cost),
        "route_text": route_text,
        "second_anchor_order": second_anchor_order,
        "second_step_costs": second_step_costs_out,
        "second_total_cost": second_total_cost,
        "second_route_text": second_route_text,
        "second_best_gap": second_best_gap,
        "num_feasible_paths": int(feasible_count),
    }


def solve_fixed_endpoint_hamiltonian_path_with_forced_adjacency(
    anchor_shortest_dists: np.ndarray,
    forced_adjacent_pairs: list[tuple[int, int]] | None = None,
    ordered_forced_pairs: bool = False,
) -> dict[str, Any]:
    anchor_shortest_dists = np.asarray(anchor_shortest_dists, dtype=np.float32)
    n_anchors = int(anchor_shortest_dists.shape[0])
    if anchor_shortest_dists.shape != (n_anchors, n_anchors):
        raise ValueError("anchor_shortest_dists must be a square matrix")
    if n_anchors < 2:
        raise ValueError("At least start and goal anchors are required")

    normalized_pairs: list[tuple[int, int]] = []
    for pair in (forced_adjacent_pairs or []):
        if len(pair) != 2:
            raise ValueError("Each forced adjacency must contain exactly two anchor indices")
        a, b = int(pair[0]), int(pair[1])
        if a == b:
            raise ValueError("Forced adjacency cannot contain identical endpoints")
        if ordered_forced_pairs:
            normalized_pairs.append((a, b))
        else:
            normalized_pairs.append((min(a, b), max(a, b)))

    waypoint_indices = list(range(1, n_anchors - 1))
    best_order = None
    best_cost = np.inf
    best_step_costs = None

    for waypoint_order in permutations(waypoint_indices):
        order = np.asarray((0,) + waypoint_order + (n_anchors - 1,), dtype=np.int32)
        if ordered_forced_pairs:
            order_pairs = {
                (int(order[i]), int(order[i + 1]))
                for i in range(len(order) - 1)
            }
        else:
            order_pairs = {
                (min(int(order[i]), int(order[i + 1])), max(int(order[i]), int(order[i + 1])))
                for i in range(len(order) - 1)
            }
        if any(pair not in order_pairs for pair in normalized_pairs):
            continue

        step_costs = anchor_shortest_dists[order[:-1], order[1:]]
        if not np.all(np.isfinite(step_costs)):
            continue
        total_cost = float(np.sum(step_costs, dtype=np.float64))
        if total_cost < best_cost:
            best_cost = total_cost
            best_order = order
            best_step_costs = np.asarray(step_costs, dtype=np.float32)

    if best_order is None or best_step_costs is None:
        return {
            "feasible": False,
            "anchor_order": np.zeros((0,), dtype=np.int32),
            "step_costs": np.zeros((0,), dtype=np.float32),
            "total_cost": np.inf,
            "route_text": "no feasible Hamiltonian path",
        }

    route_text = " -> ".join(anchor_short_label(int(idx), n_anchors) for idx in best_order.tolist())
    return {
        "feasible": True,
        "anchor_order": best_order,
        "step_costs": best_step_costs,
        "total_cost": float(best_cost),
        "route_text": route_text,
    }


def batch_solve_fixed_endpoint_hamiltonian_paths(
    full_pairwise_dists: np.ndarray,
    combo_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    full_pairwise_dists = np.asarray(full_pairwise_dists, dtype=np.float32)
    combo_indices = np.asarray(combo_indices, dtype=np.int32)
    if combo_indices.ndim != 2:
        raise ValueError("combo_indices must have shape (num_combos, num_waypoints)")
    if full_pairwise_dists.ndim != 2 or full_pairwise_dists.shape[0] != full_pairwise_dists.shape[1]:
        raise ValueError("full_pairwise_dists must be a square matrix")

    num_combos, num_waypoints = combo_indices.shape
    if num_combos == 0:
        return {
            "feasible_mask": np.zeros((0,), dtype=bool),
            "best_perm_indices": np.zeros((0,), dtype=np.int32),
            "best_costs": np.zeros((0,), dtype=np.float32),
            "route_orders": np.zeros((0, num_waypoints + 2), dtype=np.int32),
        }

    start_index = 0
    goal_index = int(full_pairwise_dists.shape[0] - 1)
    perm_list = list(permutations(range(num_waypoints)))
    best_perm_indices = np.full((num_combos,), -1, dtype=np.int32)
    best_costs = np.full((num_combos,), np.inf, dtype=np.float64)
    best_orders = np.full((num_combos, num_waypoints + 2), -1, dtype=np.int32)

    start_col = np.full((num_combos, 1), start_index, dtype=np.int32)
    goal_col = np.full((num_combos, 1), goal_index, dtype=np.int32)

    for perm_idx, perm in enumerate(perm_list):
        permuted = combo_indices[:, perm]
        route_orders = np.concatenate([start_col, permuted, goal_col], axis=1)
        step_costs = full_pairwise_dists[route_orders[:, :-1], route_orders[:, 1:]]
        feasible = np.all(np.isfinite(step_costs), axis=1)
        total_costs = np.sum(step_costs, axis=1, dtype=np.float64)
        better_mask = feasible & (total_costs < best_costs)
        if not np.any(better_mask):
            continue
        best_perm_indices[better_mask] = int(perm_idx)
        best_costs[better_mask] = total_costs[better_mask]
        best_orders[better_mask] = route_orders[better_mask]

    feasible_mask = best_perm_indices >= 0
    return {
        "feasible_mask": feasible_mask,
        "best_perm_indices": best_perm_indices,
        "best_costs": best_costs.astype(np.float32),
        "route_orders": best_orders,
    }
