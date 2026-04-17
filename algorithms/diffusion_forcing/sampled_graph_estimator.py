"""sampled_graph_estimator.py
Sampled-state graph caching utilities for visualization prototypes.

Provides:
  build_or_load_sampled_graph_cache()           — standalone builder/loader
  build_or_load_sampled_graph_cache_from_npz()  — cache-aware dataset loader
  query_nearest_node()                          — nearest sampled node for one 2D query
  precompute_nearest_node_rankings()           — sorted nearest-node rankings for many queries
  assign_distinct_nodes_from_rankings()        — unique node assignment from cached rankings
  query_distinct_nearest_nodes()                — unique nearest nodes for multiple 2D queries
  query_goal_shortest_distances()               — nearest-node + APSP row lookup
  extract_shortest_path_submatrix()             — gather an M x M APSP submatrix
  query_shortest_path_between_node_indices()    — node-index path reconstruction
  query_shortest_path_between_positions()       — nearest-node path reconstruction
  SampledGraphEstimatorMixin                    — lazy-loading helper mixin

Cache payload:
  {dataset}_sampled_graph_r{ratio}_seed{seed}_rad{radius}.pkl

The cache stores:
  - sampled points
  - undirected radius-graph edges
  - all-pairs shortest-path distance matrix
  - next-hop matrix for shortest-path reconstruction
"""

from __future__ import annotations

import heapq
import os
import pickle

import numpy as np


def _float_token(value: float) -> str:
    """Return a compact filename-safe token for a float."""
    return f"{float(value):g}".replace("-", "m")


def _graph_cache_path(
    dataset: str,
    save_dir: str,
    sample_ratio: float,
    edge_radius: float,
    seed: int,
) -> str:
    filename = (
        f"{dataset}_sampled_graph_"
        f"r{_float_token(sample_ratio)}_"
        f"seed{int(seed)}_"
        f"rad{_float_token(edge_radius)}.pkl"
    )
    return os.path.join(save_dir, filename)


def _build_radius_graph(points_xy: np.ndarray, edge_radius: float) -> tuple[np.ndarray, np.ndarray, list[list[tuple[int, float]]]]:
    """Build an undirected radius graph from sampled (x, y) points."""
    n_nodes = len(points_xy)
    if n_nodes == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            [],
        )

    diffs = points_xy[:, None, :] - points_xy[None, :, :]
    dists_sq = np.sum(diffs * diffs, axis=-1, dtype=np.float32)
    radius_sq = float(edge_radius) * float(edge_radius)
    valid_mask = (dists_sq <= radius_sq) & (dists_sq > 0.0)
    upper_mask = np.triu(valid_mask, k=1)

    src, dst = np.nonzero(upper_mask)
    edge_index = np.stack([src, dst], axis=-1).astype(np.int32, copy=False) if len(src) > 0 \
        else np.zeros((0, 2), dtype=np.int32)
    edge_weights = np.sqrt(dists_sq[src, dst]).astype(np.float32, copy=False) if len(src) > 0 \
        else np.zeros((0,), dtype=np.float32)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n_nodes)]
    for (u, v), w in zip(edge_index, edge_weights):
        wf = float(w)
        adjacency[int(u)].append((int(v), wf))
        adjacency[int(v)].append((int(u), wf))

    return edge_index, edge_weights, adjacency


def _all_pairs_shortest_paths(adjacency: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute APSP distances + next-hop matrix with repeated Dijkstra."""
    n_nodes = len(adjacency)
    shortest = np.full((n_nodes, n_nodes), np.inf, dtype=np.float32)
    next_hops = np.full((n_nodes, n_nodes), -1, dtype=np.int32)
    if n_nodes == 0:
        return shortest, next_hops

    for src in range(n_nodes):
        dist = np.full((n_nodes,), np.inf, dtype=np.float64)
        first_hop = np.full((n_nodes,), -1, dtype=np.int32)
        dist[src] = 0.0
        first_hop[src] = src
        pq: list[tuple[float, int]] = [(0.0, src)]
        while pq:
            curr_dist, u = heapq.heappop(pq)
            if curr_dist > dist[u]:
                continue
            for v, weight in adjacency[u]:
                cand = curr_dist + weight
                if cand < dist[v]:
                    dist[v] = cand
                    first_hop[v] = v if u == src else first_hop[u]
                    heapq.heappush(pq, (cand, v))
        shortest[src] = dist.astype(np.float32)
        next_hops[src] = first_hop

    return shortest, next_hops


def build_or_load_sampled_graph_cache(
    data_xy: np.ndarray,
    sample_ratio: float,
    dataset: str,
    save_dir: str,
    edge_radius: float = 3.0,
    seed: int = 42,
) -> dict:
    """Build or load sampled-state radius graph cache.

    Args:
        data_xy:      (M, 2) float32 world-coordinate states.
        sample_ratio: fraction of states sampled into the graph.
        dataset:      dataset name used in the cache filename.
        save_dir:     cache directory.
        edge_radius:  connect undirected edges when Euclidean dist <= radius.
        seed:         deterministic RNG seed for state subsampling.

    Returns:
        dict with sampled points, edge list, APSP matrix, and next-hop matrix.
    """
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")

    os.makedirs(save_dir, exist_ok=True)
    cache_path = _graph_cache_path(
        dataset=dataset,
        save_dir=save_dir,
        sample_ratio=sample_ratio,
        edge_radius=edge_radius,
        seed=seed,
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        required_keys = {
            "points_xy",
            "sample_indices",
            "edge_index",
            "edge_weights",
            "shortest_dists",
            "next_hops",
            "sample_ratio",
            "edge_radius",
            "seed",
            "n_total",
        }
        if required_keys.issubset(cache.keys()):
            print(f"[SampledGraph] Loaded cache: {cache_path}", flush=True)
            return cache
        print(f"[SampledGraph] Cache missing required keys, rebuilding: {cache_path}", flush=True)

    n_total = int(len(data_xy))
    n_use = max(1, int(n_total * sample_ratio))
    rng = np.random.default_rng(seed)
    sample_indices = np.sort(rng.choice(n_total, size=n_use, replace=False)).astype(np.int64, copy=False)
    points_xy = np.asarray(data_xy[sample_indices], dtype=np.float32)

    edge_index, edge_weights, adjacency = _build_radius_graph(points_xy, edge_radius=edge_radius)
    shortest_dists, next_hops = _all_pairs_shortest_paths(adjacency)

    cache = {
        "points_xy": points_xy,
        "sample_indices": sample_indices,
        "edge_index": edge_index,
        "edge_weights": edge_weights,
        "shortest_dists": shortest_dists,
        "next_hops": next_hops,
        "sample_ratio": float(sample_ratio),
        "edge_radius": float(edge_radius),
        "seed": int(seed),
        "n_total": n_total,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(
        f"[SampledGraph] Built and saved cache: {cache_path} "
        f"(N={len(points_xy):,}, edges={len(edge_index):,})",
        flush=True,
    )
    return cache


def build_or_load_sampled_graph_cache_from_npz(
    npz_path: str,
    dataset: str,
    save_dir: str,
    sample_ratio: float,
    edge_radius: float = 3.0,
    seed: int = 42,
) -> dict:
    """Load sampled-graph cache if present, otherwise build it from a dataset npz."""
    cache_path = _graph_cache_path(
        dataset=dataset,
        save_dir=save_dir,
        sample_ratio=sample_ratio,
        edge_radius=edge_radius,
        seed=seed,
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        required_keys = {
            "points_xy",
            "sample_indices",
            "edge_index",
            "edge_weights",
            "shortest_dists",
            "next_hops",
            "sample_ratio",
            "edge_radius",
            "seed",
            "n_total",
        }
        if required_keys.issubset(cache.keys()):
            print(f"[SampledGraph] Loaded cache: {cache_path}", flush=True)
            return cache
        print(f"[SampledGraph] Cache missing required keys, rebuilding: {cache_path}", flush=True)

    data = np.load(npz_path)
    data_xy = data["observations"][:, :2].astype(np.float32)
    return build_or_load_sampled_graph_cache(
        data_xy=data_xy,
        sample_ratio=sample_ratio,
        dataset=dataset,
        save_dir=save_dir,
        edge_radius=edge_radius,
        seed=seed,
    )


def query_goal_shortest_distances(graph_cache: dict, goal_xy: np.ndarray) -> dict:
    """Map a goal coordinate to its nearest sampled node and return APSP distances."""
    goal_info = query_nearest_node(graph_cache, goal_xy)
    goal_node_index = int(goal_info["node_index"])
    shortest_row = np.asarray(graph_cache["shortest_dists"][goal_node_index], dtype=np.float32)
    return {
        "goal_xy": np.asarray(goal_info["query_xy"], dtype=np.float32),
        "goal_node_index": goal_node_index,
        "goal_node_xy": np.asarray(goal_info["node_xy"], dtype=np.float32),
        "goal_node_euclidean_dist": float(goal_info["node_euclidean_dist"]),
        "shortest_dists": shortest_row,
    }


def _nearest_node_query(graph_cache: dict, query_xy: np.ndarray) -> dict:
    """Return nearest sampled node to a 2D position."""
    points_xy = np.asarray(graph_cache["points_xy"], dtype=np.float32)
    query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1)[:2]
    if len(points_xy) == 0:
        raise ValueError("graph_cache contains no sampled points")

    diff = points_xy - query_xy[None, :]
    euclidean = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32))
    node_index = int(np.argmin(euclidean))
    return {
        "query_xy": query_xy,
        "node_index": node_index,
        "node_xy": points_xy[node_index].copy(),
        "node_euclidean_dist": float(euclidean[node_index]),
    }


def query_nearest_node(graph_cache: dict, query_xy: np.ndarray) -> dict:
    """Return the nearest sampled node to a single 2D query position."""
    return _nearest_node_query(graph_cache, query_xy)


def precompute_nearest_node_rankings(graph_cache: dict, query_xys: np.ndarray) -> dict:
    """Precompute sorted nearest-node rankings for a batch of 2D queries."""
    points_xy = np.asarray(graph_cache["points_xy"], dtype=np.float32)
    query_xys = np.asarray(query_xys, dtype=np.float32)
    if len(points_xy) == 0:
        raise ValueError("graph_cache contains no sampled points")
    if query_xys.size == 0:
        query_xys = np.zeros((0, 2), dtype=np.float32)
    else:
        query_xys = query_xys.reshape(-1, 2)

    diffs = points_xy[None, :, :] - query_xys[:, None, :]
    euclidean = np.sqrt(np.sum(diffs * diffs, axis=-1, dtype=np.float32))
    sorted_node_indices = np.argsort(euclidean, axis=1, kind="stable")
    sorted_euclidean = np.take_along_axis(euclidean, sorted_node_indices, axis=1)
    return {
        "query_xys": query_xys.copy(),
        "sorted_node_indices": sorted_node_indices.astype(np.int32, copy=False),
        "sorted_euclidean": sorted_euclidean.astype(np.float32, copy=False),
    }


def assign_distinct_nodes_from_rankings(
    graph_cache: dict,
    ranking_cache: dict,
    query_indices: np.ndarray,
    *,
    priority_order: np.ndarray | list[int] | None = None,
    query_labels: list[str] | None = None,
) -> dict:
    """Assign distinct sampled nodes for selected queries using cached rankings."""
    points_xy = np.asarray(graph_cache["points_xy"], dtype=np.float32)
    all_query_xys = np.asarray(ranking_cache["query_xys"], dtype=np.float32)
    sorted_node_indices = np.asarray(ranking_cache["sorted_node_indices"], dtype=np.int32)
    sorted_euclidean = np.asarray(ranking_cache["sorted_euclidean"], dtype=np.float32)

    query_indices = np.asarray(query_indices, dtype=np.int32).reshape(-1)
    if query_indices.size == 0:
        raise ValueError("query_indices must contain at least one query index")
    n_queries = int(len(query_indices))

    if priority_order is None:
        priority = np.arange(n_queries, dtype=np.int32)
    else:
        priority = np.asarray(priority_order, dtype=np.int32).reshape(-1)
        if len(priority) != n_queries:
            raise ValueError(
                f"priority_order length {len(priority)} must match number of selected queries {n_queries}"
            )
        if sorted(priority.tolist()) != list(range(n_queries)):
            raise ValueError("priority_order must be a permutation of [0, ..., num_selected_queries-1]")

    if query_labels is not None and len(query_labels) != n_queries:
        raise ValueError(
            f"query_labels length {len(query_labels)} must match number of selected queries {n_queries}"
        )

    node_indices = np.full((n_queries,), -1, dtype=np.int32)
    node_ranks = np.full((n_queries,), -1, dtype=np.int32)
    node_euclidean_dists = np.full((n_queries,), np.inf, dtype=np.float32)
    used_nodes: set[int] = set()

    for local_query_idx in priority.tolist():
        global_query_idx = int(query_indices[local_query_idx])
        ranked_nodes = sorted_node_indices[global_query_idx]
        ranked_dists = sorted_euclidean[global_query_idx]
        chosen_rank = None
        chosen_node = None
        for rank, node_idx in enumerate(ranked_nodes.tolist()):
            if int(node_idx) in used_nodes:
                continue
            chosen_rank = int(rank)
            chosen_node = int(node_idx)
            break
        if chosen_node is None or chosen_rank is None:
            raise RuntimeError(
                f"Failed to find an unused sampled node for query index {global_query_idx}."
            )
        used_nodes.add(chosen_node)
        node_indices[local_query_idx] = chosen_node
        node_ranks[local_query_idx] = chosen_rank
        node_euclidean_dists[local_query_idx] = float(ranked_dists[chosen_rank])

    node_xys = points_xy[node_indices].astype(np.float32, copy=False)
    selected_query_xys = all_query_xys[query_indices].astype(np.float32, copy=False)
    assignments = []
    for local_query_idx in range(n_queries):
        assignments.append(
            {
                "query_index": int(query_indices[local_query_idx]),
                "query_label": None if query_labels is None else query_labels[local_query_idx],
                "query_xy": selected_query_xys[local_query_idx].copy(),
                "node_index": int(node_indices[local_query_idx]),
                "node_xy": node_xys[local_query_idx].copy(),
                "node_euclidean_dist": float(node_euclidean_dists[local_query_idx]),
                "node_rank": int(node_ranks[local_query_idx]),
            }
        )

    return {
        "query_indices": query_indices.copy(),
        "query_xys": selected_query_xys.copy(),
        "priority_order": priority.copy(),
        "node_indices": node_indices,
        "node_ranks": node_ranks,
        "node_xys": node_xys.copy(),
        "node_euclidean_dists": node_euclidean_dists,
        "assignments": assignments,
    }


def query_distinct_nearest_nodes(
    graph_cache: dict,
    query_xys: np.ndarray,
    *,
    priority_order: np.ndarray | list[int] | None = None,
    query_labels: list[str] | None = None,
) -> dict:
    """Assign each query a distinct nearest sampled node.

    Queries are processed in ``priority_order`` and each query takes the nearest
    still-unused sampled node. This implements the requested "fallback to the
    next-nearest node when collisions happen" behavior.
    """
    query_xys = np.asarray(query_xys, dtype=np.float32)
    if query_xys.size == 0:
        raise ValueError("query_xys must contain at least one 2D position")
    query_xys = query_xys.reshape(-1, 2)
    n_queries = int(len(query_xys))
    n_nodes = int(len(np.asarray(graph_cache["points_xy"], dtype=np.float32)))
    if n_nodes == 0:
        raise ValueError("graph_cache contains no sampled points")
    if n_queries > n_nodes:
        raise ValueError(
            f"Cannot assign {n_queries} distinct queries to only {n_nodes} sampled nodes."
        )
    ranking_cache = precompute_nearest_node_rankings(graph_cache, query_xys)
    return assign_distinct_nodes_from_rankings(
        graph_cache,
        ranking_cache,
        query_indices=np.arange(n_queries, dtype=np.int32),
        priority_order=priority_order,
        query_labels=query_labels,
    )


def _reconstruct_node_path(next_hops: np.ndarray, src_idx: int, dst_idx: int) -> np.ndarray:
    """Reconstruct node index path from src to dst using the next-hop matrix."""
    if src_idx == dst_idx:
        return np.asarray([src_idx], dtype=np.int32)

    if int(next_hops[src_idx, dst_idx]) < 0:
        return np.zeros((0,), dtype=np.int32)

    n_nodes = int(next_hops.shape[0])
    path = [int(src_idx)]
    curr = int(src_idx)
    for _ in range(n_nodes):
        nxt = int(next_hops[curr, dst_idx])
        if nxt < 0:
            return np.zeros((0,), dtype=np.int32)
        path.append(nxt)
        curr = nxt
        if curr == int(dst_idx):
            return np.asarray(path, dtype=np.int32)

    raise RuntimeError(
        f"Path reconstruction exceeded {n_nodes} hops; next_hops may be inconsistent "
        f"(src={src_idx}, dst={dst_idx})."
    )


def extract_shortest_path_submatrix(graph_cache: dict, node_indices: np.ndarray) -> np.ndarray:
    """Extract an APSP submatrix for the provided sampled-node indices."""
    shortest_dists = np.asarray(graph_cache["shortest_dists"], dtype=np.float32)
    node_indices = np.asarray(node_indices, dtype=np.int64).reshape(-1)
    if node_indices.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return shortest_dists[np.ix_(node_indices, node_indices)].astype(np.float32, copy=True)


def query_shortest_path_between_node_indices(
    graph_cache: dict,
    src_node_index: int,
    dst_node_index: int,
) -> dict:
    """Query shortest path between two sampled-node indices."""
    next_hops = np.asarray(graph_cache["next_hops"], dtype=np.int32)
    shortest_dists = np.asarray(graph_cache["shortest_dists"], dtype=np.float32)
    points_xy = np.asarray(graph_cache["points_xy"], dtype=np.float32)

    src_idx = int(src_node_index)
    dst_idx = int(dst_node_index)
    n_nodes = int(len(points_xy))
    if not (0 <= src_idx < n_nodes) or not (0 <= dst_idx < n_nodes):
        raise IndexError(
            f"Sampled node indices must be in [0, {n_nodes}), got src={src_idx}, dst={dst_idx}"
        )

    node_path = _reconstruct_node_path(next_hops, src_idx, dst_idx)
    if len(node_path) == 0:
        path_xy = np.zeros((0, 2), dtype=np.float32)
        path_edges = np.zeros((0, 2, 2), dtype=np.float32)
    else:
        path_xy = points_xy[node_path].astype(np.float32, copy=False)
        if len(path_xy) >= 2:
            path_edges = np.stack([path_xy[:-1], path_xy[1:]], axis=1).astype(np.float32, copy=False)
        else:
            path_edges = np.zeros((0, 2, 2), dtype=np.float32)

    return {
        "src_node_index": src_idx,
        "src_node_xy": points_xy[src_idx].copy(),
        "dst_node_index": dst_idx,
        "dst_node_xy": points_xy[dst_idx].copy(),
        "shortest_distance": float(shortest_dists[src_idx, dst_idx]),
        "node_path": node_path,
        "path_xy": path_xy,
        "path_edges": path_edges,
        "reachable": bool(np.isfinite(shortest_dists[src_idx, dst_idx])),
    }


def query_shortest_path_between_positions(
    graph_cache: dict,
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
) -> dict:
    """Query shortest path between two arbitrary 2D positions via nearest sampled nodes."""
    src_info = query_nearest_node(graph_cache, src_xy)
    dst_info = query_nearest_node(graph_cache, dst_xy)
    path_info = query_shortest_path_between_node_indices(
        graph_cache,
        src_node_index=int(src_info["node_index"]),
        dst_node_index=int(dst_info["node_index"]),
    )
    return {
        "src_query": src_info["query_xy"],
        "src_node_index": int(src_info["node_index"]),
        "src_node_xy": src_info["node_xy"],
        "src_node_euclidean_dist": float(src_info["node_euclidean_dist"]),
        "dst_query": dst_info["query_xy"],
        "dst_node_index": int(dst_info["node_index"]),
        "dst_node_xy": dst_info["node_xy"],
        "dst_node_euclidean_dist": float(dst_info["node_euclidean_dist"]),
        "shortest_distance": float(path_info["shortest_distance"]),
        "node_path": np.asarray(path_info["node_path"], dtype=np.int32),
        "path_xy": np.asarray(path_info["path_xy"], dtype=np.float32),
        "path_edges": np.asarray(path_info["path_edges"], dtype=np.float32),
        "reachable": bool(path_info["reachable"]),
    }


class SampledGraphEstimatorMixin:
    """Lazy-loading helper for sampled-state graph caches."""

    def _ensure_sampled_graph_cache_loaded(self) -> None:
        if not hasattr(self, "_sampled_graph_cache"):
            self._load_sampled_graph_cache()

    def _load_sampled_graph_cache(self) -> None:
        dataset_npz = os.path.join(self._sampled_graph_data_dir, f"{self.dataset}.npz")
        self._sampled_graph_cache = build_or_load_sampled_graph_cache_from_npz(
            npz_path=dataset_npz,
            dataset=self.dataset,
            save_dir=self._sampled_graph_cache_dir,
            sample_ratio=self.sampled_graph_sample_ratio,
            edge_radius=self.sampled_graph_edge_radius,
            seed=self.sampled_graph_seed,
        )
        print(
            f"[INIT] Sampled graph cache loaded: {len(self._sampled_graph_cache['points_xy']):,} nodes "
            f"from {dataset_npz}",
            flush=True,
        )

    def _get_sampled_graph_cache(self) -> dict:
        self._ensure_sampled_graph_cache_loaded()
        return self._sampled_graph_cache

    def _query_sampled_graph_goal_distances(self, goal_xy: np.ndarray) -> dict:
        self._ensure_sampled_graph_cache_loaded()
        return query_goal_shortest_distances(self._sampled_graph_cache, goal_xy)

    def _query_sampled_graph_shortest_path(self, src_xy: np.ndarray, dst_xy: np.ndarray) -> dict:
        self._ensure_sampled_graph_cache_loaded()
        return query_shortest_path_between_positions(self._sampled_graph_cache, src_xy, dst_xy)
