"""Utilities for uncertainty estimation.

This module provides two uncertainty estimators:

1. ``compute_uncertainty_from_embeddings``: Radial–angular entropy estimator
   described in ``final_uncertainty_theory_complete.md``.  Local entropy terms
   are computed from Euclidean geometry in embedding space; the final
   multiplier ``T_curr`` uses embedding-to-temporal-distance conversion.

2. ``compute_uncertainty_variance``: Temporal-distance variance estimator
   described in ``uncertainty_variance_theory.md``.  All spread quantities are
   computed directly in temporal-distance space using three variance terms:
   internal pairwise variance (``V_int``), external target variance
   (``V_ext_g``), and external current variance (``V_ext_c``).

3. ``cluster_tail_by_temporal_dist``: Complete-linkage hierarchical clustering
   of tail embeddings using temporal distance as the pairwise metric.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _offdiag_pairwise_average(pairwise_values: np.ndarray) -> float:
    """Return (2N(N-1))^-1 * sum_{i != j} pairwise_values[i, j]."""
    n = pairwise_values.shape[0]
    if n < 2:
        raise ValueError("Uncertainty estimation requires at least two tail samples.")
    total = float(np.sum(pairwise_values) - np.trace(pairwise_values))
    return total / (2.0 * n * (n - 1))


def _pairwise_squared_distances(z: np.ndarray) -> np.ndarray:
    """Compute squared Euclidean distances for all sample pairs."""
    sq_norms = np.sum(z * z, axis=1, keepdims=True)
    pairwise_sq = sq_norms + sq_norms.T - 2.0 * (z @ z.T)
    return np.maximum(pairwise_sq, 0.0)


def compute_uncertainty_from_embeddings(
    z_curr: np.ndarray,
    z_goal: np.ndarray,
    z_tail: np.ndarray,
    temporal_dist_fn: Callable[[np.ndarray], np.ndarray],
    lambda_weight: float = 1.0,
    eta_weight: float = 1.0,
    eps: float = 1e-8,
) -> dict:
    """Compute uncertainty from embedding-space samples.

    Args:
        z_curr: Current-node embedding with shape (D,).
        z_goal: Goal embedding with shape (D,).
        z_tail: Tail embeddings with shape (N, D).
        temporal_dist_fn: Maps embedding distance(s) to temporal distance(s).
        lambda_weight: Weight on the angular entropy surrogate.
        eta_weight: Weight on the geometry correction.
        eps: Numerical-stability constant.

    Returns:
        A diagnostics dictionary containing the final uncertainty score and the
        intermediate radial-angular terms.
    """
    z_curr = np.asarray(z_curr, dtype=np.float64)
    z_goal = np.asarray(z_goal, dtype=np.float64)
    z_tail = np.asarray(z_tail, dtype=np.float64)

    if z_tail.ndim != 2:
        raise ValueError(f"Expected z_tail to have shape (N, D), got {z_tail.shape!r}.")

    num_samples, emb_dim = z_tail.shape
    if num_samples < 2:
        raise ValueError(
            f"Uncertainty estimation requires at least two tail samples, got {num_samples}."
        )

    emb_dist_curr = float(np.linalg.norm(z_goal - z_curr))

    radial_vectors = z_tail - z_goal[None, :]
    radii = np.linalg.norm(radial_vectors, axis=-1)

    # S_R equals the unbiased sample variance of the goal distances.
    s_r = float(np.var(radii, ddof=1))

    pairwise_sq = _pairwise_squared_distances(z_tail)
    radial_diff_sq = (radii[:, None] - radii[None, :]) ** 2
    residual_sq = np.maximum(pairwise_sq - radial_diff_sq, 0.0)
    normalized_residual = residual_sq / (radii[:, None] * radii[None, :] + eps)

    s_u = _offdiag_pairwise_average(residual_sq)
    s_a = _offdiag_pairwise_average(normalized_residual)

    h_r = 0.5 * math.log(2.0 * math.pi * math.e * (s_r + eps))

    angular_dim = emb_dim - 1
    if angular_dim > 0:
        h_a = 0.5 * angular_dim * math.log(
            (math.pi * math.e / angular_dim) * (s_a + eps)
        )
        c_geom_hat = angular_dim * float(np.mean(np.log(radii + eps)))
    else:
        h_a = 0.0
        c_geom_hat = 0.0

    h_local = h_r + lambda_weight * h_a + eta_weight * c_geom_hat
    t_curr = float(np.asarray(temporal_dist_fn(np.asarray(emb_dist_curr))).item())
    u = t_curr * h_local

    return {
        "U": u,
        "H_local": h_local,
        "H_R": h_r,
        "H_A": h_a,
        "C_geom_hat": c_geom_hat,
        "S_R": s_r,
        "S_A": s_a,
        "S_U": s_u,
        "T_curr": t_curr,
        "emb_dist_curr": emb_dist_curr,
        "num_samples": num_samples,
        "emb_dim": emb_dim,
        "angular_dim": angular_dim,
        "degenerate": False,
    }


# ---------------------------------------------------------------------------
# Temporal-distance variance estimator
# ---------------------------------------------------------------------------

def compute_uncertainty_variance(
    z_curr: np.ndarray,
    z_goal: np.ndarray,
    z_tail: np.ndarray,
    temporal_dist_fn: Callable[[np.ndarray], np.ndarray],
    lambda_weight: float = 1.0,
    eta_weight: float = 1.0,
) -> dict:
    """Compute uncertainty via temporal-distance variance decomposition.

    Implements the three-term variance estimator described in
    ``uncertainty_variance_theory.md``::

        U = V_int + lambda_weight * V_ext_g + eta_weight * V_ext_c

    where:
      - ``V_int``   : unbiased variance of all pairwise temporal distances
                      between tail embeddings (requires N >= 3, i.e. M >= 3).
      - ``V_ext_g`` : unbiased variance of temporal distances from goal to
                      each tail.
      - ``V_ext_c`` : unbiased variance of temporal distances from current
                      node to each tail.

    All distances are computed as embedding-space L2 norms converted to
    temporal distances via ``temporal_dist_fn``.

    Args:
        z_curr: Current-node embedding, shape ``(D,)``.
        z_goal: Goal/target embedding, shape ``(D,)``.
        z_tail: Tail embeddings, shape ``(N, D)``.
        temporal_dist_fn: Maps embedding distance(s) (scalar or array) to
            temporal distance(s).  Should handle arbitrary numpy array shapes.
        lambda_weight: Weight on ``V_ext_g``.
        eta_weight: Weight on ``V_ext_c``.

    Returns:
        dict with keys ``"U"``, ``"V_int"``, ``"V_ext_g"``, ``"V_ext_c"``,
        ``"num_samples"``, ``"num_pairs"``.

    Raises:
        ValueError: If ``z_tail`` does not have shape ``(N, D)``.
        AssertionError: If ``N < 3`` (minimum required for ``V_int``).
    """
    z_curr = np.asarray(z_curr, dtype=np.float64)
    z_goal = np.asarray(z_goal, dtype=np.float64)
    z_tail = np.asarray(z_tail, dtype=np.float64)

    if z_tail.ndim != 2:
        raise ValueError(f"Expected z_tail shape (N, D), got {z_tail.shape!r}.")

    num_samples = z_tail.shape[0]
    assert num_samples >= 3, (
        f"compute_uncertainty_variance requires N >= 3 tail samples, got {num_samples}. "
        "Increase fast_sampling_multiple or the number of guidance scales."
    )

    # --- External target variance: temporal dist from goal to each tail ---
    emb_dist_goal = np.linalg.norm(z_tail - z_goal[None, :], axis=-1)  # (N,)
    t_ext_g = np.asarray(temporal_dist_fn(emb_dist_goal), dtype=np.float64)  # (N,)
    v_ext_g = float(np.var(t_ext_g, ddof=1))

    # --- External current variance: temporal dist from curr to each tail ---
    emb_dist_curr = np.linalg.norm(z_tail - z_curr[None, :], axis=-1)  # (N,)
    t_ext_c = np.asarray(temporal_dist_fn(emb_dist_curr), dtype=np.float64)  # (N,)
    v_ext_c = float(np.var(t_ext_c, ddof=1))

    # --- Internal pairwise variance: temporal dist between all tail pairs ---
    # Compute all pairwise L2 distances using the squared-distance helper,
    # then take the square root to get L2 norms before applying temporal_dist_fn.
    pairwise_sq = _pairwise_squared_distances(z_tail)  # (N, N)
    # Extract upper-triangle indices (i < j), giving M = N*(N-1)/2 pairs.
    idx_i, idx_j = np.triu_indices(num_samples, k=1)
    pairwise_emb_dist = np.sqrt(pairwise_sq[idx_i, idx_j])  # (M,)
    d_int = np.asarray(temporal_dist_fn(pairwise_emb_dist), dtype=np.float64)  # (M,)
    num_pairs = len(d_int)
    assert num_pairs >= 3, (
        f"V_int requires at least 3 pairs (N >= 3), got M={num_pairs} (N={num_samples})."
    )
    v_int = float(np.var(d_int, ddof=1))

    variance_sum = v_int + lambda_weight * v_ext_g + eta_weight * v_ext_c

    # Compute T_curr: temporal distance from current node to goal (same as in
    # compute_uncertainty_from_embeddings).
    emb_dist_curr = float(np.linalg.norm(z_goal - z_curr))
    t_curr = float(np.asarray(temporal_dist_fn(np.asarray(emb_dist_curr))).item())

    # u = math.log(variance_sum + 1e-8) * t_curr
    u = math.log(v_int + 1e-8) * t_curr

    return {
        "U": u,
        "V_int": v_int,
        "V_ext_g": v_ext_g,
        "V_ext_c": v_ext_c,
        "variance_sum": variance_sum,
        "T_curr": t_curr,
        "emb_dist_curr": emb_dist_curr,
        "num_samples": num_samples,
        "num_pairs": num_pairs,
    }


# ---------------------------------------------------------------------------
# Complete-linkage hierarchical clustering on temporal distance
# ---------------------------------------------------------------------------

def cluster_tail_by_temporal_dist(
    z_tail: np.ndarray,
    temporal_dist_fn: Callable[[np.ndarray], np.ndarray],
    max_intra_dist: float,
) -> np.ndarray:
    """Cluster tail embeddings using complete-linkage hierarchical clustering.

    Pairwise distances are measured in temporal-distance space: embedding L2
    distances are first computed and then converted via ``temporal_dist_fn``
    (i.e. ``emb_dist_to_temporal_dist``).  The dendrogram is cut at
    ``max_intra_dist`` so that the maximum temporal distance between any two
    members within a cluster does not exceed that threshold.

    Args:
        z_tail: Tail embeddings, shape ``(N, D)``.
        temporal_dist_fn: Maps embedding L2 distances (scalar or array) to
            temporal distances.  Should accept arbitrary numpy array shapes.
        max_intra_dist: Cut threshold in temporal-distance units (e.g. 20.0
            means at most 20 planning steps apart within a cluster).

    Returns:
        cluster_labels: Integer array of shape ``(N,)``, 0-indexed cluster
            assignments.  All N samples are assigned a cluster; if N == 1 the
            single sample always receives label 0.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    z_tail = np.asarray(z_tail, dtype=np.float64)
    n = z_tail.shape[0]

    if n == 1:
        return np.zeros(1, dtype=int)

    # Pairwise L2 distances → upper-triangle condensed form
    pairwise_sq = _pairwise_squared_distances(z_tail)
    idx_i, idx_j = np.triu_indices(n, k=1)
    condensed_emb_dist = np.sqrt(np.maximum(pairwise_sq[idx_i, idx_j], 0.0))

    # Convert to temporal distances
    condensed_td = np.asarray(
        temporal_dist_fn(condensed_emb_dist), dtype=np.float64
    )

    # Complete-linkage hierarchical clustering
    Z = linkage(condensed_td, method="complete")

    # Cut dendrogram at max_intra_dist threshold
    labels = fcluster(Z, t=max_intra_dist, criterion="distance")

    # scipy returns 1-indexed labels; convert to 0-indexed
    return (labels - 1).astype(int)
