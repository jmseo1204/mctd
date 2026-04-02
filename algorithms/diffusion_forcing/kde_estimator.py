"""kde_estimator.py
Kernel Density Estimation utilities for trajectory-coverage scoring.

Provides:
  build_or_load_kde_grid()  — standalone function (importable from scripts)
  KDEEstimatorMixin         — mixin class that delegates to the above

pkl filename format (all params self-contained):
  {dataset}_kde_grid_s{sigma}_n{n}_r{res}.pkl
  e.g. antmaze-large-navigate-v0_kde_grid_s0.2_n100000_r256.pkl
"""

from __future__ import annotations

import os
import pickle

import numpy as np


# ─── Module-level standalone function (reusable from scripts) ─────────────────

def build_or_load_kde_grid(
    data_xy: np.ndarray,
    sigma: float,
    dataset: str,
    save_dir: str,
    res: int = 256,
) -> dict:
    """Build (or load from cache) a 2-D grid of KDE scores and log-densities.

    Grid covers the data extent + 3σ margin at ``res × res`` resolution.
    Results are persisted as a pkl whose filename encodes all parameters so
    stale caches are never reused without manual deletion.

    pkl filename: ``{dataset}_kde_grid_s{sigma}_n{n}_r{res}.pkl``

    Args:
        data_xy:  (M, 2) float32 training positions, already subsampled.
        sigma:    KDE bandwidth (world units).
        dataset:  Dataset name string — used as part of the pkl filename.
        save_dir: Directory where the pkl is saved/loaded.
        res:      Grid resolution per axis (default 256).

    Returns:
        dict with keys:
            "xs"       — (res,) x grid coordinates
            "ys"       — (res,) y grid coordinates
            "scores"   — (res, res, 2) ∇log p score field
            "log_dens" — (res, res) log-density field  (for heatmap)
    """
    from algorithms.diffusion_forcing.guidance import _kde_score_np as _kde_score

    n = len(data_xy)
    pkl_name = f"{dataset}_kde_grid_s{sigma}_n{n}_r{res}.pkl"
    pkl_path = os.path.join(save_dir, pkl_name)

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            grid = pickle.load(f)
        # Backward compat: old pkls may lack log_dens — recompute if missing
        if "log_dens" not in grid:
            print(f"[KDE] Cache missing log_dens, rebuilding: {pkl_path}", flush=True)
        else:
            print(f"[KDE] Loaded grid from cache: {pkl_path}", flush=True)
            return grid

    # ── Build grid ─────────────────────────────────────────────────────────────
    margin = sigma * 3.0
    x_min = float(data_xy[:, 0].min()) - margin
    x_max = float(data_xy[:, 0].max()) + margin
    y_min = float(data_xy[:, 1].min()) - margin
    y_max = float(data_xy[:, 1].max()) + margin
    xs = np.linspace(x_min, x_max, res, dtype=np.float32)
    ys = np.linspace(y_min, y_max, res, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (res*res, 2)

    from tqdm import tqdm

    chunk = 512
    n_chunks = (len(grid_pts) + chunk - 1) // chunk

    # Score field ∇log p — chunked to keep memory bounded
    score_parts = []
    for i in tqdm(range(0, len(grid_pts), chunk), total=n_chunks,
                  desc="KDE score field", unit="chunk", leave=True):
        score_parts.append(_kde_score(grid_pts[i:i + chunk], data_xy, sigma))
    grid_scores = np.concatenate(score_parts, axis=0).reshape(res, res, 2)

    # Log-density field — logsumexp of Gaussian kernels
    log_dens_parts = []
    for i in tqdm(range(0, len(grid_pts), chunk), total=n_chunks,
                  desc="KDE log-density ", unit="chunk", leave=True):
        q = grid_pts[i:i + chunk]                       # (C, 2)
        diff = data_xy[None, :, :] - q[:, None, :]      # (C, M, 2)
        sq   = (diff ** 2).sum(-1)                       # (C, M)
        logw = -sq / (2.0 * sigma ** 2)
        lmax = logw.max(-1, keepdims=True)
        log_dens_parts.append(
            np.log(np.exp(logw - lmax).sum(-1)) + lmax[:, 0]
        )
    grid_log_dens = np.concatenate(log_dens_parts, axis=0).reshape(res, res)

    grid = {"xs": xs, "ys": ys, "scores": grid_scores, "log_dens": grid_log_dens}
    os.makedirs(save_dir, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(grid, f)
    print(
        f"[KDE] Grid built and saved: {pkl_path} "
        f"(res={res}×{res}, chunks={len(score_parts)})",
        flush=True,
    )
    return grid


# ─── Mixin (delegates to the standalone function above) ───────────────────────

class KDEEstimatorMixin:
    """Mixin providing kernel-density-estimation helpers."""

    def _load_kde_data_xy(self) -> None:
        """Preload full training positions (x,y) for KDE score computation.

        Called once at __init__ time. Uses the entire dataset without subsampling
        so that the density estimate is as accurate as possible.
        """
        path = os.path.join(self._kde_save_dir, f"{self.dataset}.npz")
        xy = np.load(path)["observations"][:, :2].astype(np.float32)
        if self.kde_sample_ratio < 1.0:
            n = max(1, int(len(xy) * self.kde_sample_ratio))
            xy = xy[np.random.default_rng(42).choice(len(xy), n, replace=False)]
        self._kde_data_xy_cache = xy
        print(f"[INIT] KDE data loaded: {len(xy):,} samples from {path}", flush=True)
        # Pre-compute spatial grid for O(N) interpolation instead of O(N*M) per-call
        self._build_or_load_kde_grid()

    def _build_or_load_kde_grid(self) -> None:
        """Delegate to module-level build_or_load_kde_grid()."""
        self._kde_grid_cache = build_or_load_kde_grid(
            data_xy=self._kde_data_xy_cache,
            sigma=self.kde_sigma,
            dataset=self.dataset,
            save_dir=self._kde_save_dir,
        )

    def _get_kde_score_grid(self, query_xy: np.ndarray) -> np.ndarray:
        """Fast KDE score via bilinear interpolation on precomputed grid.

        Replaces O(N*M) brute-force with O(N) grid lookup.

        Args:
            query_xy: (N, 2) world coordinates

        Returns:
            (N, 2) score vectors ∇log p
        """
        grid = self._kde_grid_cache
        xs, ys = grid["xs"], grid["ys"]
        scores = grid["scores"]  # (res_x, res_y, 2)

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        qx = np.clip(query_xy[:, 0], xs[0], xs[-1])
        qy = np.clip(query_xy[:, 1], ys[0], ys[-1])

        ix = np.clip(((qx - xs[0]) / dx).astype(np.int32), 0, len(xs) - 2)
        iy = np.clip(((qy - ys[0]) / dy).astype(np.int32), 0, len(ys) - 2)
        tx = ((qx - xs[ix]) / dx)[:, None]
        ty = ((qy - ys[iy]) / dy)[:, None]

        s00 = scores[ix,     iy    ]  # (N, 2)
        s10 = scores[ix + 1, iy    ]
        s01 = scores[ix,     iy + 1]
        s11 = scores[ix + 1, iy + 1]
        return (s00 * (1 - tx) * (1 - ty) + s10 * tx * (1 - ty) +
                s01 * (1 - tx) * ty       + s11 * tx * ty)

    def _get_kde_data_xy(self) -> np.ndarray:
        return self._kde_data_xy_cache
