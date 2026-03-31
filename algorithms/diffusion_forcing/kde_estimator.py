"""kde_estimator.py
Kernel Density Estimation utilities for trajectory-coverage scoring.

Provides KDEEstimatorMixin — a mixin class whose methods handle:
  - Loading training (x,y) positions from dataset (_load_kde_data_xy)
  - Pre-computing a 2D KDE grid and persisting to pkl (_build_or_load_kde_grid)
  - Fast O(N) bilinear-interpolation KDE score lookup (_get_kde_score_grid)
  - Raw data accessor (_get_kde_data_xy)

Intended to be inherited by DiffusionForcingPlanning alongside DiffusionForcingBase.
All methods reference `self.*` attributes provided by the base class.
"""

from __future__ import annotations

import numpy as np


class KDEEstimatorMixin:
    """Mixin providing kernel-density-estimation helpers."""

    def _load_kde_data_xy(self) -> None:
        """Preload full training positions (x,y) for KDE score computation.

        Called once at __init__ time. Uses the entire dataset without subsampling
        so that the density estimate is as accurate as possible.
        """
        import os
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
        """Pre-compute a 2-D grid of KDE scores and persist as pkl.

        Grid covers the data extent + 3σ margin at 256×256 resolution.
        Subsequent runs load the pkl and skip the O(grid_pts × M) computation.
        The pkl filename encodes sigma and M so stale caches are never reused.
        """
        import os, pickle
        from algorithms.diffusion_forcing.guidance import _kde_score_np as _kde_score

        xy = self._kde_data_xy_cache
        n = len(xy)
        pkl_name = f"{self.dataset}_kde_grid_s{self.kde_sigma}_n{n}.pkl"
        pkl_path = os.path.join(self._kde_save_dir, pkl_name)

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                self._kde_grid_cache = pickle.load(f)
            print(f"[INIT] KDE grid loaded from cache: {pkl_path}", flush=True)
            return

        # Build grid
        margin = self.kde_sigma * 3.0
        x_min, x_max = float(xy[:, 0].min()) - margin, float(xy[:, 0].max()) + margin
        y_min, y_max = float(xy[:, 1].min()) - margin, float(xy[:, 1].max()) + margin
        res = 256
        xs = np.linspace(x_min, x_max, res, dtype=np.float32)
        ys = np.linspace(y_min, y_max, res, dtype=np.float32)
        grid_pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)  # (res*res, 2)

        # Compute in chunks to keep memory bounded (~40 MB per chunk for M=100K)
        chunk = 512
        parts = []
        for i in range(0, len(grid_pts), chunk):
            parts.append(_kde_score(grid_pts[i:i+chunk], xy, self.kde_sigma))
        grid_scores = np.concatenate(parts, axis=0).reshape(res, res, 2)  # (res, res, 2)

        self._kde_grid_cache = {"xs": xs, "ys": ys, "scores": grid_scores}
        with open(pkl_path, "wb") as f:
            pickle.dump(self._kde_grid_cache, f)
        print(f"[INIT] KDE grid built and saved: {pkl_path} (res={res}x{res}, chunks={len(parts)})", flush=True)

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

        s00 = scores[ix,   iy  ]  # (N, 2)
        s10 = scores[ix+1, iy  ]
        s01 = scores[ix,   iy+1]
        s11 = scores[ix+1, iy+1]
        return (s00*(1-tx)*(1-ty) + s10*tx*(1-ty) +
                s01*(1-tx)*ty     + s11*tx*ty)

    def _get_kde_data_xy(self) -> np.ndarray:
        return self._kde_data_xy_cache
