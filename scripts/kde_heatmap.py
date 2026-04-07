"""scripts/kde_heatmap.py
KDE density & score-field heatmap for a single dataset.

Called by scripts/visualize_kde.sh.  All KDE hyperparameters are read from
configurations/algorithm/df_planning.yaml so this script never hard-codes them.

Usage:
  python scripts/kde_heatmap.py --dataset antmaze-large-navigate-v0
  python scripts/kde_heatmap.py --dataset antmaze-large-navigate-v0 --no_show
"""

import argparse
import os
import sys
from pathlib import Path

# ── repo root on sys.path so we can import algorithms.* ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_kde_cfg() -> dict:
    """Read KDE-relevant keys from configurations/algorithm/df_planning.yaml."""
    try:
        import yaml
    except ImportError:
        print("[ERROR] PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)

    yaml_path = _REPO_ROOT / "configurations" / "algorithm" / "df_planning.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    return {
        "kde_sigma":                   float(raw.get("kde_sigma",                   0.2)),
        "kde_grad_thres_sigma_coeff":  float(raw.get("kde_grad_thres_sigma_coeff",  0.0)),
        "kde_sample_ratio":            float(raw.get("kde_sample_ratio",            0.1)),
        "kde_save_dir":                os.path.expanduser(raw.get("kde_save_dir",   "~/.ogbench/data")),
    }


def _draw_maze_overlay(ax, dataset: str, alpha: float = 0.2) -> None:
    """Overlay maze walls on an axes whose coordinates are in world units.

    Coordinate mapping (from plan_viz.py):
        plot_coord p = world / 4 + 1  →  world = (p - 1) * 4
    plot_maze_layout draws walls as Rectangle((i+0.5, j+0.5), 1, 1) in plot
    coords, which corresponds to world rect origin ((i-0.5)*4, (j-0.5)*4)
    with width=4, height=4.
    """
    from matplotlib.patches import Rectangle
    from utils.logging_utils import get_maze_grid

    try:
        maze_grid = get_maze_grid(dataset)
    except Exception:
        return  # unknown maze type — skip silently

    for i, row in enumerate(maze_grid):
        for j, cell in enumerate(row):
            if cell == "#":
                x_world = (i - 0.5) * 4
                y_world = (j - 0.5) * 4
                rect = Rectangle(
                    (x_world, y_world), 4, 4,
                    facecolor="black", edgecolor="none", alpha=alpha,
                    zorder=3,
                )
                ax.add_patch(rect)


def main():
    parser = argparse.ArgumentParser(description="KDE density & score heatmap")
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name, e.g. antmaze-large-navigate-v0")
    parser.add_argument("--quiver_res", type=int, default=60,
                        help="Grid resolution for quiver arrows (default 60)")
    parser.add_argument("--scatter_n",  type=int, default=5000,
                        help="Max scatter points to overlay; 0 = skip (default 5000)")
    parser.add_argument("--out",        default=None,
                        help="Output image path (default: <save_dir>/<dataset>_kde_heatmap.png)")
    parser.add_argument("--no_show",    action="store_true",
                        help="Skip plt.show() (for headless / SSH use)")
    args = parser.parse_args()

    # ── Load hyperparams from yaml ────────────────────────────────────────────
    cfg = _load_kde_cfg()
    sigma        = cfg["kde_sigma"]
    thres_coeff  = cfg["kde_grad_thres_sigma_coeff"]
    sample_ratio = cfg["kde_sample_ratio"]
    save_dir     = cfg["kde_save_dir"]

    print(f"[CFG] sigma={sigma}  thres_coeff={thres_coeff}  sample_ratio={sample_ratio}  save_dir={save_dir}")

    # ── Load raw observations ─────────────────────────────────────────────────
    npz_path = os.path.join(save_dir, f"{args.dataset}.npz")
    if not os.path.exists(npz_path):
        print(f"[ERROR] npz not found: {npz_path}")
        print("  Download the dataset first (ogbench) or adjust kde_save_dir in df_planning.yaml.")
        sys.exit(1)

    import numpy as np
    data = np.load(npz_path)
    obs_all = data["observations"].astype(np.float32)   # (N, obs_dim)
    xy_all  = obs_all[:, :2]                            # (N, 2)
    print(f"[INFO] Loaded {len(xy_all):,} observations from {npz_path}")

    # ── Subsample (same RNG seed as kde_estimator.py) ─────────────────────────
    n_use = max(1, int(len(xy_all) * sample_ratio))
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(xy_all), n_use, replace=False)
    xy_kde = xy_all[idx].astype(np.float32)
    print(f"[INFO] Using {n_use:,} / {len(xy_all):,} points (sample_ratio={sample_ratio})")

    # ── Build / load KDE grid (reuses kde_estimator.build_or_load_kde_grid) ───
    from algorithms.diffusion_forcing.kde_estimator import build_or_load_kde_grid
    grid = build_or_load_kde_grid(
        data_xy=xy_kde,
        sigma=sigma,
        dataset=args.dataset,
        save_dir=save_dir,
    )
    xs, ys          = grid["xs"], grid["ys"]
    log_dens_grid   = grid["log_dens"]   # (res, res)
    scores_grid     = grid["scores"]     # (res, res, 2)
    res             = len(xs)

    # ── Build coarser quiver grid via bilinear interp on the cached grid ──────
    xs_q = np.linspace(xs[0], xs[-1], args.quiver_res, dtype=np.float32)
    ys_q = np.linspace(ys[0], ys[-1], args.quiver_res, dtype=np.float32)
    gx_q, gy_q = np.meshgrid(xs_q, ys_q, indexing="ij")
    query_q = np.stack([gx_q.ravel(), gy_q.ravel()], axis=-1)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    qx = np.clip(query_q[:, 0], xs[0], xs[-1])
    qy = np.clip(query_q[:, 1], ys[0], ys[-1])
    ix = np.clip(((qx - xs[0]) / dx).astype(np.int32), 0, res - 2)
    iy = np.clip(((qy - ys[0]) / dy).astype(np.int32), 0, res - 2)
    tx = ((qx - xs[ix]) / dx)[:, None]
    ty = ((qy - ys[iy]) / dy)[:, None]
    s00 = scores_grid[ix,     iy    ]
    s10 = scores_grid[ix + 1, iy    ]
    s01 = scores_grid[ix,     iy + 1]
    s11 = scores_grid[ix + 1, iy + 1]
    score_q = s00*(1-tx)*(1-ty) + s10*tx*(1-ty) + s01*(1-tx)*ty + s11*tx*ty  # (Q², 2)

    sx = score_q[:, 0].reshape(args.quiver_res, args.quiver_res)
    sy = score_q[:, 1].reshape(args.quiver_res, args.quiver_res)

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"KDE  |  dataset: {args.dataset}\n"
        f"σ={sigma}  N={n_use:,} pts  grid={res}²",
        fontsize=11,
    )

    # — Panel 1: log-density heatmap + maze overlay + scatter —
    ax = axes[0]
    ax.imshow(
        log_dens_grid.T,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="plasma",
        aspect="auto",
        zorder=1,
    )
    # colorbar — use a ScalarMappable so colorbar doesn't require the imshow handle
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    sm = cm.ScalarMappable(
        cmap="plasma",
        norm=mcolors.Normalize(vmin=log_dens_grid.min(), vmax=log_dens_grid.max()),
    )
    fig.colorbar(sm, ax=ax, label="log p(x,y)")
    _draw_maze_overlay(ax, args.dataset, alpha=0.2)
    if args.scatter_n > 0:
        n_sc   = min(args.scatter_n, len(xy_all))
        idx_sc = rng.choice(len(xy_all), n_sc, replace=False)
        ax.scatter(
            xy_all[idx_sc, 0], xy_all[idx_sc, 1],
            s=0.3, alpha=0.2, color="white", linewidths=0,
            label=f"data (n={n_sc:,})",
            zorder=2,
        )
        ax.legend(loc="upper right", fontsize=8, markerscale=10)
    ax.set_xlabel("x (world)")
    ax.set_ylabel("y (world)")
    ax.set_title("log p(x,y)  —  KDE density")

    # — Panel 2: λ·∇log p quiver + maze overlay —
    ax2 = axes[1]
    ax2.imshow(
        log_dens_grid.T,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="plasma",
        aspect="auto",
        alpha=0.65,
        zorder=1,
    )
    _draw_maze_overlay(ax2, args.dataset, alpha=0.5)

    # Gradient magnitude
    mag = np.sqrt(sx**2 + sy**2) + 1e-8   # (quiver_res, quiver_res)

    # Normalize by mean & std to get standardized gradient lengths
    mag_mean = float(mag.mean())
    mag_std  = float(mag.std()) + 1e-8
    mag_normalized = (mag - mag_mean) / mag_std  # z-score: mean=0, std=1
    
    # Filter: set magnitude to 0 where mag < μ + coeff * σ  (mirrors guidance.py:151)
    threshold = mag_mean + thres_coeff * mag_std
    mask_below_threshold = mag < threshold
    mag_filtered = mag.copy()
    mag_filtered[mask_below_threshold] = 0.0
    
    # Count filtered gradients
    n_filtered = mask_below_threshold.sum()
    n_total = mask_below_threshold.size
    print(f"[INFO] Filtered {n_filtered:,} / {n_total:,} gradients "
          f"({100*n_filtered/n_total:.1f}%) below threshold μ = {threshold:.4f}")
    
    # Scale to [0.1, 3.0] range for better visibility
    # Using percentile-based clipping to handle outliers
    mag_p05 = np.percentile(mag_normalized, 5)
    mag_p95 = np.percentile(mag_normalized, 95)
    mag_vis = np.clip(mag_normalized, mag_p05, mag_p95)
    # Linearly rescale to [0.1, 3.0]
    mag_vis = 0.1 + 2.9 * (mag_vis - mag_vis.min()) / (mag_vis.max() - mag_vis.min() + 1e-8)
    
    # Apply filter mask to visualization magnitude
    mag_vis[mask_below_threshold] = 0.0

    # Arrow vectors: unit direction × normalized length
    U = (sx / mag) * mag_vis
    V = (sy / mag) * mag_vis

    ax2.quiver(
        gx_q, gy_q,
        U, V,
        mag,                          # color by raw λ·|∇log p|
        cmap="cool",
        scale=args.quiver_res * 0.5,  # tighter scale for denser grid
        width=0.002,
        headwidth=3,
        headlength=4,
        alpha=0.85,
        zorder=4,
    )
    ax2.set_xlabel("x (world)")
    ax2.set_ylabel("y (world)")
    ax2.set_title(
        f"∇log p  —  KDE score field\n"
        f"arrow length: z-score normalized (μ={mag_mean:.3f}, σ={mag_std:.3f}), scaled to [0.1, 3.0]\n"
        f"filtered: {n_filtered:,}/{n_total:,} ({100*n_filtered/n_total:.1f}%) gradients < μ+{thres_coeff}σ"
    )

    plt.tight_layout()

    out_path = args.out or os.path.join(save_dir, f"{args.dataset}_kde_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved → {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
