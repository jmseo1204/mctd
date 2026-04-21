#!/usr/bin/env python3
"""Fit ``temporal_dist_overestimate_coeff`` (alpha) to the edge-level
paired (T, G) observations produced by extract_temporal_calibration_pairs.py.

Model:  G  ≈  s * (T + alpha * T**2)

We do not know the global scale s up front (T is in HILP-temporal units,
G is in world units), so for each candidate alpha we fit s in closed form
and evaluate the residual on a scale-invariant criterion.

Reported scoring criteria:
  - log-RMSE          : RMS of log(G) − log(s*T_corr)   (geometric scale)
  - mean abs rel err  : mean |s*T_corr − G| / G
  - Spearman(T_corr, G) : ranking quality (alpha-only effect)

We pick alpha by minimizing log-RMSE (stable, scale-invariant) and report the
chosen value alongside the others, plus diagnostic plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def _fit_scale(T_corr: np.ndarray, G: np.ndarray) -> float:
    """Closed-form least-squares scale s minimizing ||G - s*T_corr||^2."""
    num = float(np.sum(T_corr * G))
    den = float(np.sum(T_corr * T_corr))
    return num / den if den > 0 else float("nan")


def _fit_scale_log(T_corr: np.ndarray, G: np.ndarray) -> float:
    """Scale that minimizes RMS of log(G) − log(s) − log(T_corr)."""
    return float(np.exp(np.mean(np.log(G) - np.log(T_corr))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        default="outputs/waypoint_gap_analysis/temporal_pairs.csv",
        help="CSV from extract_temporal_calibration_pairs.py",
    )
    parser.add_argument(
        "--output-dir", default="outputs/waypoint_gap_analysis"
    )
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=0.05)
    parser.add_argument("--alpha-steps", type=int, default=251)
    args = parser.parse_args()

    pairs_path = Path(args.pairs).expanduser()
    if not pairs_path.is_absolute():
        pairs_path = (REPO_ROOT / pairs_path).resolve()
    df = pd.read_csv(pairs_path)
    df = df.loc[(df["T"] > 0) & (df["G"] > 0)].copy()
    T = df["T"].to_numpy(dtype=np.float64)
    G = df["G"].to_numpy(dtype=np.float64)
    n = len(T)
    print(f"Loaded {n} pairs across tasks={sorted(df['task_id'].unique().tolist())}")
    print(
        f"T range [{T.min():.4f}, {T.max():.4f}], "
        f"G range [{G.min():.4f}, {G.max():.4f}]"
    )

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    log_rmse = np.full_like(alphas, np.nan)
    rel_err_mean = np.full_like(alphas, np.nan)
    spearman = np.full_like(alphas, np.nan)
    scales = np.full_like(alphas, np.nan)

    for k, alpha in enumerate(alphas):
        T_corr = T + alpha * T * T
        if not np.all(T_corr > 0):
            continue
        s = _fit_scale_log(T_corr, G)
        scales[k] = s
        if not np.isfinite(s) or s <= 0:
            continue
        pred = s * T_corr
        log_rmse[k] = float(np.sqrt(np.mean((np.log(pred) - np.log(G)) ** 2)))
        rel_err_mean[k] = float(np.mean(np.abs(pred - G) / G))
        spearman[k] = _spearman(T_corr, G)

    valid = np.isfinite(log_rmse)
    if not np.any(valid):
        raise RuntimeError("No valid alpha found")
    best_idx = int(np.argmin(np.where(valid, log_rmse, np.inf)))
    alpha_star = float(alphas[best_idx])
    s_star = float(scales[best_idx])
    log_rmse_star = float(log_rmse[best_idx])
    rel_err_star = float(rel_err_mean[best_idx])
    spearman_star = float(spearman[best_idx])

    log_rmse_at_zero = float(log_rmse[0])
    rel_err_at_zero = float(rel_err_mean[0])

    print(
        f"\nBest alpha = {alpha_star:.5f}\n"
        f"  scale s          = {s_star:.5f}\n"
        f"  log-RMSE         = {log_rmse_star:.5f}  (alpha=0: {log_rmse_at_zero:.5f})\n"
        f"  mean abs rel err = {rel_err_star:.5f}  (alpha=0: {rel_err_at_zero:.5f})\n"
        f"  Spearman(T_corr, G) = {spearman_star:.5f}"
    )

    # ---- Plots ----
    out = Path(args.output_dir).expanduser()
    if not out.is_absolute():
        out = (REPO_ROOT / out).resolve()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(alphas, log_rmse, label="log-RMSE")
    ax.axvline(alpha_star, color="red", linestyle="--", label=f"α*={alpha_star:.4f}")
    ax.set_xlabel("alpha (overestimate coefficient)")
    ax.set_ylabel("log-RMSE residual")
    ax.set_title("Calibration sweep: log-RMSE vs alpha")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_log_rmse_vs_alpha.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(alphas, rel_err_mean, label="mean abs rel err")
    ax.axvline(alpha_star, color="red", linestyle="--", label=f"α*={alpha_star:.4f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean |s*T_corr - G| / G")
    ax.set_title("Calibration sweep: mean abs rel err vs alpha")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_relerr_vs_alpha.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.scatter(T, G, s=4, alpha=0.15, edgecolors="none", label="raw (T, G)")
    Tg = np.linspace(0, T.max(), 200)
    s0 = _fit_scale_log(Tg + 0.0 * Tg * Tg, np.full_like(Tg, np.nan))
    # For visualization fit s separately on raw T -> G:
    s_raw = _fit_scale_log(T, G)
    ax.plot(Tg, s_raw * Tg, color="orange", label=f"α=0 fit (s={s_raw:.3f})")
    ax.plot(
        Tg,
        s_star * (Tg + alpha_star * Tg * Tg),
        color="red",
        label=f"α=α* fit (α={alpha_star:.4f}, s={s_star:.3f})",
    )
    ax.set_xlabel("T (raw temporal distance)")
    ax.set_ylabel("G (graph shortest distance)")
    ax.set_title(f"T → G fit (n={n})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_TG_scatter.png", dpi=130)
    plt.close(fig)

    # Bin-wise mean & std of G/T as a function of T → diagnose underestimate.
    bins = np.quantile(T, np.linspace(0, 1, 12))
    bin_centers, mean_GT, q25_GT, q75_GT = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (T >= lo) & (T < hi)
        if mask.sum() < 20:
            continue
        ratio = G[mask] / T[mask]
        bin_centers.append(0.5 * (lo + hi))
        mean_GT.append(float(np.mean(ratio)))
        q25_GT.append(float(np.quantile(ratio, 0.25)))
        q75_GT.append(float(np.quantile(ratio, 0.75)))
    bin_centers = np.asarray(bin_centers)
    mean_GT_arr = np.asarray(mean_GT)
    q25_arr = np.asarray(q25_GT)
    q75_arr = np.asarray(q75_GT)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.fill_between(bin_centers, q25_arr, q75_arr, alpha=0.25, label="IQR")
    ax.plot(bin_centers, mean_GT_arr, marker="o", label="mean(G/T)")
    ax.axhline(s_raw, color="orange", linestyle="--", label=f"global mean s={s_raw:.3f} (α=0)")
    ax.set_xlabel("T (binned)")
    ax.set_ylabel("G / T  (effective scale)")
    ax.set_title("If G/T grows with T → temporal underestimates at long range")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration_GT_ratio_vs_T.png", dpi=130)
    plt.close(fig)

    # Save numeric sweep + summary
    sweep = pd.DataFrame(
        {
            "alpha": alphas,
            "log_rmse": log_rmse,
            "rel_err_mean": rel_err_mean,
            "spearman": spearman,
            "scale_s": scales,
        }
    )
    sweep.to_csv(out / "calibration_sweep.csv", index=False)
    summary = {
        "n_pairs": int(n),
        "alpha_star": alpha_star,
        "scale_s_star": s_star,
        "log_rmse_star": log_rmse_star,
        "rel_err_star": rel_err_star,
        "spearman_star": spearman_star,
        "log_rmse_at_alpha_zero": log_rmse_at_zero,
        "rel_err_at_alpha_zero": rel_err_at_zero,
        "T_min": float(T.min()),
        "T_max": float(T.max()),
        "G_min": float(G.min()),
        "G_max": float(G.max()),
        "current_default_alpha": 0.0015,
    }
    pd.DataFrame([summary]).to_csv(out / "calibration_summary.csv", index=False)

    md = []
    md.append("# Temporal-Distance Calibration: `temporal_dist_overestimate_coeff`")
    md.append("")
    md.append(
        "Model: `corrected = T + alpha * T**2`. We fit `alpha` against edge-level "
        "(T, G) pairs across all 5 task anchor sets (start + candidates + goal)."
    )
    md.append("")
    md.append(f"- pairs: **{n}**  (T range [{T.min():.3f}, {T.max():.3f}], G range [{G.min():.3f}, {G.max():.3f}])")
    md.append(f"- **best alpha = {alpha_star:.5f}**")
    md.append(f"- fitted scale s = {s_star:.5f}")
    md.append(f"- log-RMSE: **{log_rmse_star:.5f}** vs {log_rmse_at_zero:.5f} at alpha=0  (Δ = {(log_rmse_at_zero - log_rmse_star):+.5f})")
    md.append(f"- mean abs rel err: **{rel_err_star:.4f}** vs {rel_err_at_zero:.4f} at alpha=0  (Δ = {(rel_err_at_zero - rel_err_star):+.4f})")
    md.append(f"- Spearman(T_corr, G) at α*: {spearman_star:.4f}")
    md.append(f"- current df_planning.yaml default: 0.0015  (×{alpha_star / 0.0015 if alpha_star > 0 else 0:.1f} of suggested)")
    md.append("")
    md.append("![sweep](figures/calibration_log_rmse_vs_alpha.png)")
    md.append("")
    md.append("![relerr](figures/calibration_relerr_vs_alpha.png)")
    md.append("")
    md.append("![scatter](figures/calibration_TG_scatter.png)")
    md.append("")
    md.append("![ratio](figures/calibration_GT_ratio_vs_T.png)")
    md.append("")
    md.append("## Recommendation")
    md.append("")
    if alpha_star > 0.0015 * 1.5:
        md.append(
            f"`temporal_dist_overestimate_coeff` should be **increased** from 0.0015 to **≈ {alpha_star:.4f}**. "
            "The current default underestimates the long-range correction needed by HILP-temporal distance."
        )
    elif alpha_star < 0.0015 * 0.5:
        md.append(
            f"`temporal_dist_overestimate_coeff` should be **reduced** from 0.0015 to **≈ {alpha_star:.4f}**. "
            "Current default over-corrects relative to actual temporal-vs-graph distance growth."
        )
    else:
        md.append(
            f"Current default 0.0015 is within ±50% of the data-optimal value (≈ {alpha_star:.4f}); change is optional."
        )
    md.append("")
    md.append(
        "**Diagnostic** — the `G/T vs T` panel shows whether HILP-temporal "
        "underestimates at long range (ratio increases) or overestimates "
        "(ratio decreases). A monotonically increasing curve justifies α > 0; "
        "a flat/decreasing curve suggests α should be small or negative."
    )

    md_path = out / "calibration_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[SAVED] {md_path}")
    print(f"[SAVED] {out / 'calibration_sweep.csv'}")
    print(f"[SAVED] {out / 'calibration_summary.csv'}")


if __name__ == "__main__":
    main()
