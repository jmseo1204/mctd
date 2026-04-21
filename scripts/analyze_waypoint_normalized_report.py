#!/usr/bin/env python3
"""Analysis #1: normalize the gap by ``graph_second_best_gap`` to remove the
definitional dominance of "how strongly the graph metric prefers its best
ordering" and re-rank features.

New targets derived from the existing features.csv:
  - gap_over_2nd_best  = gap_abs / graph_second_best_gap
        Interpretation: 0 means temporal picked the best ordering (no gap),
        1 means temporal picked exactly the 2nd-best ordering, >1 means even
        worse. Only well-defined when graph_second_best_gap > eps.

We re-run univariate Spearman + Ridge + quartile analysis on this normalized
target, and on the original ``gap_rel`` for side-by-side comparison after
explicitly excluding the second-best-gap features (so we see what drives gap
*beyond* the trivial denominator effect).
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

EXCLUDED_FEATURES = {
    # In the denominator of gap_over_2nd_best, so trivially correlated.
    "graph_second_best_gap",
    "graph_second_best_gap_rel",
    # Targets / derived from targets:
    "gap_abs",
    "gap_rel",
    "gap_over_2nd_best",
    "graph_under_temporal_order_len",
}


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    xr = pd.Series(x[mask]).rank().to_numpy()
    yr = pd.Series(y[mask]).rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _univariate(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    y = df[target].to_numpy()
    for feat in features:
        x = df[feat].to_numpy()
        rho = _spearman_safe(x, y)
        rows.append({"feature": feat, "spearman": rho, "abs_rho": abs(rho)})
    return (
        pd.DataFrame(rows)
        .sort_values("abs_rho", ascending=False)
        .drop(columns=["abs_rho"])
        .reset_index(drop=True)
    )


def _ridge_standardized(
    df: pd.DataFrame, features: list[str], target: str, alpha: float = 1.0
) -> dict[str, float]:
    sub = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(features) + 5:
        return {}
    X = sub[features].to_numpy(dtype=np.float64)
    y = sub[target].to_numpy(dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd_safe
    gram = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
    coefs = np.asarray(np.linalg.solve(gram, Xs.T @ y)).reshape(-1)
    if coefs.shape[0] != len(features):
        return {}
    return {feat: float(c) for feat, c in zip(features, coefs.tolist())}


def _quartile_table(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    sub = df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8:
        return pd.DataFrame()
    try:
        sub = sub.copy()
        sub["quartile"] = pd.qcut(sub[feature], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = sub.groupby("quartile", observed=True)[target].agg(["count", "mean", "median", "std"])
    return g.reset_index()


def _scatter(df: pd.DataFrame, feature: str, target: str, out_path: Path) -> None:
    sub = df[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(sub[feature], sub[target], s=4, alpha=0.15, edgecolors="none")
    rho = _spearman_safe(sub[feature].to_numpy(), sub[target].to_numpy())
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.set_title(f"{feature} vs {target} (Spearman ρ={rho:.3f}, n={len(sub)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _bar(values: dict[str, float], title: str, out_path: Path, xlabel: str) -> None:
    if not values:
        return
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.28 * len(names))))
    ax.barh(range(len(names)), vals)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _hist(values: np.ndarray, title: str, out_path: Path, xlabel: str) -> None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(values, bins=80)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", default="outputs/waypoint_gap_analysis/features.pkl"
    )
    parser.add_argument(
        "--output-dir", default="outputs/waypoint_gap_analysis_normalized"
    )
    parser.add_argument("--top-scatter", type=int, default=10)
    parser.add_argument("--secondbest-eps", type=float, default=1e-3)
    args = parser.parse_args()

    feat_path = Path(args.features).expanduser()
    if not feat_path.is_absolute():
        feat_path = (REPO_ROOT / feat_path).resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if feat_path.suffix == ".csv":
        df = pd.read_csv(feat_path)
    else:
        df = pd.read_pickle(feat_path)
    n_total = len(df)
    print(f"Loaded {n_total} rows, {df.shape[1]} columns")

    # Compute new normalized target.
    second_best = df["graph_second_best_gap"].to_numpy(dtype=np.float64)
    eligible_mask = second_best > args.secondbest_eps
    df_norm = df.loc[eligible_mask].copy()
    df_norm["gap_over_2nd_best"] = df_norm["gap_abs"] / df_norm["graph_second_best_gap"]
    n_eligible = len(df_norm)
    print(
        f"After filter (second_best_gap > {args.secondbest_eps}): {n_eligible} rows "
        f"({100.0 * n_eligible / n_total:.1f}%)"
    )

    targets = ["gap_over_2nd_best", "gap_rel"]
    features = [
        c
        for c in df_norm.columns
        if c not in EXCLUDED_FEATURES
        and pd.api.types.is_numeric_dtype(df_norm[c])
        and not c.startswith(("w1_", "w2_", "w3_", "task_id", "combo_idx"))
    ]
    # Manually drop integer location features:
    features = [
        c
        for c in features
        if c not in ("task_id", "combo_idx_in_task", "w1_i", "w1_j", "w2_i", "w2_j", "w3_i", "w3_j")
    ]
    print(f"Feature columns considered: {len(features)}")

    # ---- Target distributions ----
    for tgt in targets:
        _hist(
            df_norm[tgt].to_numpy(dtype=np.float64),
            title=f"Distribution of {tgt}",
            out_path=fig_dir / f"hist_{tgt}.png",
            xlabel=tgt,
        )

    # ---- Univariate ----
    univariate: dict[str, pd.DataFrame] = {}
    for tgt in targets:
        tab = _univariate(df_norm, tgt, features)
        tab.to_csv(output_dir / f"univariate_{tgt}.csv", index=False)
        univariate[tgt] = tab

    # ---- Top scatter plots ----
    top_features: dict[str, list[str]] = {}
    for tgt in targets:
        top = univariate[tgt].head(args.top_scatter)["feature"].tolist()
        top_features[tgt] = top
        for feat in top:
            _scatter(df_norm, feat, tgt, fig_dir / f"scatter_{tgt}_{feat}.png")

    # ---- Ridge ----
    ridge_coefs: dict[str, dict[str, float]] = {}
    for tgt in targets:
        coefs = _ridge_standardized(df_norm, features, tgt)
        ridge_coefs[tgt] = coefs
        if coefs:
            _bar(
                {k: abs(v) for k, v in coefs.items()},
                title=f"Ridge |coef| (standardized) — {tgt}",
                out_path=fig_dir / f"ridge_abs_coef_{tgt}.png",
                xlabel="|standardized coefficient|",
            )

    # ---- Quartile tables for top features ----
    quartile_tabs: dict[str, dict[str, pd.DataFrame]] = {}
    for tgt in targets:
        per_target: dict[str, pd.DataFrame] = {}
        for feat in top_features[tgt]:
            qt = _quartile_table(df_norm, feat, tgt)
            if not qt.empty:
                per_target[feat] = qt
                qt.to_csv(output_dir / f"quartile_{tgt}_{feat}.csv", index=False)
        quartile_tabs[tgt] = per_target

    # ---- Markdown report ----
    md: list[str] = []
    md.append("# Waypoint Gap Analysis — Normalized by graph_second_best_gap")
    md.append("")
    md.append(f"- Source features: `{feat_path.relative_to(REPO_ROOT)}`")
    md.append(f"- Rows used: **{n_eligible}** of {n_total} (filtered to graph_second_best_gap > {args.secondbest_eps})")
    md.append("")
    md.append("## Why this analysis")
    md.append(
        "In the original analysis `graph_second_best_gap` dominated (ρ≈0.90 for both gap_abs and gap_rel) — "
        "but it is essentially the *upper envelope* on how badly any wrong ordering can hurt under the graph metric. "
        "Here we divide it out so the new target measures: **conditional on a non-trivial decision, how poorly did "
        "temporal pick on the rank-position scale?**"
    )
    md.append("")
    md.append("- `gap_over_2nd_best = gap_abs / graph_second_best_gap`")
    md.append("  - 0 = temporal picked the optimal ordering")
    md.append("  - 1 = temporal picked exactly the 2nd-best ordering (a 'minimal' miss)")
    md.append("  - >1 = temporal picked something worse than the 2nd-best (3rd, 4th, … among the 6 possible)")
    md.append("")
    md.append("`graph_second_best_gap*` features are excluded from the predictor set.")
    md.append("")
    md.append("## Target distributions")
    for tgt in targets:
        md.append(f"### `{tgt}`")
        md.append("")
        s = (
            df_norm[tgt]
            .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
            .to_frame()
            .to_markdown()
        )
        md.append(s)
        md.append("")
        md.append(f"![hist_{tgt}](figures/hist_{tgt}.png)")
        md.append("")

    md.append("## Univariate Spearman ρ (top 25)")
    for tgt in targets:
        md.append(f"### vs `{tgt}`")
        md.append("")
        md.append(univariate[tgt].head(25).to_markdown(index=False))
        md.append("")

    md.append("## Top scatter plots")
    for tgt in targets:
        md.append(f"### `{tgt}` — top {args.top_scatter} features")
        md.append("")
        for feat in top_features[tgt]:
            md.append(f"#### {feat}")
            md.append(f"![scatter_{tgt}_{feat}](figures/scatter_{tgt}_{feat}.png)")
            md.append("")
            qt = quartile_tabs[tgt].get(feat)
            if qt is not None and not qt.empty:
                md.append(qt.to_markdown(index=False))
                md.append("")

    md.append("## Ridge multivariate (standardized)")
    for tgt in targets:
        md.append(f"### `{tgt}`")
        md.append("")
        md.append(f"![ridge_abs_coef_{tgt}](figures/ridge_abs_coef_{tgt}.png)")
        md.append("")
        if ridge_coefs[tgt]:
            top_signed = sorted(
                ridge_coefs[tgt].items(), key=lambda kv: abs(kv[1]), reverse=True
            )[:15]
            md.append(
                pd.DataFrame(top_signed, columns=["feature", "ridge_coef"])
                .to_markdown(index=False)
            )
            md.append("")

    out = output_dir / "report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[SAVED] {out}")
    print(f"[SAVED] figures dir: {fig_dir}")


if __name__ == "__main__":
    main()
