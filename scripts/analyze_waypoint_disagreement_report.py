#!/usr/bin/env python3
"""Generate matplotlib figures + markdown report from the per-combo features
parquet produced by analyze_waypoint_disagreement_gap.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGETS = ["gap_abs", "gap_rel"]

# Group features by hypothesis for the report.
FEATURE_GROUPS: dict[str, list[str]] = {
    "A_temporal_accuracy": [
        "ratio_TG_median",
        "ratio_TG_std",
        "ratio_TG_iqr",
        "ratio_TG_max",
        "ratio_TG_min",
        "scaled_relerr_max",
        "scaled_relerr_mean",
        "scaled_relerr_std",
        "log_ratio_abs_mean",
        "log_ratio_abs_max",
        "spearman_TG",
        "pearson_TG",
        "edge_inversions",
    ],
    "B_point_spread": [
        "G_max",
        "G_min",
        "G_mean",
        "G_std",
        "G_median",
        "G_range",
        "G_compactness",
        "T_max",
        "T_min",
        "T_mean",
        "T_std",
        "T_median",
    ],
    "C_ordering_robustness": [
        "graph_second_best_gap",
        "graph_second_best_gap_rel",
        "temporal_second_best_gap",
        "temporal_second_best_gap_rel",
    ],
    "D_tour_characteristics": [
        "graph_optimal_tour_len",
        "temporal_optimal_tour_len",
        "graph_SG_direct",
        "graph_SG_shortcut_ratio",
        "graph_max_edge_in_tour",
        "graph_max_edge_in_tour_ratio",
        "graph_step_std_in_tour",
    ],
}
ALL_FEATURES: list[str] = sum(FEATURE_GROUPS.values(), [])


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    xr = pd.Series(x[mask]).rank().to_numpy()
    yr = pd.Series(y[mask]).rank().to_numpy()
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _univariate_table(
    df: pd.DataFrame, target: str, features: list[str]
) -> pd.DataFrame:
    rows = []
    y = df[target].to_numpy()
    for feat in features:
        if feat not in df.columns:
            continue
        x = df[feat].to_numpy()
        rho = _spearman_safe(x, y)
        rows.append({"feature": feat, "spearman_vs_" + target: rho, "abs_rho": abs(rho)})
    out = pd.DataFrame(rows).sort_values("abs_rho", ascending=False).drop(columns=["abs_rho"])
    return out


def _quartile_table(df: pd.DataFrame, feature: str, target: str) -> pd.DataFrame:
    sub = df[[feature, target]].dropna()
    if len(sub) < 8:
        return pd.DataFrame()
    try:
        sub["quartile"] = pd.qcut(sub[feature], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = sub.groupby("quartile", observed=True)[target].agg(["count", "mean", "median", "std"])
    return g.reset_index()


def _scatter(df: pd.DataFrame, feature: str, target: str, out_path: Path) -> None:
    sub = df[[feature, target]].dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(sub[feature], sub[target], s=4, alpha=0.15, edgecolors="none")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    rho = _spearman_safe(sub[feature].to_numpy(), sub[target].to_numpy())
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
    ax.hist(values, bins=60)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _ridge_coefs(
    df: pd.DataFrame, features: list[str], target: str
) -> dict[str, float]:
    """Standardized OLS / Ridge coefficients. Falls back to numpy if sklearn missing."""
    sub = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < len(features) + 5:
        return {}
    X = sub[features].to_numpy(dtype=np.float64)
    y = sub[target].to_numpy(dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Xs = (X - mu) / sd_safe
    if _HAS_SKLEARN:
        model = Ridge(alpha=1.0)
        model.fit(Xs, y)
        coefs = model.coef_
    else:
        # Ridge closed form: (X'X + alpha I)^-1 X' y
        alpha = 1.0
        gram = Xs.T @ Xs + alpha * np.eye(Xs.shape[1])
        coefs = np.linalg.solve(gram, Xs.T @ y)
    return {feat: float(c) for feat, c in zip(features, coefs)}


def _rf_perm_importance(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    n_estimators: int = 200,
    n_samples: int = 20000,
    random_state: int = 0,
) -> tuple[dict[str, float], float]:
    if not _HAS_SKLEARN:
        return {}, float("nan")
    sub = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 200:
        return {}, float("nan")
    if len(sub) > n_samples:
        sub = sub.sample(n=n_samples, random_state=random_state)
    X = sub[features].to_numpy()
    y = sub[target].to_numpy()
    n = len(sub)
    n_train = int(0.8 * n)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
    y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    test_r2 = float(model.score(X_test, y_test))
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=random_state, n_jobs=-1
    )
    return (
        {feat: float(v) for feat, v in zip(features, perm_result.importances_mean)},
        test_r2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="outputs/waypoint_gap_analysis/features.pkl",
        help="Path to features file (.pkl, .parquet, or .csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/waypoint_gap_analysis",
    )
    parser.add_argument(
        "--top-scatter",
        type=int,
        default=8,
        help="Number of top features (per target) to draw scatter figures for",
    )
    args = parser.parse_args()

    feat_path = Path(args.features).expanduser()
    if not feat_path.is_absolute():
        feat_path = (REPO_ROOT / feat_path).resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if feat_path.suffix == ".parquet":
        df = pd.read_parquet(feat_path)
    elif feat_path.suffix == ".csv":
        df = pd.read_csv(feat_path)
    else:
        df = pd.read_pickle(feat_path)
    n_total = len(df)
    print(f"Loaded {n_total} rows, {df.shape[1]} columns")

    available_features = [f for f in ALL_FEATURES if f in df.columns]

    # ---- Target distributions ----
    for target in TARGETS:
        _hist(
            df[target].to_numpy(),
            title=f"Distribution of {target}",
            out_path=fig_dir / f"hist_{target}.png",
            xlabel=target,
        )

    # ---- Univariate Spearman tables ----
    univariate: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        tab = _univariate_table(df, target, available_features)
        tab.to_csv(output_dir / f"univariate_{target}.csv", index=False)
        univariate[target] = tab

    # ---- Top scatter plots ----
    top_features_per_target: dict[str, list[str]] = {}
    for target in TARGETS:
        top_feats = univariate[target].head(args.top_scatter)["feature"].tolist()
        top_features_per_target[target] = top_feats
        for feat in top_feats:
            _scatter(df, feat, target, fig_dir / f"scatter_{target}_{feat}.png")

    # ---- Ridge coefficients ----
    ridge_coefs: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        coefs = _ridge_coefs(df, available_features, target)
        ridge_coefs[target] = coefs
        if coefs:
            _bar(
                {k: abs(v) for k, v in coefs.items()},
                title=f"Ridge |coef| (standardized) — {target}",
                out_path=fig_dir / f"ridge_abs_coef_{target}.png",
                xlabel="|standardized coefficient|",
            )

    # ---- RandomForest permutation importance ----
    rf_importance: dict[str, dict[str, float]] = {}
    rf_r2: dict[str, float] = {}
    for target in TARGETS:
        importance, r2 = _rf_perm_importance(df, available_features, target)
        rf_importance[target] = importance
        rf_r2[target] = r2
        if importance:
            _bar(
                importance,
                title=f"RF permutation importance — {target} (test R²={r2:.3f})",
                out_path=fig_dir / f"rf_perm_importance_{target}.png",
                xlabel="permutation importance (mean ΔR²)",
            )

    # ---- Quartile tables ----
    quartile_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for target in TARGETS:
        per_target: dict[str, pd.DataFrame] = {}
        for feat in top_features_per_target[target]:
            qt = _quartile_table(df, feat, target)
            if not qt.empty:
                per_target[feat] = qt
                qt.to_csv(
                    output_dir / f"quartile_{target}_{feat}.csv", index=False
                )
        quartile_tables[target] = per_target

    # ---- Markdown report ----
    md_lines: list[str] = []
    md_lines.append("# Waypoint Hamiltonian-Gap Analysis Report")
    md_lines.append("")
    md_lines.append(f"- Input features: `{feat_path.relative_to(REPO_ROOT)}`")
    md_lines.append(f"- Rows: **{n_total}**, columns: {df.shape[1]}")
    md_lines.append(f"- Tasks present: {sorted(df['task_id'].unique().tolist())}")
    md_lines.append("")
    md_lines.append("## Target definitions")
    md_lines.append(
        "- `gap_abs = G_len(temporal_optimal_order) − G_len(graph_optimal_order)`  (≥0)"
    )
    md_lines.append("- `gap_rel = gap_abs / G_len(graph_optimal_order)`")
    md_lines.append("")
    md_lines.append("## Target distribution")
    for target in TARGETS:
        s = df[target].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_frame().to_markdown()
        md_lines.append(f"### `{target}`")
        md_lines.append("")
        md_lines.append(s)
        md_lines.append("")
        md_lines.append(f"![hist_{target}](figures/hist_{target}.png)")
        md_lines.append("")

    md_lines.append("## Univariate Spearman ρ (sorted by |ρ|)")
    for target in TARGETS:
        md_lines.append(f"### vs `{target}`")
        md_lines.append("")
        md_lines.append(univariate[target].head(30).to_markdown(index=False))
        md_lines.append("")

    md_lines.append("## Top scatter plots")
    for target in TARGETS:
        md_lines.append(f"### `{target}` — top {args.top_scatter} features")
        md_lines.append("")
        for feat in top_features_per_target[target]:
            md_lines.append(f"#### {feat}")
            md_lines.append(f"![scatter_{target}_{feat}](figures/scatter_{target}_{feat}.png)")
            md_lines.append("")
            qt = quartile_tables[target].get(feat)
            if qt is not None and not qt.empty:
                md_lines.append(qt.to_markdown(index=False))
                md_lines.append("")

    md_lines.append("## Multivariate models")
    for target in TARGETS:
        md_lines.append(f"### `{target}`")
        md_lines.append("")
        md_lines.append(f"- RandomForest test R²: **{rf_r2.get(target, float('nan')):.4f}**")
        md_lines.append("")
        md_lines.append(f"![ridge_abs_coef_{target}](figures/ridge_abs_coef_{target}.png)")
        md_lines.append("")
        md_lines.append(f"![rf_perm_importance_{target}](figures/rf_perm_importance_{target}.png)")
        md_lines.append("")

    md_lines.append("## Hypothesis-grouped Spearman summary")
    for target in TARGETS:
        md_lines.append(f"### vs `{target}`")
        md_lines.append("")
        rows = []
        uni = univariate[target].set_index("feature")
        for group, feats in FEATURE_GROUPS.items():
            for feat in feats:
                if feat in uni.index:
                    rho = uni.loc[feat, "spearman_vs_" + target]
                    rows.append({"group": group, "feature": feat, "rho": rho})
        gtab = pd.DataFrame(rows)
        if not gtab.empty:
            summary = (
                gtab.assign(abs_rho=lambda d: d["rho"].abs())
                .groupby("group")
                .agg(max_abs_rho=("abs_rho", "max"), mean_abs_rho=("abs_rho", "mean"))
                .reset_index()
                .sort_values("max_abs_rho", ascending=False)
            )
            md_lines.append(summary.to_markdown(index=False))
            md_lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[SAVED] {report_path}")
    print(f"[SAVED] figures dir: {fig_dir}")


if __name__ == "__main__":
    main()
