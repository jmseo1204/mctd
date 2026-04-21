#!/usr/bin/env python3
"""Analysis #2 (report): for the full combo dataset, study which features
predict mismatch (binary) using AUC + logistic regression + quartile rates.
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
    # These are computed from the chosen orderings — leak the label.
    "graph_under_temporal_order_len",
    "gap_abs",
    "gap_rel",
    # The second-best gap is independent of which ordering temporal picks,
    # but it is a structural property of G; we keep it as a feature.
}

ID_COLS = {
    "task_id",
    "combo_idx_in_task",
    "w1_i",
    "w1_j",
    "w2_i",
    "w2_j",
    "w3_i",
    "w3_j",
    "mismatch",
}


def _auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U based AUC. Returns nan if degenerate."""
    mask = np.isfinite(scores) & np.isfinite(labels)
    s = scores[mask]
    y = labels[mask].astype(np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ranks = np.empty(len(s_sorted), dtype=np.float64)
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[i:j] = avg_rank
        i = j
    n_pos = int(y_sorted.sum())
    n_neg = int(n - n_pos)
    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _univariate_auc(df: pd.DataFrame, label: str, features: list[str]) -> pd.DataFrame:
    rows = []
    y = df[label].to_numpy(dtype=np.int64)
    for feat in features:
        x = df[feat].to_numpy(dtype=np.float64)
        auc = _auc_binary(x, y)
        rows.append(
            {"feature": feat, "auc": auc, "auc_minus_0_5": (auc - 0.5)}
        )
    out = pd.DataFrame(rows)
    out["abs_signed"] = out["auc_minus_0_5"].abs()
    return (
        out.sort_values("abs_signed", ascending=False)
        .drop(columns=["abs_signed"])
        .reset_index(drop=True)
    )


def _quartile_mismatch_rate(
    df: pd.DataFrame, feature: str, label: str
) -> pd.DataFrame:
    sub = df[[feature, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8:
        return pd.DataFrame()
    try:
        sub = sub.copy()
        sub["quartile"] = pd.qcut(
            sub[feature],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()
    g = sub.groupby("quartile", observed=True)[label].agg(
        ["count", "mean", "sum"]
    )
    g = g.rename(columns={"mean": "mismatch_rate", "sum": "n_mismatch"})
    return g.reset_index()


def _logistic_l2(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0, n_iter: int = 200, lr: float = 0.5
) -> np.ndarray:
    """Standardized-features L2 logistic regression with batch gradient descent.
    Returns coefficient vector aligned with X columns (intercept dropped).
    """
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = float(np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6))))
    y = y.astype(np.float64)
    for _ in range(n_iter):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = X.T @ (p - y) / n + alpha * w / n
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w


def _scatter_box(
    df: pd.DataFrame, feature: str, label: str, out_path: Path
) -> None:
    sub = df[[feature, label]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    sub_neg = sub.loc[sub[label] == 0, feature]
    sub_pos = sub.loc[sub[label] == 1, feature]
    ax.violinplot(
        [sub_neg, sub_pos],
        positions=[0, 1],
        showmedians=True,
        widths=0.7,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non-mismatch", "mismatch"])
    ax.set_ylabel(feature)
    auc = _auc_binary(sub[feature].to_numpy(), sub[label].to_numpy())
    ax.set_title(f"{feature} | AUC={auc:.3f} (n_neg={len(sub_neg)}, n_pos={len(sub_pos)})")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", default="outputs/waypoint_gap_analysis_full/features_full.pkl"
    )
    parser.add_argument(
        "--output-dir", default="outputs/waypoint_gap_analysis_full"
    )
    parser.add_argument("--top-scatter", type=int, default=10)
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
    n = len(df)
    n_pos = int(df["mismatch"].sum())
    print(
        f"Loaded {n} rows; mismatch_rate={n_pos / n:.4f}, n_pos={n_pos}, n_neg={n - n_pos}"
    )

    features = [
        c
        for c in df.columns
        if c not in EXCLUDED_FEATURES
        and c not in ID_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Feature columns: {len(features)}")

    # ---- Univariate AUC ----
    auc_table = _univariate_auc(df, "mismatch", features)
    auc_table.to_csv(output_dir / "univariate_auc.csv", index=False)

    # ---- Top features: violin + quartile mismatch-rate ----
    top_features = auc_table.head(args.top_scatter)["feature"].tolist()
    quartile_tabs: dict[str, pd.DataFrame] = {}
    for feat in top_features:
        _scatter_box(df, feat, "mismatch", fig_dir / f"violin_{feat}.png")
        qt = _quartile_mismatch_rate(df, feat, "mismatch")
        if not qt.empty:
            quartile_tabs[feat] = qt
            qt.to_csv(output_dir / f"quartile_mismatch_{feat}.csv", index=False)

    # ---- Logistic regression with standardized features ----
    sub = df[features + ["mismatch"]].replace([np.inf, -np.inf], np.nan).dropna()
    Xraw = sub[features].to_numpy(dtype=np.float64)
    y = sub["mismatch"].to_numpy(dtype=np.int64)
    mu = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Xstd = (Xraw - mu) / sd_safe
    coefs = _logistic_l2(Xstd, y, alpha=1.0, n_iter=400, lr=0.5)
    coef_dict = {feat: float(c) for feat, c in zip(features, coefs)}
    _bar(
        {k: abs(v) for k, v in coef_dict.items()},
        title="Logistic L2 |coef| (standardized) — mismatch",
        out_path=fig_dir / "logistic_abs_coef.png",
        xlabel="|standardized coefficient|",
    )

    # ---- Markdown report ----
    md: list[str] = []
    md.append("# Full-combos Mismatch Classification Report")
    md.append("")
    md.append(f"- Source features: `{feat_path.relative_to(REPO_ROOT)}`")
    md.append(
        f"- Total combos: **{n}**, mismatch: **{n_pos}** ({n_pos / n * 100:.2f}%), "
        f"non-mismatch: **{n - n_pos}**"
    )
    md.append("")
    md.append("## Univariate AUC (sorted by |AUC − 0.5|)")
    md.append("")
    md.append(auc_table.head(25).to_markdown(index=False))
    md.append("")

    md.append(f"## Violin + quartile rates — top {args.top_scatter} features")
    for feat in top_features:
        md.append(f"### {feat}")
        md.append(f"![violin_{feat}](figures/violin_{feat}.png)")
        md.append("")
        if feat in quartile_tabs:
            md.append(quartile_tabs[feat].to_markdown(index=False))
            md.append("")

    md.append("## L2 Logistic regression (standardized)")
    md.append("")
    md.append(f"![logistic_abs_coef](figures/logistic_abs_coef.png)")
    md.append("")
    sorted_coefs = sorted(
        coef_dict.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:20]
    md.append(
        pd.DataFrame(sorted_coefs, columns=["feature", "logistic_coef"]).to_markdown(
            index=False
        )
    )
    md.append("")

    out = output_dir / "report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[SAVED] {out}")
    print(f"[SAVED] figures dir: {fig_dir}")


if __name__ == "__main__":
    main()
