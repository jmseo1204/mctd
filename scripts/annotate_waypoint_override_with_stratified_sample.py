#!/usr/bin/env python3
"""Stratified-sample waypoint groups from a task override YAML and place them
at the front of each task's waypoint_ij_groups list.

Reads waypoint_group_graph_second_best_gap_rel per task (written by
find_waypoint_metric_disagreements.py). Divides each task's mismatch groups
into 3 difficulty bins by graph_second_best_gap_rel quantile:
  high  – top    25 % (graph_second_best_gap_rel >= 75th-pct)
  mid   – 25–50 %     (50th–75th-pct)
  low   – 50–75 %     (25th–50th-pct)
  (bottom 25 % is excluded from evaluation)

Samples --n-per-bin groups from each bin using a fixed seed.

After re-ordering, writes back to the same YAML with:
  • eval_sampling  (top-level metadata: seed, n_per_bin, bin index ranges)
  • tasks.<id>.eval_sample_metadata  (per selected group: rank, bin, gap_rel value)

The first n_per_bin*3 entries in each task's waypoint_ij_groups are the
stratified sample (indices 0..n-1 = high, n..2n-1 = mid, 2n..3n-1 = low).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── YAML helpers ─────────────────────────────────────────────────────────────

class _FlowStyleList(list):
    """Sequence rendered on a single line (flow style)."""


class _WaypointDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def _represent_flow_seq(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_WaypointDumper.add_representer(_FlowStyleList, _represent_flow_seq)


def _wrap_groups(groups: list) -> list:
    """Re-wrap plain lists back to flow-style lists for compact YAML output."""
    out = []
    for group in groups:
        point_list = [_FlowStyleList([int(pt[0]), int(pt[1])]) for pt in group]
        out.append(_FlowStyleList(point_list))
    return out


# ── Core logic ────────────────────────────────────────────────────────────────

def _stratified_sample(
    n_groups: int,
    gap_rel_values: list[float],
    n_per_bin: int,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    """Return {bin_name: [original_indices]} for the stratified sample.

    Bins are based on percentile thresholds of gap_rel_values (descending sort
    assumed but percentiles computed from the actual values for robustness).
    """
    vals = np.asarray(gap_rel_values, dtype=np.float64)
    q75 = float(np.percentile(vals, 75))
    q50 = float(np.percentile(vals, 50))
    q25 = float(np.percentile(vals, 25))

    bins: dict[str, list[int]] = {"high": [], "mid": [], "low": []}
    for idx, v in enumerate(vals):
        if v >= q75:
            bins["high"].append(idx)
        elif v >= q50:
            bins["mid"].append(idx)
        elif v >= q25:
            bins["low"].append(idx)
        # below q25: excluded

    sampled: dict[str, list[int]] = {}
    for bin_name, indices in bins.items():
        if len(indices) < n_per_bin:
            raise ValueError(
                f"Bin '{bin_name}' has only {len(indices)} groups "
                f"but n_per_bin={n_per_bin}. Reduce --n-per-bin."
            )
        chosen = rng.choice(len(indices), size=n_per_bin, replace=False)
        chosen.sort()
        sampled[bin_name] = [indices[i] for i in chosen]

    return sampled, {"q75": q75, "q50": q50, "q25": q25}


def annotate(yaml_path: Path, n_per_bin: int, seed: int) -> None:
    with open(yaml_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    tasks = payload.get("tasks", {})
    if not tasks:
        raise ValueError("No 'tasks' key in YAML")

    total_eval = n_per_bin * 3
    rng = np.random.default_rng(seed)

    for task_id_raw, task_data in tasks.items():
        task_id = int(task_id_raw)
        groups = task_data.get("waypoint_ij_groups", [])
        gap_rels = task_data.get("waypoint_group_graph_second_best_gap_rel")

        if not gap_rels:
            raise ValueError(
                f"Task {task_id} is missing 'waypoint_group_graph_second_best_gap_rel'. "
                "Re-generate the YAML with the updated find_waypoint_metric_disagreements.py."
            )
        if len(gap_rels) != len(groups):
            raise ValueError(
                f"Task {task_id}: len(gap_rels)={len(gap_rels)} != len(groups)={len(groups)}"
            )

        n_groups = len(groups)
        sampled_indices, thresholds = _stratified_sample(n_groups, gap_rels, n_per_bin, rng)

        # Build re-ordered group list: stratified sample first, then remaining
        selected_flat = (
            sampled_indices["high"] + sampled_indices["mid"] + sampled_indices["low"]
        )
        selected_set = set(selected_flat)
        remaining = [i for i in range(n_groups) if i not in selected_set]

        new_order = selected_flat + remaining
        new_groups = [groups[i] for i in new_order]
        new_gap_rels = [gap_rels[i] for i in new_order]

        # Build eval_sample_metadata for the first total_eval entries
        eval_sample_metadata = []
        bin_order = (
            [("high", idx) for idx in sampled_indices["high"]]
            + [("mid",  idx) for idx in sampled_indices["mid"]]
            + [("low",  idx) for idx in sampled_indices["low"]]
        )
        for local_rank, (bin_name, orig_idx) in enumerate(bin_order):
            eval_sample_metadata.append({
                "local_rank": int(local_rank),
                "bin": bin_name,
                "graph_second_best_gap_rel": round(float(gap_rels[orig_idx]), 6),
            })

        # Update task payload
        task_data["waypoint_ij_groups"] = _wrap_groups(new_groups)
        task_data["waypoint_group_graph_second_best_gap_rel"] = _FlowStyleList(
            [round(float(v), 6) for v in new_gap_rels]
        )
        task_data["eval_sample_metadata"] = eval_sample_metadata

        print(
            f"[Task {task_id}] n_groups={n_groups}  "
            f"thresholds: q75={thresholds['q75']:.4f} q50={thresholds['q50']:.4f} "
            f"q25={thresholds['q25']:.4f}  "
            f"bin sizes: high={len(sampled_indices['high'])} "
            f"mid={len(sampled_indices['mid'])} "
            f"low={len(sampled_indices['low'])} "
            f"(sampled {n_per_bin} each)",
            flush=True,
        )

    # Write top-level eval_sampling metadata
    n_per_bin_val = n_per_bin
    payload["eval_sampling"] = {
        "seed": int(seed),
        "n_per_bin": int(n_per_bin_val),
        "strategy": "stratified_by_graph_second_best_gap_rel",
        "total_eval_groups_per_task": int(total_eval),
        "bins": [
            {"name": "high", "quantile_range": [0.75, 1.0],
             "group_index_range": [0, n_per_bin_val]},
            {"name": "mid",  "quantile_range": [0.50, 0.75],
             "group_index_range": [n_per_bin_val, n_per_bin_val * 2]},
            {"name": "low",  "quantile_range": [0.25, 0.50],
             "group_index_range": [n_per_bin_val * 2, total_eval]},
        ],
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            payload,
            f,
            Dumper=_WaypointDumper,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )
    print(f"[SAVED] {yaml_path}  (eval_groups_per_task={total_eval})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified-sample waypoint groups by graph_second_best_gap_rel."
    )
    parser.add_argument(
        "--waypoint-override",
        required=True,
        help="Path to the task override YAML (antmaze_giant_waypoints.yaml).",
    )
    parser.add_argument(
        "--n-per-bin",
        type=int,
        default=7,
        help="Number of groups to sample per difficulty bin (default: 7).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    args = parser.parse_args()

    yaml_path = Path(args.waypoint_override).expanduser()
    if not yaml_path.is_absolute():
        yaml_path = (REPO_ROOT / yaml_path).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    annotate(yaml_path, n_per_bin=args.n_per_bin, seed=args.seed)


if __name__ == "__main__":
    main()
