#!/usr/bin/env python3
"""
Guidance Analysis Script (JSONL-based)
Reads guidance_*.jsonl files and reports per-tree (forward/backward) stats.
"""
import sys
import os
import json
import collections
import statistics
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _mean(vals: list) -> float:
    return statistics.mean(vals) if vals else 0.0


def _extract_guidance_records(records: List[dict]) -> List[dict]:
    """Extract guidance data from SLP-format records (tag=guidance.combined → data={...})
    or from legacy flat-format records (direct keys)."""
    out = []
    for r in records:
        if r.get("tag") == "guidance.combined":
            # SLP format: fields are nested under "data"
            d = r.get("data", {})
            out.append(d)
        elif "tree_tag" in r and "eff_goal_scale" in r:
            # Legacy flat format (old guidance_*.jsonl)
            out.append(r)
    return out


def analyze(records: List[dict], log_name: str) -> None:
    guidance_records = _extract_guidance_records(records)

    if not guidance_records:
        print(f"\nNo guidance records found in {log_name}.")
        print("  (looking for tag='guidance.combined' or legacy flat format)")
        return

    # Group by (tree_tag, eff_goal_scale)
    # tree_tag: "bidir_mcts_from_start" (forward) or "bidir_mcts_from_goal" (backward)
    by_tree: Dict[str, Dict[float, List[dict]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in guidance_records:
        tree = r.get("tree_tag", "unknown")
        scale = round(float(r.get("eff_goal_scale", 0.0)), 4)
        by_tree[tree][scale].append(r)
    records = guidance_records  # use filtered records for rest of analysis

    print("\n" + "=" * 140)
    print(f"  Guidance Analysis: {log_name}")
    print("=" * 140)

    tree_order = sorted(by_tree.keys())
    for tree_tag in tree_order:
        # Human-readable label
        if "start" in tree_tag:
            label = "FORWARD  (bidir_mcts_from_start)"
        elif "goal" in tree_tag:
            label = "BACKWARD (bidir_mcts_from_goal)"
        else:
            label = tree_tag

        scale_data = by_tree[tree_tag]
        total_n = sum(len(v) for v in scale_data.values())

        print(f"\n  Tree: {label}  [{total_n} guidance calls]")
        print(
            f"  {'Scale':<10} {'DistBatch(mean)':<18} {'FinalTokDist(mean)':<20} "
            f"{'N':<7} {'GoalInner':<12} {'AnchorLoss':<13} {'GoalLoss':<12} "
            f"{'G/A Ratio':<12} {'RDFLoss':<10}"
        )
        print("  " + "-" * 118)

        for scale in sorted(scale_data.keys()):
            recs = scale_data[scale]
            n = len(recs)

            # dist_per_batch and final_token_dist are lists-per-record
            dpb_vals = [_mean(r["dist_per_batch"]) for r in recs if r.get("dist_per_batch")]
            ftd_vals = [_mean(r["final_token_dist"]) for r in recs if r.get("final_token_dist")]

            avg_dpb = _mean(dpb_vals)
            avg_ftd = _mean(ftd_vals)
            avg_goal_inner = _mean([r.get("goal_inner", 0.0) for r in recs])
            avg_anchor = _mean([r.get("anchor_loss", 0.0) for r in recs])
            avg_goal_loss = _mean([r.get("goal_loss", 0.0) for r in recs])
            avg_ga_ratio = _mean([r.get("goal_anchor_ratio", 0.0) for r in recs])
            avg_rdf = _mean([r.get("rdf_loss", 0.0) for r in recs])

            print(
                f"  {scale:<10.4f} {avg_dpb:<18.6f} {avg_ftd:<20.6f} "
                f"{n:<7} {avg_goal_inner:<12.5f} {avg_anchor:<13.5f} {avg_goal_loss:<12.5f} "
                f"{avg_ga_ratio:<12.4f} {avg_rdf:<10.5f}"
            )

    # --- Grad norm & clip summary (SLP format: tag=diffusion.grad_norm / diffusion.clip_warning) ---
    grad_norms = [r["data"]["grad_norm"] for r in records
                  if r.get("tag") == "diffusion.grad_norm" and isinstance(r.get("data"), dict)]
    clip_count = sum(1 for r in records if r.get("tag") == "diffusion.clip_warning")
    if grad_norms:
        print(f"\n  Grad Norm  — mean: {_mean(grad_norms):.5f}  "
              f"min: {min(grad_norms):.5f}  max: {max(grad_norms):.5f}  "
              f"n={len(grad_norms)}  clips={clip_count}")

    print("\n" + "=" * 140)
    print(f"  Total guidance records: {sum(len(v) for d in by_tree.values() for v in d.values())}")
    print("=" * 140 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/guidance_analysis.py <guidance_*.jsonl>")
        return

    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f"Error: file not found: {log_path}")
        return

    records = load_jsonl(log_path)
    analyze(records, os.path.basename(log_path))


if __name__ == "__main__":
    main()
