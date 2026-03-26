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


def _analyze_grad_comparison(records: List[dict]) -> None:
    """Analyze diffusion.grad_comparison records: prior norm vs guidance grad norm."""
    comp_records = [r["data"] for r in records
                    if r.get("tag") == "diffusion.grad_comparison" and isinstance(r.get("data"), dict)]
    if not comp_records:
        return

    prior_norms = [r["prior_norm"] for r in comp_records if "prior_norm" in r]
    total_norms = [r["guidance_total_norm"] for r in comp_records if "guidance_total_norm" in r]
    ratios = [r["ratio"] for r in comp_records if "ratio" in r]

    # Collect per-guidance keys across all records
    all_keys = set()
    for r in comp_records:
        all_keys.update(r.get("per_guidance_norms", {}).keys())
    per_key = {k: [r["per_guidance_norms"][k] for r in comp_records if k in r.get("per_guidance_norms", {})]
               for k in sorted(all_keys)}

    n = len(comp_records)
    print("\n" + "=" * 100)
    print(f"  Guidance Grad vs Diffusion Prior  [{n} denoising steps with guidance]")
    print("=" * 100)
    print(f"  {'Metric':<30} {'Mean':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-" * 70)
    if prior_norms:
        print(f"  {'Prior (pred_noise) norm':<30} {_mean(prior_norms):>12.6f} {min(prior_norms):>12.6f} {max(prior_norms):>12.6f}")
    if total_norms:
        print(f"  {'Guidance total grad norm':<30} {_mean(total_norms):>12.6f} {min(total_norms):>12.6f} {max(total_norms):>12.6f}")
    if ratios:
        print(f"  {'Ratio (guidance/prior)':<30} {_mean(ratios):>12.4f} {min(ratios):>12.4f} {max(ratios):>12.4f}")
    for k, vals in per_key.items():
        if vals:
            print(f"  {'  └─ ' + k + ' grad norm':<30} {_mean(vals):>12.6f} {min(vals):>12.6f} {max(vals):>12.6f}")
    print("=" * 100)


def _analyze_frozen_diag(records: List[dict]) -> None:
    """Diagnose frozen-trajectory issue: clip scale, x_start positions, unit vec alignment."""
    clip_recs = [r["data"] for r in records if r.get("tag") == "diffusion.clip_scale" and isinstance(r.get("data"), dict)]
    xstart_recs = [r["data"] for r in records if r.get("tag") == "diffusion.xstart_pos" and isinstance(r.get("data"), dict)]
    uv_recs = [r["data"] for r in records if r.get("tag") == "guidance.unit_vec_diag" and isinstance(r.get("data"), dict)]

    if not any([clip_recs, xstart_recs, uv_recs]):
        return

    print("\n" + "=" * 80)
    print("  Frozen-Trajectory Diagnostics")
    print("=" * 80)

    if clip_recs:
        scale_means = [r["scale_mean"] for r in clip_recs if "scale_mean" in r]
        scale_mins  = [r["scale_min"]  for r in clip_recs if "scale_min"  in r]
        pnorms      = [r["prior_norm_mean"] for r in clip_recs if "prior_norm_mean" in r]
        gnorms      = [r["grad_norm_mean"]  for r in clip_recs if "grad_norm_mean"  in r]
        print(f"\n  [Soft Clip Scale]  n={len(clip_recs)}")
        print(f"    scale_mean — mean:{_mean(scale_means):.4f}  min:{min(scale_means):.4f}  max:{max(scale_means):.4f}")
        print(f"    scale_min  — mean:{_mean(scale_mins):.4f}   min:{min(scale_mins):.4f}")
        print(f"    prior_norm — mean:{_mean(pnorms):.6f}")
        print(f"    grad_norm  — mean:{_mean(gnorms):.6f}")
        if _mean(scale_means) < 0.1:
            print("    *** WARNING: scale_mean < 0.1 — guidance nearly zeroed by clipping! ***")

    if xstart_recs:
        guided_x  = [r["xstart_mean_xy"][0] for r in xstart_recs if "xstart_mean_xy" in r]
        guided_y  = [r["xstart_mean_xy"][1] for r in xstart_recs if "xstart_mean_xy" in r]
        prior_x   = [r["prior_xstart_mean_xy"][0] for r in xstart_recs if "prior_xstart_mean_xy" in r]
        prior_y   = [r["prior_xstart_mean_xy"][1] for r in xstart_recs if "prior_xstart_mean_xy" in r]
        std_x     = [r["xstart_std_xy"][0] for r in xstart_recs if "xstart_std_xy" in r]
        std_y     = [r["xstart_std_xy"][1] for r in xstart_recs if "xstart_std_xy" in r]
        print(f"\n  [x_start Positions (normalized)]  n={len(xstart_recs)}")
        print(f"    guided  x: mean={_mean(guided_x):.4f}  y: mean={_mean(guided_y):.4f}")
        print(f"    prior   x: mean={_mean(prior_x):.4f}  y: mean={_mean(prior_y):.4f}")
        print(f"    std     x: mean={_mean(std_x):.4f}  y: mean={_mean(std_y):.4f}")
        if _mean(std_x) < 0.05 and _mean(std_y) < 0.05:
            print("    *** WARNING: x_start std near 0 — model predicting same position always ***")

        # Noise-level stratification (only if new fields present)
        hi_std_x = [r["xstart_std_high_noise"][0] for r in xstart_recs if "xstart_std_high_noise" in r]
        hi_std_y = [r["xstart_std_high_noise"][1] for r in xstart_recs if "xstart_std_high_noise" in r]
        lo_std_x = [r["xstart_std_low_noise"][0]  for r in xstart_recs if "xstart_std_low_noise"  in r]
        lo_std_y = [r["xstart_std_low_noise"][1]  for r in xstart_recs if "xstart_std_low_noise"  in r]
        nl_means = [r["noise_level_mean"] for r in xstart_recs if "noise_level_mean" in r]
        if hi_std_x:
            print(f"\n  [x_start std by noise level]  (noise > 0.5 = high, ≤ 0.5 = low)")
            print(f"    HIGH noise std  x:{_mean(hi_std_x):.4f}  y:{_mean(hi_std_y):.4f}")
            print(f"    LOW  noise std  x:{_mean(lo_std_x):.4f}  y:{_mean(lo_std_y):.4f}")
            print(f"    noise_level_mean: {_mean(nl_means):.4f}")
            if _mean(hi_std_x) > 3 * _mean(lo_std_x) + 0.01:
                print("    *** HIGH-NOISE blowup: std much larger at high noise — DPS amplification problem ***")
            elif _mean(lo_std_x) > 3 * _mean(hi_std_x) + 0.01:
                print("    *** LOW-NOISE blowup: guidance pushing off-manifold at late denoising steps ***")

    if uv_recs:
        cos_means = [r["cos_sim_mean"] for r in uv_recs if "cos_sim_mean" in r]
        cos_mins  = [r["cos_sim_min"]  for r in uv_recs if "cos_sim_min"  in r]
        uv_norms  = [r["unit_vec_norm_mean"] for r in uv_recs if "unit_vec_norm_mean" in r]
        dists     = [r["pred_to_goal_dist_mean"] for r in uv_recs if "pred_to_goal_dist_mean" in r]
        print(f"\n  [HILP Unit Vec Alignment]  n={len(uv_recs)}")
        print(f"    cos_sim (↑1=toward goal) — mean:{_mean(cos_means):.4f}  min:{_mean(cos_mins):.4f}")
        print(f"    unit_vec_norm            — mean:{_mean(uv_norms):.4f}  (should be ~1.0)")
        print(f"    pred→goal dist (unnorm)  — mean:{_mean(dists):.4f}")
        if _mean(cos_means) < 0.0:
            print("    *** WARNING: cos_sim < 0 — unit vec pointing AWAY from goal! ***")
        if _mean(uv_norms) < 0.5:
            print("    *** WARNING: unit_vec_norm < 0.5 — possible NaN/zero gradient ***")

    print("=" * 80)


def analyze(records: List[dict], log_name: str) -> None:
    # Always show grad comparison (prior vs guidance) if available
    _analyze_grad_comparison(records)
    _analyze_frozen_diag(records)

    guidance_records = _extract_guidance_records(records)

    if not guidance_records:
        print(f"\nNo guidance.combined records found in {log_name}.")
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
            f"{'G/A Ratio':<12} {'RDFLoss':<10} {'ParticleLoss':<14} {'PrtclScale':<11} {'BatchSz':<8}"
        )
        print("  " + "-" * 147)

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
            avg_particle = _mean([r.get("particle_loss", 0.0) for r in recs])
            avg_pscale = _mean([r.get("particle_guidance_scale", 0.0) for r in recs])
            avg_bsz = _mean([float(r.get("batch_size", 1)) for r in recs])

            print(
                f"  {scale:<10.4f} {avg_dpb:<18.6f} {avg_ftd:<20.6f} "
                f"{n:<7} {avg_goal_inner:<12.5f} {avg_anchor:<13.5f} {avg_goal_loss:<12.5f} "
                f"{avg_ga_ratio:<12.4f} {avg_rdf:<10.5f} {avg_particle:<14.5f} "
                f"{avg_pscale:<11.3f} {avg_bsz:<8.1f}"
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
