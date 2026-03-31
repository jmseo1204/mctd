#!/usr/bin/env python3
"""
scripts/latency_analysis.py
Latency analysis for MCTD planning logs.
Reads timing.* records from a jsonl log and prints a summary to stdout.

Expected log tags (insert these in df_planning.py):
  timing.mpc_loop         — one record per interact() call
  timing.phase_breakdown  — one record per _run_mcts_search() call (per tree)
  timing.gpu_sample_step  — one record per parallel_plan() call
  timing.guidance_fn_in_denoising — one record per guidance call batch
  timing.guidance_breakdown       — per-call guidance timing
  timing.execute_plan     — one record per _execute_plan_in_env() call
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_jsonl(log_path: Path) -> List[Dict]:
    records = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                pass
    return records


def get_tag(records: List[Dict], tag: str) -> List[Dict]:
    return [r for r in records if r.get("tag") == tag]


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def fmt(ms, unit="ms"):
    if unit == "ms":
        return f"{ms:.1f} ms"
    return f"{ms:.1f}"


def section(title: str):
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def table(rows, headers):
    """Print a simple fixed-width table."""
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    fmt_row = lambda r: "  " + "  ".join(str(v).ljust(w) for v, w in zip(r, col_w))
    print(fmt_row(headers))
    print("  " + "  ".join("-" * w for w in col_w))
    for row in rows:
        print(fmt_row(row))


# ─────────────────────────────────────────────────────────────────────────────
# Per-section analyzers
# ─────────────────────────────────────────────────────────────────────────────

def report_mpc_loop(records: List[Dict]):
    """timing.mpc_loop: total per-episode latency (planning + execution)."""
    recs = get_tag(records, "timing.mpc_loop")
    if not recs:
        print("  [timing.mpc_loop] No records — add logging in interact()")
        return

    planning = [r["data"].get("planning_ms", 0) for r in recs if "data" in r]
    execution = [r["data"].get("execution_ms", 0) for r in recs if "data" in r]
    total = [r["data"].get("total_ms", p + e) for r, p, e in zip(recs, planning, execution)]

    section("MPC Loop  (timing.mpc_loop)")
    table(
        [
            ("n episodes",       len(recs),                ""),
            ("avg planning",     f"{avg(planning):.0f} ms", f"max {max(planning):.0f} ms"),
            ("avg execution",    f"{avg(execution):.0f} ms", f"max {max(execution):.0f} ms"),
            ("avg total",        f"{avg(total):.0f} ms",    f"max {max(total):.0f} ms"),
        ],
        ["Metric", "Mean", "Peak"],
    )


def report_phase_breakdown(records: List[Dict]):
    """timing.phase_breakdown: MCTS phase decomposition per tree per search."""
    recs = get_tag(records, "timing.phase_breakdown")
    if not recs:
        print("  [timing.phase_breakdown] No records — add logging in _run_mcts_search()")
        return

    by_tree: Dict[str, list] = defaultdict(list)
    for r in recs:
        d = r.get("data", {})
        by_tree[d.get("tree_tag", "unknown")].append(d)

    section("MCTS Phase Breakdown  (timing.phase_breakdown)")
    for tree_tag, entries in sorted(by_tree.items()):
        label = tree_tag
        n = len(entries)
        print(f"\n  Tree: {label}  (n={n})")

        phases = ["selection", "expansion", "simulation", "backprop", "early_term"]
        rows = []
        total_vals = [e.get("total_ms", 0) for e in entries]
        for ph in phases:
            ms_key = f"{ph}_ms"
            vals = [e.get(ms_key, 0) for e in entries]
            ratio_key = f"{ph}_ratio"
            ratios = [e.get(ratio_key, 0) for e in entries]
            mean_ms = avg(vals)
            mean_ratio = avg(ratios)
            if mean_ms == 0 and not any(v > 0 for v in vals):
                continue
            rows.append((ph, f"{mean_ms:.0f} ms", f"{mean_ratio*100:.1f}%", f"{max(vals):.0f} ms"))
        rows.append(("TOTAL", f"{avg(total_vals):.0f} ms", "100%", f"{max(total_vals):.0f} ms"))
        table(rows, ["Phase", "Mean", "Ratio", "Peak"])

        # Sub-phases inside simulation
        sim_sub = {
            "sim_noise_zeropad": "sim_noise_zeropad_ms",
            "sim_value_estimation": "sim_value_estimation_ms",
            "sim_value_calculation": "sim_value_calculation_ms",
            "sim_node_allocation": "sim_node_allocation_ms",
        }
        sub_rows = []
        for label_s, key in sim_sub.items():
            vals = [e.get(key, 0) for e in entries]
            if any(v > 0 for v in vals):
                sub_rows.append((label_s, f"{avg(vals):.0f} ms", f"{max(vals):.0f} ms"))
        if sub_rows:
            print("  Simulation sub-phases:")
            table(sub_rows, ["Sub-phase", "Mean", "Peak"])

        # Bottleneck warning
        exp_ratios = [e.get("expansion_ratio", 0) for e in entries]
        if avg(exp_ratios) > 0.85:
            print(f"  ⚠  expansion dominates ({avg(exp_ratios)*100:.0f}%) — GPU diffusion is the bottleneck")


def report_gpu_sample(records: List[Dict]):
    """timing.gpu_sample_step: GPU denoising cost per parallel_plan() call."""
    recs = get_tag(records, "timing.gpu_sample_step")
    if not recs:
        print("  [timing.gpu_sample_step] No records — add logging in parallel_plan()")
        return

    data = [r.get("data", {}) for r in recs]
    total_gpu = [d.get("total_gpu_ms", 0) for d in data]
    per_step = [d.get("mean_per_step_ms", 0) for d in data]
    n_steps = data[0].get("n_denoising_steps", "?") if data else "?"

    section("GPU Denoising  (timing.gpu_sample_step)")
    table(
        [
            ("n calls",           len(recs),                    ""),
            ("n_denoising_steps", n_steps,                      ""),
            ("avg total GPU",     f"{avg(total_gpu):.0f} ms",   f"max {max(total_gpu):.0f} ms"),
            ("avg ms / step",     f"{avg(per_step):.1f} ms",    f"max {max(per_step):.1f} ms"),
        ],
        ["Metric", "Mean", "Peak"],
    )


def report_guidance_overhead(records: List[Dict]):
    """timing.guidance_fn_in_denoising: guidance cost as fraction of GPU time."""
    recs = get_tag(records, "timing.guidance_fn_in_denoising")
    if not recs:
        return  # optional, skip silently

    data = [r.get("data", {}) for r in recs]
    total = [d.get("total_ms", 0) for d in data]
    mean_call = [d.get("mean_ms", 0) for d in data]
    pct = [d.get("guidance_pct_of_gpu", 0) for d in data]

    section("Guidance Overhead in Denoising  (timing.guidance_fn_in_denoising)")
    table(
        [
            ("n batches",           len(recs),                    ""),
            ("avg total guidance",  f"{avg(total):.0f} ms",       f"max {max(total):.0f} ms"),
            ("avg ms / call",       f"{avg(mean_call):.1f} ms",   ""),
            ("guidance % of GPU",   f"{avg(pct):.1f}%",           f"max {max(pct):.1f}%"),
        ],
        ["Metric", "Mean", "Peak"],
    )
    if avg(pct) > 50:
        print(f"  ⚠  guidance takes {avg(pct):.0f}% of GPU time — consider reducing calls or HILP batch size")


def report_guidance_breakdown(records: List[Dict]):
    """timing.guidance_breakdown: per-component guidance timing."""
    recs = get_tag(records, "timing.guidance_breakdown")
    if not recs:
        return

    data = [r.get("data", {}) for r in recs]
    components = ["hilp_ms", "goal_ms", "rdf_ms", "total_ms"]
    rows = []
    for comp in components:
        vals = [d.get(comp, 0) for d in data if d.get(comp) is not None]
        if vals:
            rows.append((comp.replace("_ms", ""), f"{avg(vals):.1f} ms", f"{max(vals):.1f} ms"))

    if rows:
        section("Guidance Breakdown  (timing.guidance_breakdown)")
        print(f"  n calls: {len(recs)}")
        table(rows, ["Component", "Mean", "Peak"])

        # hilp_grad_ms / hilp_value_ms breakdown
        grad_vals  = [d.get("hilp_grad_ms")  for d in data if d.get("hilp_grad_ms")  is not None]
        value_vals = [d.get("hilp_value_ms") for d in data if d.get("hilp_value_ms") is not None]
        if grad_vals or value_vals:
            hilp_vals = [d.get("hilp_ms") for d in data if d.get("hilp_ms") is not None]
            mean_hilp = avg(hilp_vals) if hilp_vals else 1.0
            print()
            print("  HILP sub-breakdown (grad vs value inference):")
            rows = []
            if grad_vals:
                mean_g = avg(grad_vals)
                rows.append(("JAX grad extraction",
                              f"{mean_g:.1f} ms", f"{max(grad_vals):.1f} ms",
                              f"{mean_g/mean_hilp*100:.1f}%"))
            if value_vals:
                mean_v = avg(value_vals)
                rows.append(("value inference",
                              f"{mean_v:.1f} ms", f"{max(value_vals):.1f} ms",
                              f"{mean_v/mean_hilp*100:.1f}%"))
            table(rows, ["Sub-component", "Mean", "Peak", "% of hilp"])


def report_execute_plan(records: List[Dict]):
    """timing.execute_plan: plan execution loop timing."""
    recs = get_tag(records, "timing.execute_plan")
    if not recs:
        print("  [timing.execute_plan] No records — add logging in _execute_plan_in_env()")
        return

    data = [r.get("data", {}) for r in recs]
    total    = [d.get("total_exec_ms", 0) for d in data]
    loop_cnt = [d.get("loop_cnt", 0) for d in data]

    section("Plan Execution  (timing.execute_plan)")
    table(
        [
            ("n segments",         len(recs),                       ""),
            ("avg steps executed", f"{avg(loop_cnt):.0f}",          f"max {max(loop_cnt):.0f}"),
            ("avg total exec",     f"{avg(total):.0f} ms",          f"max {max(total):.0f} ms"),
            ("avg ms / step",      f"{avg(total)/max(avg(loop_cnt),1):.1f} ms", ""),
        ],
        ["Metric", "Mean", "Peak"],
    )

    # Per-component breakdown table
    components = [
        ("dist (subgoal TD/L2)",  "dist_total_ms",     "dist_mean_ms",     "dist_pct"),
        ("action compute",        "action_total_ms",   "action_mean_ms",   "action_pct"),
        ("env.step()",            "env_step_total_ms", "env_step_mean_ms", "env_step_pct"),
        ("_get_sim_state()",      "get_state_total_ms","get_state_mean_ms","get_state_pct"),
        ("overhead (traj/reward)","overhead_total_ms", "overhead_mean_ms", "overhead_pct"),
    ]
    rows = []
    for label, tot_key, mean_key, pct_key in components:
        tot_vals  = [d.get(tot_key,  0) for d in data]
        mean_vals = [d.get(mean_key, 0) for d in data]
        pct_vals  = [d.get(pct_key,  0) for d in data]
        if any(v > 0 for v in tot_vals):
            rows.append((
                label,
                f"{avg(tot_vals):.0f} ms",
                f"{avg(mean_vals):.2f} ms/step",
                f"{avg(pct_vals):.1f}%",
            ))
    if rows:
        print()
        table(rows, ["Component", "Total (avg)", "Per-step (avg)", "% of exec"])

    env_max = [d.get("env_step_max_ms", 0) for d in data]
    if any(v > 0 for v in env_max):
        print(f"\n  env.step() spike: max {max(env_max):.1f} ms")


def report_lifecycle(records: List[Dict]):
    """lifecycle.*: full job time breakdown as percentages."""
    init_recs   = get_tag(records, "lifecycle.algo_init_complete")
    start_recs  = get_tag(records, "lifecycle.interact_start")
    hilp_recs   = get_tag(records, "lifecycle.hilp_loaded")
    done_recs   = get_tag(records, "lifecycle.interact_complete")
    val_recs    = get_tag(records, "lifecycle.validation_complete")

    if not start_recs and not done_recs:
        print("  [lifecycle.*] No records — add lifecycle logging in df_planning.py / exp_base.py")
        return

    section("Job Time Breakdown  (lifecycle.*)")

    n = max(len(start_recs), len(done_recs))
    for i in range(n):
        sr = start_recs[i] if i < len(start_recs) else {}
        dr = done_recs[i]  if i < len(done_recs)  else {}
        ir = init_recs[i]  if i < len(init_recs)  else {}
        vr = val_recs[i]   if i < len(val_recs)   else {}

        sd = sr.get("data", {})
        dd = dr.get("data", {})

        algo_init   = ir.get("data", {}).get("elapsed_s")         # module import → __init__ done
        pre         = sd.get("pre_interact_elapsed_s", 0)         # module import → interact() start
        total       = dd.get("total_elapsed_s", 0)                # module import → interact() end
        itotal      = dd.get("interact_elapsed_s", 0)             # time spent inside interact()
        val_done    = vr.get("data", {}).get("elapsed_s")         # module import → validation complete

        hilp_elapsed = None
        if hilp_recs and i < len(hilp_recs):
            hilp_elapsed = hilp_recs[i].get("data", {}).get("elapsed_s", 0)

        grand_total = val_done if val_done else total

        def pct(d): return f"{d/grand_total*100:.0f}%" if grand_total else "?"
        def fmt(d): return f"{d:.1f}s"

        print(f"\n  Episode {i+1}  |  total: {grand_total:.1f}s")
        rows = []

        if algo_init is not None:
            rows.append(("Python start + algo.__init__",
                         fmt(algo_init), pct(algo_init)))
            trainer_setup = pre - algo_init
            if trainer_setup > 0:
                rows.append(("WandB + ckpt load + pl.Trainer setup",
                             fmt(trainer_setup), pct(trainer_setup)))
        else:
            rows.append(("Python start + algo build + WandB + ckpt",
                         fmt(pre), pct(pre)))

        env_hilp = (hilp_elapsed - pre) if hilp_elapsed else None
        if env_hilp is not None and env_hilp > 0:
            rows.append(("Env creation + HILP JAX init",
                         fmt(env_hilp), pct(env_hilp)))

        mcts_exec = itotal - (env_hilp or 0) if hilp_elapsed else itotal
        if mcts_exec > 0:
            rows.append(("MCTS + plan execution",
                         fmt(mcts_exec), pct(mcts_exec)))

        if val_done and total:
            wandb_fin = val_done - total
            if wandb_fin > 0:
                rows.append(("WandB finalize", fmt(wandb_fin), pct(wandb_fin)))

        table(rows, ["Phase", "Duration", "%"])


def report_expansions(records: List[Dict]):
    """timing.expansion: per-expansion breakdown for FWD and BWD trees."""
    recs = get_tag(records, "timing.expansion")
    if not recs:
        print("  [timing.expansion] No records — add per-expansion logging in _run_mcts_search() single_step branch")
        return

    by_tree: Dict[str, list] = defaultdict(list)
    for r in recs:
        d = r.get("data", {})
        by_tree[d.get("tree_tag", "unknown")].append(d)

    section("Per-Expansion Timing  (timing.expansion)")
    all_expansion_ms = []
    for tree_tag, entries in sorted(by_tree.items()):
        exp_ms = [e.get("expansion_ms", 0) for e in entries]
        sel_ms = [e.get("selection_ms", 0) for e in entries]
        sim_ms = [e.get("simulation_ms", 0) for e in entries]
        all_expansion_ms.extend(exp_ms)
        print(f"\n  Tree: {tree_tag}  (n_expansions={len(entries)})")
        table(
            [
                ("total expansions", len(entries),                    ""),
                ("avg expansion_ms", f"{avg(exp_ms):.0f} ms",        f"max {max(exp_ms):.0f} ms"),
                ("avg selection_ms", f"{avg(sel_ms):.1f} ms",        ""),
                ("avg simulation_ms",f"{avg(sim_ms):.1f} ms",        ""),
                ("total GPU time",   f"{sum(exp_ms):.0f} ms",        ""),
            ],
            ["Metric", "Mean/Total", "Peak"],
        )
    if len(by_tree) > 1:
        print(f"\n  Combined across all trees:")
        print(f"    total expansions : {len(recs)}")
        print(f"    avg expansion_ms : {avg(all_expansion_ms):.0f} ms")
        print(f"    total GPU time   : {sum(all_expansion_ms):.0f} ms")


def report_hilp_viz(records: List[Dict]):
    """timing.hilp_viz: heatmap + grad-field computation (inside planning_ms)."""
    recs = get_tag(records, "timing.hilp_viz")
    if not recs:
        print("  [timing.hilp_viz] No records — add logging around _compute_hilp_heatmap/_compute_guidance_grad_fields")
        return
    data = [r.get("data", {}) for r in recs]
    hm   = [d.get("heatmap_ms",    0) for d in data]
    gf   = [d.get("grad_field_ms", 0) for d in data]
    tot  = [d.get("total_ms",      0) for d in data]
    section("HILP Viz (inside planning)  (timing.hilp_viz)")
    table(
        [
            ("n calls",           len(recs),                        ""),
            ("avg heatmap",       f"{avg(hm):.0f} ms",  f"max {max(hm):.0f} ms"),
            ("avg grad_field",    f"{avg(gf):.0f} ms",  f"max {max(gf):.0f} ms"),
            ("avg total",         f"{avg(tot):.0f} ms", f"max {max(tot):.0f} ms"),
        ],
        ["Metric", "Mean", "Peak"],
    )


def report_plan_postproc(records: List[Dict]):
    """timing.plan_postproc: reorder + pre-exec viz (between planning_end and exec_start)."""
    recs = get_tag(records, "timing.plan_postproc")
    if not recs:
        print("  [timing.plan_postproc] No records — add logging around _reorder_plan_by_proximity / pre-exec viz")
        return
    data    = [r.get("data", {}) for r in recs]
    reorder = [d.get("reorder_ms",       0) for d in data]
    viz     = [d.get("pre_exec_viz_ms",  0) for d in data]
    tot     = [d.get("total_ms",         0) for d in data]
    nframes = [d.get("n_frames",         0) for d in data]
    section("Plan Postproc (between planning & exec)  (timing.plan_postproc)")
    table(
        [
            ("n calls",            len(recs),                          ""),
            ("avg n_frames",       f"{avg(nframes):.0f}",              ""),
            ("avg reorder",        f"{avg(reorder):.0f} ms",  f"max {max(reorder):.0f} ms"),
            ("avg pre_exec_viz",   f"{avg(viz):.0f} ms",     f"max {max(viz):.0f} ms"),
            ("avg total",          f"{avg(tot):.0f} ms",     f"max {max(tot):.0f} ms"),
        ],
        ["Metric", "Mean", "Peak"],
    )


def report_post_exec(records: List[Dict]):
    """timing.post_exec: trajectory stack + viz + log_image after execution."""
    recs = get_tag(records, "timing.post_exec")
    if not recs:
        print("  [timing.post_exec] No records — add logging around post-execution visualization")
        return
    data  = [r.get("data", {}) for r in recs]
    tot   = [d.get("total_ms", 0) for d in data]
    steps = [d.get("n_steps",  0) for d in data]
    section("Post-Execution Viz  (timing.post_exec)")
    table(
        [
            ("n calls",        len(recs),                         ""),
            ("avg n_steps",    f"{avg(steps):.0f}",               ""),
            ("avg total",      f"{avg(tot):.0f} ms",   f"max {max(tot):.0f} ms"),
        ],
        ["Metric", "Mean", "Peak"],
    )


def report_summary(records: List[Dict]):
    """Quick tag count overview to see what timing data is available."""
    from collections import Counter
    timing_tags = [r.get("tag", "") for r in records if r.get("tag", "").startswith("timing.")]
    counts = Counter(timing_tags)
    if not counts:
        print("\n  No timing.* records found in this log.")
        print("  Insert logging code in df_planning.py using the format from latency_analysis.py docstring.")
        return
    print("\n  Available timing tags:")
    for tag, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {count:5d}  {tag}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latency analysis for MCTD logs")
    parser.add_argument("--log-file", required=True, help="Path to .jsonl log file")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    records = parse_jsonl(log_path)
    print(f"[latency_analysis] {log_path.name}  ({len(records)} records)")

    report_summary(records)
    report_lifecycle(records)
    report_mpc_loop(records)
    report_phase_breakdown(records)
    report_expansions(records)
    report_gpu_sample(records)
    report_guidance_overhead(records)
    report_guidance_breakdown(records)
    report_execute_plan(records)
    report_hilp_viz(records)
    report_plan_postproc(records)
    report_post_exec(records)


if __name__ == "__main__":
    main()
