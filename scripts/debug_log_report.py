#!/usr/bin/env python3
"""
scripts/debug_log_report.py
Log analysis engine for AI research experiments.
Outputs a Markdown report with plan-following hypothesis analysis.
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SLP_REQUIRED = {"ts", "level", "tag"}
ERROR_LEVELS = {"ERROR", "CRITICAL"}


def parse_jsonl(log_path: Path) -> Tuple[List[Dict], bool]:
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
    is_slp = all(SLP_REQUIRED.issubset(r.keys()) for r in records) if records else False
    return records, is_slp


def extract_run_meta(records: List[Dict]) -> Dict:
    for r in records:
        if r.get("tag") == "run.start":
            return r.get("data", {})
    return {"run_id": "unknown", "purpose": "general_monitoring"}


def extract_errors(records: List[Dict]) -> List[Dict]:
    return [r for r in records if r.get("level") in ERROR_LEVELS]


def get_tag_records(records: List[Dict], tag: str) -> List[Dict]:
    return [r for r in records if r.get("tag") == tag]


def get_tag_prefix(records: List[Dict], prefix: str) -> List[Dict]:
    return [r for r in records if r.get("tag", "").startswith(prefix)]


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_exec_diag(records: List[Dict]) -> str:
    """
    H1: DQL agent can't reach sub-goals → plan not followed
    H2: open_loop_horizon too short → plan cut off before completion
    """
    exec_records = get_tag_records(records, "exec.diag")
    if not exec_records:
        return "_No `exec.diag` records found. Add logging in `_execute_plan_in_env` and re-run._\n"

    lines = []
    lines.append(f"**Total execution segments logged:** {len(exec_records)}\n")

    steps_ratios, sub_advances, mean_dists, mean_tracking_errs, done_early_count = [], [], [], [], 0

    for r in exec_records:
        d = r.get("data", {})
        executed = d.get("steps_executed", 0)
        open_loop = d.get("open_loop_horizon", 1)
        ratio = executed / open_loop if open_loop else 0
        steps_ratios.append(ratio)
        sub_advances.append(d.get("sub_goal_advances", 0))
        mean_dists.append(d.get("mean_dist_to_subgoal", -1))
        if d.get("mean_tracking_err", -1) >= 0:
            mean_tracking_errs.append(d["mean_tracking_err"])
        if d.get("done_early", False):
            done_early_count += 1

    avg_ratio = sum(steps_ratios) / len(steps_ratios)
    avg_advances = sum(sub_advances) / len(sub_advances)
    avg_dist = sum(d for d in mean_dists if d >= 0) / max(1, sum(1 for d in mean_dists if d >= 0))
    avg_track = sum(mean_tracking_errs) / len(mean_tracking_errs) if mean_tracking_errs else -1

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Avg steps_executed / open_loop_horizon | {avg_ratio:.2f} |")
    lines.append(f"| Avg sub_goal_advances per segment | {avg_advances:.1f} |")
    lines.append(f"| Avg mean_dist_to_subgoal | {avg_dist:.2f} |")
    lines.append(f"| Avg mean_tracking_err (agent vs plan) | {avg_track:.2f} |")
    lines.append(f"| Segments with done_early=True | {done_early_count}/{len(exec_records)} |")
    lines.append("")

    # Hypothesis verdicts
    lines.append("#### Hypothesis Verdicts\n")

    # H1: DQL can't follow sub-goals
    h1_evidence = []
    if avg_dist > 3.0:
        h1_evidence.append(f"mean_dist_to_subgoal={avg_dist:.2f} > 3.0 → agent stays far from sub-goal")
    if avg_advances < 1.0:
        h1_evidence.append(f"avg sub_goal_advances={avg_advances:.1f} < 1 → sub-goal rarely advanced (DQL not reaching it)")
    if avg_track > 5.0:
        h1_evidence.append(f"mean_tracking_err={avg_track:.2f} > 5.0 → agent deviates far from plan")
    if h1_evidence:
        lines.append(f"**H1 (DQL performance issue): LIKELY** ⚠️")
        for e in h1_evidence:
            lines.append(f"  - {e}")
    else:
        lines.append(f"**H1 (DQL performance issue): NOT SUPPORTED** ✓")
        lines.append(f"  - mean_dist_to_subgoal={avg_dist:.2f}, advances={avg_advances:.1f}, tracking_err={avg_track:.2f}")

    lines.append("")

    # H2: open_loop_horizon too short
    h2_evidence = []
    if avg_ratio > 0.95 and avg_advances > 0.5:
        h2_evidence.append(f"steps_executed/horizon={avg_ratio:.2f} (hitting the limit) with sub_goal_advances={avg_advances:.1f} > 0 (agent WAS following)")
    if done_early_count < len(exec_records) * 0.3:
        h2_evidence.append(f"Only {done_early_count}/{len(exec_records)} segments ended early (env done) → most are hitting horizon limit")
    if h2_evidence:
        lines.append(f"**H2 (open_loop_horizon too short): LIKELY** ⚠️")
        for e in h2_evidence:
            lines.append(f"  - {e}")
    else:
        lines.append(f"**H2 (open_loop_horizon too short): NOT SUPPORTED** ✓")

    lines.append("")

    # Show worst segments
    worst = sorted(exec_records, key=lambda r: r.get("data", {}).get("mean_dist_to_subgoal", 0), reverse=True)[:5]
    if worst:
        lines.append("#### Worst Segments (by mean_dist_to_subgoal)\n")
        lines.append("| steps | advances | mean_dist | tracking_err | done_early |")
        lines.append("|-------|----------|-----------|--------------|------------|")
        for r in worst:
            d = r.get("data", {})
            lines.append(f"| {d.get('steps_executed','?')}/{d.get('open_loop_horizon','?')} "
                         f"| {d.get('sub_goal_advances','?')} "
                         f"| {d.get('mean_dist_to_subgoal', '?'):.2f} "
                         f"| {d.get('mean_tracking_err','?'):.2f} "
                         f"| {d.get('done_early','?')} |")

    return "\n".join(lines)


def analyze_mcts_search(records: List[Dict]) -> str:
    """Analyze MCTS search behavior: value progression, achieved rates."""
    value_records = get_tag_records(records, "mcts.bidir_values")
    if not value_records:
        value_records = get_tag_records(records, "mcts.value_calc")

    lines = []
    if not value_records:
        lines.append("_No MCTS value records found._\n")
        return "\n".join(lines)

    achieved_count = sum(1 for r in value_records if "Achieved" in str(r.get("data", {})))
    not_reached = sum(1 for r in value_records if "NotReached" in str(r.get("data", {})))
    warp = sum(1 for r in value_records if "Warp" in str(r.get("data", {})))

    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Achieved | {achieved_count} |")
    lines.append(f"| NotReached | {not_reached} |")
    lines.append(f"| Warp | {warp} |")
    lines.append(f"| Total | {len(value_records)} |")
    lines.append("")

    if achieved_count + not_reached > 0:
        rate = achieved_count / (achieved_count + not_reached) * 100
        lines.append(f"**Achieved rate:** {rate:.1f}%\n")
        if rate < 20:
            lines.append("⚠️ Low achieved rate — bidir trees may not be meeting. Check `meeting_delta` and tree depth.")
        elif rate > 80:
            lines.append("✓ High achieved rate — bidir trees are connecting well.")

    return "\n".join(lines)


def analyze_obs_pos(records: List[Dict]) -> str:
    """Analyze node position progression across tree depths."""
    obs_pos_records = get_tag_records(records, "mcts.obs_pos")
    if not obs_pos_records:
        return "_No `mcts.obs_pos` records found._\n"

    lines = []
    lines.append(f"**Total OBS_POS records:** {len(obs_pos_records)}\n")

    # Group by tree tag
    by_tree: Dict[str, List] = defaultdict(list)
    for r in obs_pos_records:
        d = r.get("data", {})
        tag = d.get("tree_tag", d.get("tree", "unknown"))
        by_tree[tag].append(d)

    for tree_tag, entries in sorted(by_tree.items()):
        lines.append(f"**Tree: {tree_tag}** ({len(entries)} expansions)")
        # Show depth progression
        for e in entries[:8]:
            parent_pos = e.get("parent_pos", "?")
            child_pos = e.get("child_pos", "?")
            depth_from = e.get("depth_from", "?")
            depth_to = e.get("depth_to", "?")
            lines.append(f"  - depth {depth_from}→{depth_to}: {parent_pos} → {child_pos}")
        if len(entries) > 8:
            lines.append(f"  - ... ({len(entries)-8} more)")
        lines.append("")

    return "\n".join(lines)


def analyze_start_drift(records: List[Dict]) -> str:
    """
    H3: Plan start drifts from actual agent position over loops.
    If plan[0] is consistently far from actual start, the plan is being generated
    from stale/wrong initial position.
    """
    drift_records = get_tag_records(records, "mcts.start_drift")
    if not drift_records:
        return "_No `mcts.start_drift` records found._\n"

    lines = []
    drifts = []
    for r in drift_records:
        d = r.get("data", {})
        dist = d.get("dist_to_start", d.get("drift", -1))
        if dist >= 0:
            drifts.append(dist)

    if not drifts:
        return "_No drift distance values in records._\n"

    avg_drift = sum(drifts) / len(drifts)
    max_drift = max(drifts)

    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Avg plan[0]→start dist | {avg_drift:.2f} |")
    lines.append(f"| Max plan[0]→start dist | {max_drift:.2f} |")
    lines.append(f"| Records | {len(drifts)} |")
    lines.append("")

    lines.append("#### Hypothesis Verdict\n")
    if avg_drift > 2.0:
        lines.append(f"**H3 (plan starts far from agent): LIKELY** ⚠️")
        lines.append(f"  - avg drift={avg_drift:.2f} > 2.0 → each new plan does not start from current agent position")
        lines.append(f"  - This causes the agent to try to follow a plan that begins at the wrong location")
    else:
        lines.append(f"**H3 (plan start drift): NOT SUPPORTED** ✓")
        lines.append(f"  - avg drift={avg_drift:.2f} ≤ 2.0 → plan starts close to agent position")

    return "\n".join(lines)


def analyze_guidance(records: List[Dict]) -> str:
    """Analyze guidance.combined records — scale effectiveness, goal vs anchor balance."""
    g_records = get_tag_records(records, "guidance.combined")
    if not g_records:
        return "_No `guidance.combined` records found._\n"

    lines = []
    lines.append(f"**Total guidance calls:** {len(g_records)}\n")

    by_tree: Dict[str, list] = defaultdict(list)
    for r in g_records:
        d = r.get("data", {})
        tree = d.get("tree_tag", "unknown")
        by_tree[tree].append(d)

    for tree_tag, entries in sorted(by_tree.items()):
        label = "FORWARD" if "start" in tree_tag else ("BACKWARD" if "goal" in tree_tag else tree_tag)
        lines.append(f"#### {label} ({len(entries)} calls)\n")

        scales = [e.get("eff_goal_scale", 0.0) for e in entries]
        anchors = [abs(e.get("anchor_loss", 0.0)) for e in entries]
        goals = [abs(e.get("goal_loss", 0.0)) for e in entries]
        ratios = [e.get("goal_anchor_ratio", 0.0) for e in entries]
        rdfs = [abs(e.get("rdf_loss", 0.0)) for e in entries]
        ftds = [e["final_token_dist"][-1] if e.get("final_token_dist") else 0.0 for e in entries]

        def _avg(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        lines.append("| Metric | Mean |")
        lines.append("|--------|------|")
        lines.append(f"| eff_goal_scale | {_avg(scales):.4f} |")
        lines.append(f"| anchor_loss (abs) | {_avg(anchors):.5f} |")
        lines.append(f"| goal_loss (abs) | {_avg(goals):.5f} |")
        lines.append(f"| goal/anchor ratio | {_avg(ratios):.4f} |")
        lines.append(f"| rdf_loss (abs) | {_avg(rdfs):.5f} |")
        lines.append(f"| final_token_dist (last) | {_avg(ftds):.5f} |")
        lines.append("")

        # Quick verdicts
        avg_ftd = _avg(ftds)
        avg_ratio = _avg(ratios)
        if avg_ftd > 5.0:
            lines.append(f"⚠️ final_token_dist={avg_ftd:.2f} — plan end is far from goal after denoising.")
        elif avg_ftd < 1.0:
            lines.append(f"✓ final_token_dist={avg_ftd:.2f} — plan end is close to goal.")
        if avg_ratio < 0.1 and _avg(scales) > 0:
            lines.append(f"⚠️ goal/anchor ratio={avg_ratio:.3f} — anchor dominates; goal guidance may be ineffective.")
        elif avg_ratio > 10.0:
            lines.append(f"⚠️ goal/anchor ratio={avg_ratio:.1f} — goal dominates; consider increasing anchor_guidance_scale.")
        lines.append("")

    return "\n".join(lines)


def analyze_errors(records: List[Dict]) -> str:
    errors = extract_errors(records)
    if not errors:
        return "_No errors found._ ✓\n"

    lines = []
    lines.append(f"**Total errors:** {len(errors)}\n")
    lines.append("| Tag | Step | Message |")
    lines.append("|-----|------|---------|")
    for e in errors[:20]:
        data = e.get("data", {})
        msg = data.get("exc_msg", str(data))[:80]
        lines.append(f"| {e.get('tag','?')} | {e.get('step','?')} | {msg} |")
    if len(errors) > 20:
        lines.append(f"\n_... and {len(errors)-20} more errors_")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Report Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_md_report(log_path: Path, records: List[Dict], is_slp: bool, run_meta: Dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = run_meta.get("run_id", "unknown")
    purpose = run_meta.get("purpose", "general_monitoring")

    sections = []

    # Header
    sections.append(f"# Log Analysis Report\n")
    sections.append(f"- **File:** `{log_path.name}`")
    sections.append(f"- **Generated:** {now}")
    sections.append(f"- **Run ID:** {run_id}")
    sections.append(f"- **Purpose:** {purpose}")
    sections.append(f"- **Records:** {len(records)} | **Format:** {'SLP' if is_slp else 'Heuristic'}")
    sections.append("")

    # Errors
    sections.append("---\n## Errors\n")
    sections.append(analyze_errors(records))

    # Plan-Following Analysis (the main diagnostic)
    sections.append("---\n## Plan-Following Analysis\n")
    sections.append("### Execution Diagnostics (`exec.diag`)\n")
    sections.append("> Tests H1 (DQL can't follow sub-goals) and H2 (open_loop_horizon too short)\n")
    sections.append(analyze_exec_diag(records))

    sections.append("### Plan Start Drift (`mcts.start_drift`)\n")
    sections.append("> Tests H3: does the new plan start from the current agent position?\n")
    sections.append(analyze_start_drift(records))

    # Guidance Analysis
    sections.append("---\n## Guidance Analysis\n")
    sections.append("### Per-Tree Guidance Stats (`guidance.combined`)\n")
    sections.append("> Goal guidance effectiveness, anchor/RDF balance, and final token distance to goal.\n")
    sections.append(analyze_guidance(records))

    # MCTS Search
    sections.append("---\n## MCTS Search Analysis\n")
    sections.append("### Value / Achieved Rate\n")
    sections.append(analyze_mcts_search(records))

    sections.append("### Node Position Progression (`mcts.obs_pos`)\n")
    sections.append(analyze_obs_pos(records))

    # All tags summary
    sections.append("---\n## Tag Summary\n")
    tag_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        tag_counts[r.get("tag", "unknown")] += 1
    sections.append("| Tag | Count |")
    sections.append("|-----|-------|")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:30]:
        sections.append(f"| `{tag}` | {count} |")

    return "\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[debug_log_report] Parsing: {log_path}")
    records, is_slp = parse_jsonl(log_path)
    print(f"[debug_log_report] Records: {len(records)}, SLP={is_slp}")

    run_meta = extract_run_meta(records)
    md = build_md_report(log_path, records, is_slp, run_meta)

    output_path = output_dir / f"{log_path.stem}_analysis.md"
    output_path.write_text(md, encoding="utf-8")
    print(f"[debug_log_report] Report saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
