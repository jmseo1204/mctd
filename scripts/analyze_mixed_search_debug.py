#!/usr/bin/env python3
"""Summarize structured mixed-search debug logs.

Parses lines emitted with the prefix:
  [mixed-search-debug] {...json...}

Typical usage:
  python scripts/analyze_mixed_search_debug.py --log-file path/to/run.log
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


PREFIX = "[mixed-search-debug] "


def parse_log_file(log_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            if PREFIX not in line:
                continue
            payload = line.split(PREFIX, 1)[1].strip()
            try:
                rec = json.loads(payload)
            except json.JSONDecodeError as exc:
                print(f"[WARN] line {lineno}: failed to parse JSON: {exc}")
                continue
            rec["_lineno"] = lineno
            entries.append(rec)
    return entries


def _short_root_summary(root_entry: Dict[str, Any]) -> str:
    if not root_entry:
        return "None"
    root = root_entry.get("root") or {}
    return (
        f"{root_entry.get('tree_tag')}: "
        f"flag={root.get('is_expandable_flag')} "
        f"now={root.get('is_expandable_now')} "
        f"empty={root.get('n_empty_alive_slots')} "
        f"dead={root.get('n_permanently_dead_slots')} "
        f"created={root.get('n_created_children')} "
        f"cluster={root.get('cluster_subplans')}"
    )


def print_summary(entries: List[Dict[str, Any]], show_video_events: bool) -> None:
    print(f"[analyze_mixed_search_debug] parsed {len(entries)} structured entries")
    if not entries:
        return

    tag_counts = Counter(e.get("tag", "<missing>") for e in entries)
    print()
    print("Tag Counts")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")

    selection_entries = [e for e in entries if e.get("tag") == "round.selection"]
    if selection_entries:
        print()
        print("Selections")
        for e in selection_entries:
            parents = e.get("selected_parents", [])
            parent_desc = ", ".join(
                f"{p.get('tree_tag')}:{p.get('node')}@d{p.get('depth')} v={p.get('value')}"
                for p in parents
            ) or "<none>"
            print(
                f"  line={e['_lineno']:>6} loop={e.get('loop')} "
                f"selected={e.get('selected_parent_count')} "
                f"g_before={e.get('global_search_num_before')} "
                f"parents=[{parent_desc}]"
            )

    meeting_entries = [e for e in entries if e.get("tag") == "round.meeting_check"]
    if meeting_entries:
        print()
        print("Meeting Checks")
        for e in meeting_entries:
            roots = e.get("roots", [])
            root_str = " | ".join(_short_root_summary(r) for r in roots)
            print(
                f"  line={e['_lineno']:>6} loop={e.get('loop')} "
                f"is_meeting={e.get('is_meeting')} "
                f"trees_exhausted={e.get('trees_exhausted')} "
                f"gap={e.get('gap')} delta={e.get('meeting_delta')} "
                f"best={e.get('best_node')} target={e.get('target_node')} "
                f"target_hist={e.get('target_plan_history_len')}"
            )
            print(f"    roots: {root_str}")

    break_entries = [e for e in entries if e.get("tag") == "loop.break"]
    if break_entries:
        print()
        print("Loop Breaks")
        for e in break_entries:
            print(
                f"  line={e['_lineno']:>6} loop={e.get('loop')} "
                f"reason={e.get('reason')} global={e.get('global_search_num')}"
            )
            roots = e.get("roots")
            if roots:
                print(f"    roots: {' | '.join(_short_root_summary(r) for r in roots)}")

    complete_entries = [e for e in entries if e.get("tag") == "search.complete"]
    if complete_entries:
        last = complete_entries[-1]
        print()
        print("Final Search Summary")
        print(
            f"  loops={last.get('loops')} "
            f"global={last.get('global_search_num')}/{last.get('max_search_num')} "
            f"best={last.get('best_node')} "
            f"is_meeting={last.get('is_meeting')} terminate={last.get('terminate')}"
        )
        roots = last.get("roots", [])
        if roots:
            print(f"  roots: {' | '.join(_short_root_summary(r) for r in roots)}")

    viz_checks = [e for e in entries if e.get("tag") == "uncertainty_viz.check"]
    video_skips = [e for e in entries if e.get("tag") == "video.skip"]
    video_logged = [e for e in entries if e.get("tag") == "video.logged"]
    if viz_checks or video_skips or video_logged:
        print()
        print("Video Summary")
        print(f"  uncertainty_viz.check: {len(viz_checks)}")
        print(f"  video.logged: {len(video_logged)}")
        print(f"  video.skip: {len(video_skips)}")
        if video_skips:
            skip_reasons = Counter(e.get("reason", "<missing>") for e in video_skips)
            print("  skip_reasons:")
            for reason, count in sorted(skip_reasons.items()):
                print(f"    {reason}: {count}")

    if show_video_events:
        detailed = [
            e
            for e in entries
            if e.get("tag") in {"uncertainty_viz.check", "video.skip", "video.logged"}
        ]
        if detailed:
            print()
            print("Video Events")
            for e in detailed:
                print(
                    f"  line={e['_lineno']:>6} tag={e.get('tag')} "
                    f"candidate={e.get('candidate')} prefix={e.get('log_prefix')} "
                    f"reason={e.get('reason')} should_log={e.get('should_log')} "
                    f"frames={e.get('n_frames')} hist_steps={e.get('hist_steps')}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, help="Path to stdout/stderr log file")
    parser.add_argument(
        "--show-video-events",
        action="store_true",
        help="Print every video-related structured event",
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise SystemExit(f"Error: {log_path} not found")

    print_summary(parse_log_file(log_path), show_video_events=args.show_video_events)


if __name__ == "__main__":
    main()
