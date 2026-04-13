#!/usr/bin/env python3
"""Analyze uncertainty-estimate visualization debug logs.

Parses lines emitted by plan_viz.py in the form:
  [uncertainty-viz-debug] {"label": "...", ...}

Usage:
  python scripts/uncertainty_viz_log_analysis.py --log-file path/to/log.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


PREFIX = "[uncertainty-viz-debug] "


def parse_log_file(log_path: Path) -> List[Dict]:
    entries: List[Dict] = []
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


def print_summary(entries: List[Dict]) -> None:
    print(f"[uncertainty_viz_log_analysis] parsed {len(entries)} entries")
    if not entries:
        return

    unc_none = sum(1 for e in entries if e.get("unc_diag_is_none"))
    at_terminal = sum(1 for e in entries if e.get("depth") == e.get("terminal_depth"))
    over_terminal = sum(1 for e in entries if e.get("depth", -1) > e.get("terminal_depth", -1))

    print()
    print("Summary")
    print(f"  unc_diag_is_none=True : {unc_none}")
    print(f"  depth == terminal     : {at_terminal}")
    print(f"  depth > terminal      : {over_terminal}")

    depth_pairs = Counter((e.get("depth"), e.get("terminal_depth")) for e in entries)
    print()
    print("Depth Pairs")
    for (depth, terminal_depth), count in sorted(depth_pairs.items()):
        print(f"  depth={depth} terminal_depth={terminal_depth} : {count}")

    print()
    print("Entries")
    for e in entries:
        pos = e.get("expanded_node_pos")
        pos_str = "None" if pos is None else f"({pos[0]:.3f}, {pos[1]:.3f})"
        print(
            f"  line={e['_lineno']:>6}  "
            f"label={e.get('label')}  "
            f"depth={e.get('depth')}  "
            f"terminal={e.get('terminal_depth')}  "
            f"unc_diag_is_none={e.get('unc_diag_is_none')}  "
            f"pos={pos_str}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True, help="Path to stdout/stderr log file")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise SystemExit(f"Error: {log_path} not found")

    print_summary(parse_log_file(log_path))


if __name__ == "__main__":
    main()
