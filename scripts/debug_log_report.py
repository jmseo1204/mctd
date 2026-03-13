#!/usr/bin/env python3
"""
scripts/debug_log_report.py
Log analysis engine for AI research experiments.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SLP_REQUIRED = {"ts", "level", "tag"}
ERROR_LEVELS = {"ERROR", "CRITICAL"}
AI_ERROR_PATTERNS = [
    (r"nan", "NaN Detected"),
    (r"inf(?:inity)?", "Inf Detected"),
    (r"out of memory|OOM|CUDA out", "OOM"),
    (r"gradient.*explod", "Gradient Explosion"),
    (r"cuda error", "CUDA Error"),
]

def parse_jsonl(log_path: Path) -> Tuple[List[Dict], bool]:
    records = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
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
        if "purpose" in r:
            return r.copy()
    return {"run_id": "unknown", "purpose": "general_monitoring"} if records else {}

def extract_errors(records: List[Dict]) -> List[Dict]:
    errors = []
    for i, r in enumerate(records):
        if r.get("level") in ERROR_LEVELS:
            errors.append(r)
    return errors

def extract_numeric_series(records: List[Dict]) -> Dict[str, List[Dict]]:
    series: Dict[str, List] = defaultdict(list)
    for r in records:
        tag = r.get("tag", "")
        data = r.get("data", {})
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not (math.isnan(v) if isinstance(v, float) else False):
                    series_key = f"{tag}.{k}" if k != "value" else tag
                    series[series_key].append({
                        "step": r.get("step"),
                        "value": v,
                    })
    return dict(series)

def extract_tensor_stats(records: List[Dict]) -> List[Dict]:
    tensor_records = []
    for r in records:
        data = r.get("data", {})
        if isinstance(data, dict) and any(k in data for k in {"nan_count", "inf_count", "norm"}):
            tensor_records.append({
                "tag": r.get("tag", ""),
                "step": r.get("step"),
                "data": data,
            })
    return tensor_records

def build_html_report(log_path, records, is_slp, run_meta, errors, series, tensor_stats):
    sections = []
    
    # Summary
    error_count = len(errors)
    sections.append(f"""
    <section>
      <h2>Summary</h2>
      <p>Records: {len(records)} | Errors: {error_count} | Format: {"SLP" if is_slp else "Heuristic"}</p>
      <p>Run ID: {run_meta.get("run_id", "unknown")}</p>
      <p>Purpose: {run_meta.get("purpose", "general_monitoring")}</p>
    </section>
    """)
    
    # Errors
    if errors:
        error_rows = ""
        for e in errors[:50]:
            error_rows += f"<tr><td>{e.get('tag')}</td><td>{e.get('step')}</td><td>{str(e.get('data'))[:100]}</td></tr>"
        sections.append(f"""
        <section>
          <h2>Errors ({len(errors)})</h2>
          <table border="1"><tr><th>Tag</th><th>Step</th><th>Data</th></tr>{error_rows}</table>
        </section>
        """)
    
    # Series Stats
    if series:
        series_info = ""
        for tag, points in list(series.items())[:20]:
            values = [p["value"] for p in points if isinstance(p["value"], (int, float))]
            if values:
                series_info += f"<tr><td>{tag}</td><td>{len(points)}</td><td>{min(values):.4f}</td><td>{max(values):.4f}</td><td>{sum(values)/len(values):.4f}</td></tr>"
        sections.append(f"""
        <section>
          <h2>Numeric Series ({len(series)})</h2>
          <table border="1"><tr><th>Tag</th><th>Points</th><th>Min</th><th>Max</th><th>Mean</th></tr>{series_info}</table>
        </section>
        """)
    
    # Tensor Stats
    if tensor_stats:
        tensor_rows = ""
        for t in tensor_stats[:30]:
            data = t.get("data", {})
            tensor_rows += f"<tr><td>{t.get('tag')}</td><td>{t.get('step')}</td><td>{data.get('nan_count', 0)}</td><td>{data.get('inf_count', 0)}</td></tr>"
        sections.append(f"""
        <section>
          <h2>Tensor Diagnostics ({len(tensor_stats)})</h2>
          <table border="1"><tr><th>Tag</th><th>Step</th><th>NaN</th><th>Inf</th></tr>{tensor_rows}</table>
        </section>
        """)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Log Analysis: {log_path.stem}</title>
  <style>
    body {{ font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }}
    section {{ background: #2a2a2a; border: 1px solid #444; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    h2 {{ color: #66bb6a; margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: #333; padding: 8px; text-align: left; }}
    td {{ padding: 6px; border-bottom: 1px solid #333; }}
    tr:hover td {{ background: #333; }}
  </style>
</head>
<body>
  <h1>Log Analysis Report</h1>
  <p>File: {log_path.name} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  {"".join(sections)}
</body>
</html>"""
    return html

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
    print(f"[debug_log_report] Records: {len(records)}")

    run_meta = extract_run_meta(records)
    errors = extract_errors(records)
    series = extract_numeric_series(records)
    tensor_stats = extract_tensor_stats(records)

    print(f"[debug_log_report] Errors: {len(errors)}, Series: {len(series)}")

    html = build_html_report(log_path, records, is_slp, run_meta, errors, series, tensor_stats)
    
    output_path = output_dir / f"{log_path.stem}_analysis.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"[debug_log_report] Report saved: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
