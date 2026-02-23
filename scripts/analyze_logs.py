#!/usr/bin/env python3
import argparse, json, math, re, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SLP_REQUIRED = {"ts", "level", "tag"}
ERROR_LEVELS = {"ERROR", "CRITICAL"}
AI_ERROR_PATTERNS = [(r"nan", "NaN"), (r"memory", "Memory"), (r"shape mismatch", "Shape")]

def parse_jsonl(log_path: Path) -> Tuple[List[Dict], bool]:
    records = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict): records.append(obj)
            except: pass
    is_slp = len([r for r in records if SLP_REQUIRED.issubset(r.keys())]) / max(len(records), 1) >= 0.5
    return records, is_slp

def extract_run_meta(records: List[Dict]) -> Dict:
    for r in records:
        if r.get("tag") == "run.start":
            meta = r.get("data", {}).copy()
            meta.update({"run_id": r.get("run_id", "unknown"), "purpose": r.get("purpose", ""), "start_ts": r.get("ts", 0.0)})
            return meta
    return {"run_id": records[0].get("run_id", "unknown") if records else "unknown", "purpose": "", "start_ts": records[0].get("ts", 0) if records else 0}

def extract_errors(records: List[Dict]) -> List[Dict]:
    errors = []
    for i, r in enumerate(records):
        is_error = r.get("level") in ERROR_LEVELS
        patterns = [label for pattern, label in AI_ERROR_PATTERNS if re.search(pattern, json.dumps(r, default=str).lower())]
        if is_error or patterns:
            error_record = r.copy()
            error_record["_detected_patterns"] = patterns
            errors.append(error_record)
    return errors

def extract_numeric_series(records: List[Dict]) -> Dict[str, List[Dict]]:
    series = defaultdict(list)
    for r in records:
        data = r.get("data", {})
        if not isinstance(data, dict): continue
        for k, v in data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                try:
                    if not math.isnan(v) and not math.isinf(v):
                        series_key = f"{r.get('tag', '')}.{k}" if k != "value" else r.get('tag', '')
                        series[series_key].append({"step": r.get("step"), "ts": r.get("ts", 0.0), "value": v})
                except: pass
    return dict(series)

def build_html_report(log_path, records, is_slp, run_meta, errors, series) -> str:
    start_ts = run_meta.get("start_ts", 0)
    end_ts = records[-1].get("ts", start_ts) if records else start_ts
    duration_s = end_ts - start_ts
    error_count = len(errors)
    series_count = len(series)
    
    memory_series = {k: v for k, v in series.items() if "memory" in k.lower()}
    memory_html = ""
    if memory_series:
        memory_html = "<h3>Memory Metrics</h3><table class='data-table'><thead><tr><th>Tag</th><th>Count</th><th>Min</th><th>Max</th><th>Avg</th></tr></thead><tbody>"
        for tag, points in memory_series.items():
            vals = [p['value'] for p in points]
            memory_html += f"<tr><td>{tag}</td><td>{len(points)}</td><td>{min(vals):.1f}</td><td>{max(vals):.1f}</td><td>{sum(vals)/len(vals):.1f}</td></tr>"
        memory_html += "</tbody></table>"
    
    error_html = ""
    if errors:
        error_html = f"<h3>Errors Detected: {error_count}</h3><table class='data-table'><thead><tr><th>Tag</th><th>Level</th><th>Time</th></tr></thead><tbody>"
        for e in errors[:20]:
            ts_str = datetime.fromtimestamp(e.get("ts", 0)).strftime("%H:%M:%S")
            error_html += f"<tr class='error-row'><td>{e.get('tag','')}</td><td>{e.get('level','')}</td><td>{ts_str}</td></tr>"
        error_html += "</tbody></table>"
    
    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Log Analysis: {log_path.stem}</title><style>
    body {{ background: #0f1117; color: #e2e8f0; font-family: monospace; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
    th {{ background: #1a1d27; padding: 8px; border: 1px solid #2e3347; text-align: left; }}
    td {{ padding: 6px 8px; border: 1px solid #2e3347; }}
    .error-row {{ color: #f87171; }}
    h3 {{ margin: 15px 0 10px; border-bottom: 1px solid #2e3347; padding-bottom: 5px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }}
    .summary-item {{ background: #1a1d27; padding: 15px; border-radius: 4px; border: 1px solid #2e3347; }}
    .summary-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; }}
    .summary-value {{ font-size: 18px; font-weight: bold; color: #6366f1; }}
    </style></head><body>
    <h1>Memory Efficiency Diagnosis Report</h1>
    <div class="summary">
        <div class="summary-item"><div class="summary-label">Records</div><div class="summary-value">{len(records)}</div></div>
        <div class="summary-item"><div class="summary-label">Duration (s)</div><div class="summary-value">{duration_s:.1f}</div></div>
        <div class="summary-item"><div class="summary-label">Errors</div><div class="summary-value">{error_count}</div></div>
        <div class="summary-item"><div class="summary-label">Metrics</div><div class="summary-value">{series_count}</div></div>
    </div>
    {memory_html}
    {error_html}
    <h3>All Metrics Summary</h3>
    <table class='data-table'><thead><tr><th>Tag</th><th>Points</th><th>Min</th><th>Max</th><th>Avg</th></tr></thead><tbody>"""
    
    for tag, points in sorted(series.items())[:50]:
        vals = [p['value'] for p in points]
        if vals:
            html += f"<tr><td>{tag}</td><td>{len(points)}</td><td>{min(vals):.6f}</td><td>{max(vals):.6f}</td><td>{sum(vals)/len(vals):.6f}</td></tr>"
    
    html += f"""</tbody></table></body></html>"""
    return html

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[analyze_logs] Parsing: {log_path}")
    records, is_slp = parse_jsonl(log_path)
    print(f"[analyze_logs] Records: {len(records)}")
    
    run_meta = extract_run_meta(records)
    errors = extract_errors(records)
    series = extract_numeric_series(records)
    
    print(f"[analyze_logs] Errors: {len(errors)}, Series: {len(series)}")
    
    html = build_html_report(log_path, records, is_slp, run_meta, errors, series)
    output_path = output_dir / f"{log_path.stem}_analysis.html"
    output_path.write_text(html)
    
    print(f"[analyze_logs] Report saved: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
