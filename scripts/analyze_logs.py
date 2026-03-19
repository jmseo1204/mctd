#!/usr/bin/env python3
"""
scripts/analyze_logs.py

Log analysis engine for AI research experiments.
Reads a jsonl log file (SLP-standard or heuristic fallback),
generates a self-contained interactive HTML report using Plotly.

Structured Log Protocol (SLP) fields recognized:
  ts, level, run_id, phase, step, tag, group, depth, data, source, purpose
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


# ══════════════════════════════════════════════════════════════════════════════
# 1. 파싱 레이어
# ══════════════════════════════════════════════════════════════════════════════

SLP_REQUIRED = {"ts", "level", "tag"}
SLP_OPTIONAL = {"run_id", "phase", "step", "group", "depth", "data", "source", "purpose"}
ERROR_LEVELS = {"ERROR", "CRITICAL"}
AI_ERROR_PATTERNS = [
    (r"nan", "NaN Detected"),
    (r"inf(?:inity)?", "Inf Detected"),
    (r"out of memory|OOM|CUDA out", "OOM / Memory Error"),
    (r"gradient.*explod|exploding gradient", "Gradient Explosion"),
    (r"gradient.*vanish|vanishing gradient", "Gradient Vanishing"),
    (r"cuda error|cudnn error", "CUDA Error"),
    (r"assert.*error|assertionerror", "Assertion Error"),
    (r"shape mismatch|size mismatch", "Shape/Size Mismatch"),
    (r"keyerror|indexerror|valueerror|typeerror|runtimeerror", "Python Runtime Error"),
]


def parse_jsonl(log_path: Path) -> Tuple[List[Dict], bool]:
    records = []
    parse_errors = 0

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
                parse_errors += 1
                heuristic = _heuristic_parse(line, i)
                if heuristic:
                    records.append(heuristic)

    is_slp = _detect_slp(records)

    if not is_slp:
        records = [_normalize_record(r, i) for i, r in enumerate(records)]

    return records, is_slp


def _detect_slp(records: List[Dict]) -> bool:
    if not records:
        return False
    slp_count = sum(1 for r in records if SLP_REQUIRED.issubset(r.keys()))
    return slp_count / len(records) >= 0.5


def _heuristic_parse(line: str, line_num: int) -> Optional[Dict]:
    record: Dict[str, Any] = {"_raw": line, "_line": line_num}

    ts_match = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", line)
    if ts_match:
        try:
            dt = datetime.fromisoformat(ts_match.group().replace(" ", "T"))
            record["ts"] = dt.timestamp()
        except ValueError:
            pass

    level_match = re.search(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b", line, re.I)
    if level_match:
        record["level"] = level_match.group().upper().replace("WARN", "WARNING")

    kv_matches = re.findall(r"(\w+)[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
    if kv_matches:
        record["data"] = {k: float(v) for k, v in kv_matches}
        record["tag"] = kv_matches[0][0]

    record.setdefault("level", "INFO")
    record.setdefault("tag", f"raw.line_{line_num}")
    record.setdefault("ts", 0.0)

    return record if len(record) > 3 else None


def _normalize_record(record: Dict, idx: int) -> Dict:
    normalized = {
        "ts": record.get("ts", 0.0),
        "level": record.get("level", "INFO"),
        "run_id": record.get("run_id", "unknown"),
        "phase": record.get("phase", "unknown"),
        "step": record.get("step", record.get("epoch", record.get("iter", None))),
        "tag": record.get("tag", f"unknown.record_{idx}"),
        "group": record.get("group", ""),
        "depth": record.get("depth", 0),
        "data": record.get("data", {}),
        "source": record.get("source", ""),
        "purpose": record.get("purpose", ""),
        "_heuristic": True,
    }
    if not isinstance(normalized["data"], dict):
        normalized["data"] = {"value": normalized["data"]}
    return normalized


# ══════════════════════════════════════════════════════════════════════════════
# 2. 분석 레이어
# ══════════════════════════════════════════════════════════════════════════════

def extract_run_meta(records: List[Dict]) -> Dict:
    for r in records:
        if r.get("tag") == "run.start" or r.get("tag", "").startswith("run."):
            meta = r.get("data", {}).copy()
            meta.update({
                "run_id": r.get("run_id", "unknown"),
                "purpose": r.get("purpose", ""),
                "start_ts": r.get("ts", 0.0),
            })
            return meta
    if records:
        return {
            "run_id": records[0].get("run_id", "unknown"),
            "purpose": records[0].get("purpose", ""),
            "start_ts": records[0].get("ts", 0.0),
            "_inferred": True,
        }
    return {}


def extract_errors(records: List[Dict]) -> List[Dict]:
    errors = []

    for i, r in enumerate(records):
        is_error = r.get("level") in ERROR_LEVELS
        detected_patterns = []

        record_str = json.dumps(r, default=str).lower()
        for pattern, label in AI_ERROR_PATTERNS:
            if re.search(pattern, record_str, re.I):
                detected_patterns.append(label)

        if is_error or detected_patterns:
            error_record = r.copy()
            error_record["_detected_patterns"] = detected_patterns
            error_record["_context_before"] = records[max(0, i-3):i]
            errors.append(error_record)

    return errors


def extract_numeric_series(records: List[Dict]) -> Dict[str, List[Dict]]:
    series: Dict[str, List] = defaultdict(list)

    for r in records:
        tag = r.get("tag", "")
        data = r.get("data", {})
        step = r.get("step")
        ts = r.get("ts", 0.0)
        phase = r.get("phase", "")

        if not isinstance(data, dict):
            continue

        for k, v in data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not (math.isnan(v) or math.isinf(v)):
                    series_key = f"{tag}.{k}" if k != "value" else tag
                    series[series_key].append({
                        "step": step,
                        "ts": ts,
                        "value": v,
                        "phase": phase,
                    })

    for key in series:
        series[key].sort(key=lambda x: (x["step"] is None, x["step"] or 0, x["ts"]))

    return dict(series)


def detect_anomalies(series: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    anomalies: Dict[str, List] = defaultdict(list)
    WINDOW = 10
    SPIKE_THRESHOLD = 3.0

    for tag, points in series.items():
        if len(points) < 5:
            continue

        values = [p["value"] for p in points]

        for i in range(WINDOW, len(points)):
            window = values[max(0, i-WINDOW):i]
            wmean = sum(window) / len(window)
            wstd = (sum((x - wmean)**2 for x in window) / len(window)) ** 0.5

            current = values[i]

            if wstd > 1e-8 and abs(current - wmean) > SPIKE_THRESHOLD * wstd:
                anomalies[tag].append({
                    **points[i],
                    "anomaly_type": "spike",
                    "z_score": abs(current - wmean) / wstd,
                })

        tail = values[int(len(values) * 0.8):]
        if len(tail) > 5:
            tail_range = max(tail) - min(tail)
            total_range = max(values) - min(values) if max(values) != min(values) else 1e-8
            if tail_range / total_range < 0.01:
                anomalies[tag].append({
                    "step": points[-1]["step"],
                    "ts": points[-1]["ts"],
                    "value": tail[-1],
                    "phase": points[-1]["phase"],
                    "anomaly_type": "plateau",
                    "detail": f"Last 20% range: {tail_range:.6f}",
                })

        if len(values) >= 10:
            last_half = values[len(values)//2:]
            if all(last_half[i] <= last_half[i+1] for i in range(len(last_half)-1)):
                if last_half[-1] > last_half[0] * 10:
                    anomalies[tag].append({
                        **points[-1],
                        "anomaly_type": "divergence",
                        "detail": f"Value grew {last_half[-1]/max(last_half[0],1e-8):.1f}x in second half",
                    })

    return dict(anomalies)


def extract_tensor_stats(records: List[Dict]) -> List[Dict]:
    tensor_records = []
    tensor_fields = {"nan_count", "inf_count", "norm", "mean", "std", "shape"}

    for r in records:
        data = r.get("data", {})
        if not isinstance(data, dict):
            continue
        if tensor_fields.intersection(data.keys()):
            tensor_records.append({
                "ts": r.get("ts", 0.0),
                "step": r.get("step"),
                "tag": r.get("tag", ""),
                "source": r.get("source", ""),
                "label": data.get("label", ""),
                "nan_count": data.get("nan_count", 0),
                "inf_count": data.get("inf_count", 0),
                "norm": data.get("norm"),
                "mean": data.get("mean"),
                "std": data.get("std"),
                "min": data.get("min"),
                "max": data.get("max"),
                "shape": data.get("shape"),
            })

    return tensor_records


def extract_timing(records: List[Dict]) -> List[Dict]:
    timing_records = []
    for r in records:
        tag = r.get("tag", "")
        if "timing" in tag or "time" in tag or "ms" in tag or "latency" in tag:
            data = r.get("data", {})
            value = data.get("value") if isinstance(data, dict) else None
            if value is not None:
                timing_records.append({
                    "step": r.get("step"),
                    "tag": tag,
                    "value_ms": value,
                    "phase": r.get("phase", ""),
                })
    return timing_records


def extract_purpose_relevant(records: List[Dict], purpose: str) -> Dict[str, List]:
    if not purpose or purpose == "general_monitoring":
        return {}

    keywords = re.split(r"[_\-\s]+", purpose.lower())
    relevant: Dict[str, List] = defaultdict(list)

    for r in records:
        tag = r.get("tag", "").lower()
        group = r.get("group", "").lower()
        record_str = f"{tag} {group}"

        for kw in keywords:
            if len(kw) > 2 and kw in record_str:
                relevant[kw].append(r)

    return dict(relevant)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MCTD 특화 분석
# ══════════════════════════════════════════════════════════════════════════════

def extract_mctd_analysis(records: List[Dict]) -> Dict:
    """MCTD/MCTS planning 특화 분석 - 궤적, 탐색, 실행 단계 분석."""
    result = {
        "search_events": [],
        "exec_loops": [],
        "plan_quality": [],
        "sub_goal_updates": [],
        "action_stats": [],
        "trajectory_summary": [],
        "value_estimates": [],
        "loop_outcomes": [],
    }

    for r in records:
        tag = r.get("tag", "")
        data = r.get("data", {}) or {}
        step = r.get("step")

        # MCTS 탐색 이벤트
        if "search" in tag or "mcts" in tag.lower() or "tree" in tag:
            result["search_events"].append({
                "tag": tag,
                "step": step,
                "ts": r.get("ts", 0.0),
                "data": data,
            })

        # 실행 루프
        if "exec" in tag or "loop" in tag or "execute" in tag:
            result["exec_loops"].append({
                "tag": tag,
                "step": step,
                "ts": r.get("ts", 0.0),
                "data": data,
            })

        # 플랜 품질
        if "plan" in tag and ("quality" in tag or "score" in tag or "value" in tag):
            result["plan_quality"].append({
                "tag": tag,
                "step": step,
                "value": data.get("value") or data.get("score") or data.get("quality"),
                "data": data,
            })

        # sub_goal 업데이트
        if "sub_goal" in tag or "subgoal" in tag or "goal" in tag:
            result["sub_goal_updates"].append({
                "tag": tag,
                "step": step,
                "data": data,
            })

        # 액션 통계
        if "action" in tag:
            result["action_stats"].append({
                "tag": tag,
                "step": step,
                "data": data,
            })

        # value 추정
        if "value" in tag or "reward" in tag:
            val = data.get("value") or data.get("reward")
            if isinstance(val, (int, float)):
                result["value_estimates"].append({
                    "tag": tag,
                    "step": step,
                    "value": val,
                })

        # 루프 결과 (success/fail/timeout)
        if any(kw in tag for kw in ["success", "fail", "done", "timeout", "achieve", "result", "outcome"]):
            result["loop_outcomes"].append({
                "tag": tag,
                "step": step,
                "ts": r.get("ts", 0.0),
                "data": data,
            })

        # 궤적 요약
        if "trajectory" in tag or "rollout" in tag or "interact" in tag:
            result["trajectory_summary"].append({
                "tag": tag,
                "step": step,
                "ts": r.get("ts", 0.0),
                "data": data,
            })

    return result


def extract_position_trajectory(records: List[Dict]) -> List[Dict]:
    """위치 궤적 데이터 추출 (obs_pos, qpos, position 관련)."""
    trajectory = []
    for r in records:
        tag = r.get("tag", "")
        data = r.get("data", {}) or {}

        # pos, position, qpos, obs_pos 관련 레코드 추출
        if any(kw in tag.lower() for kw in ["pos", "position", "coord", "loc"]):
            x = data.get("x") or data.get("pos_x") or (data.get("obs_pos", [None])[0] if isinstance(data.get("obs_pos"), list) else None)
            y = data.get("y") or data.get("pos_y") or (data.get("obs_pos", [None, None])[1] if isinstance(data.get("obs_pos"), list) and len(data.get("obs_pos", [])) > 1 else None)
            if x is not None or y is not None:
                trajectory.append({
                    "tag": tag,
                    "step": r.get("step"),
                    "ts": r.get("ts", 0.0),
                    "x": x,
                    "y": y,
                    "data": data,
                })

        # qpos[:2] 있는 경우
        qpos = data.get("qpos")
        if isinstance(qpos, list) and len(qpos) >= 2:
            trajectory.append({
                "tag": tag,
                "step": r.get("step"),
                "ts": r.get("ts", 0.0),
                "x": qpos[0],
                "y": qpos[1],
                "data": data,
            })

    return trajectory


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTML 생성 레이어
# ══════════════════════════════════════════════════════════════════════════════

def build_html_report(
    log_path: Path,
    records: List[Dict],
    is_slp: bool,
    run_meta: Dict,
    errors: List[Dict],
    series: Dict[str, List],
    anomalies: Dict[str, List],
    tensor_stats: List[Dict],
    timing: List[Dict],
    purpose_relevant: Dict[str, List],
    mctd_analysis: Dict = None,
    position_trajectory: List[Dict] = None,
) -> str:

    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False

    sections_html = []

    # ── 0. 요약 배너 ──────────────────────────────────────────────────────────
    start_ts = run_meta.get("start_ts", 0)
    end_ts = records[-1].get("ts", start_ts) if records else start_ts
    duration_s = end_ts - start_ts
    purpose = run_meta.get("purpose", "")
    slp_badge = (
        '<span class="badge badge-green">SLP Standard</span>'
        if is_slp else
        '<span class="badge badge-yellow">Heuristic Parse</span>'
    )

    error_count = len(errors)
    error_badge_class = "badge-red" if error_count > 0 else "badge-green"
    anomaly_count = sum(len(v) for v in anomalies.values())

    summary_html = f"""
    <div class="summary-banner">
      <div class="summary-item"><span class="summary-label">Run ID</span><span class="summary-value">{run_meta.get('run_id','unknown')}</span></div>
      <div class="summary-item"><span class="summary-label">Job</span><span class="summary-value">{run_meta.get('job_name', run_meta.get('purpose',''))}</span></div>
      <div class="summary-item"><span class="summary-label">Records</span><span class="summary-value">{len(records):,}</span></div>
      <div class="summary-item"><span class="summary-label">Duration</span><span class="summary-value">{duration_s:.1f}s</span></div>
      <div class="summary-item"><span class="summary-label">Errors</span><span class="summary-value"><span class="badge {error_badge_class}">{error_count}</span></span></div>
      <div class="summary-item"><span class="summary-label">Anomalies</span><span class="summary-value">{anomaly_count}</span></div>
      <div class="summary-item"><span class="summary-label">Log Format</span><span class="summary-value">{slp_badge}</span></div>
    </div>
    """
    sections_html.append(summary_html)

    # ── 1. 실험 설정 요약 ────────────────────────────────────────────────────
    meta_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in run_meta.items()
        if not k.startswith("_")
    )
    sections_html.append(f"""
    <section>
      <h2>Experiment Configuration</h2>
      <table class="data-table">
        <thead><tr><th>Key</th><th>Value</th></tr></thead>
        <tbody>{meta_rows}</tbody>
      </table>
    </section>
    """)

    # ── 2. MCTD 특화 분석 섹션 ────────────────────────────────────────────────
    if mctd_analysis:
        # 루프 결과 요약
        loop_outcomes = mctd_analysis.get("loop_outcomes", [])
        outcome_rows = ""
        for o in loop_outcomes[:100]:
            data_preview = json.dumps(o.get("data", {}), default=str)[:200]
            outcome_rows += f"<tr><td>{o.get('step','')}</td><td>{o.get('tag','')}</td><td>{data_preview}</td></tr>"

        # 탐색 이벤트 요약
        search_events = mctd_analysis.get("search_events", [])
        search_rows = ""
        for s in search_events[:50]:
            data_preview = json.dumps(s.get("data", {}), default=str)[:200]
            ts_str = datetime.fromtimestamp(s.get("ts", 0)).strftime("%H:%M:%S")
            search_rows += f"<tr><td>{ts_str}</td><td>{s.get('step','')}</td><td>{s.get('tag','')}</td><td>{data_preview}</td></tr>"

        # trajectory 이벤트
        traj_events = mctd_analysis.get("trajectory_summary", [])
        traj_rows = ""
        for t in traj_events[:100]:
            data_preview = json.dumps(t.get("data", {}), default=str)[:300]
            ts_str = datetime.fromtimestamp(t.get("ts", 0)).strftime("%H:%M:%S")
            traj_rows += f"<tr><td>{ts_str}</td><td>{t.get('step','')}</td><td>{t.get('tag','')}</td><td>{data_preview}</td></tr>"

        sections_html.append(f"""
        <section class="purpose-section">
          <h2>MCTD Planning Analysis</h2>

          <h3 style="margin:16px 0 8px; color:var(--warning);">Loop Outcomes ({len(loop_outcomes)} events)</h3>
          {"<p class='no-data'>No outcome events found.</p>" if not loop_outcomes else f"""
          <div class="table-scroll">
          <table class="data-table">
            <thead><tr><th>Step</th><th>Tag</th><th>Data</th></tr></thead>
            <tbody>{outcome_rows}</tbody>
          </table>
          </div>"""}

          <h3 style="margin:16px 0 8px; color:var(--info);">Trajectory / Rollout Events ({len(traj_events)} events)</h3>
          {"<p class='no-data'>No trajectory events found.</p>" if not traj_events else f"""
          <div class="table-scroll">
          <table class="data-table">
            <thead><tr><th>Time</th><th>Step</th><th>Tag</th><th>Data</th></tr></thead>
            <tbody>{traj_rows}</tbody>
          </table>
          </div>"""}

          <h3 style="margin:16px 0 8px; color:var(--success);">MCTS Search Events ({len(search_events)} events)</h3>
          {"<p class='no-data'>No search events found.</p>" if not search_events else f"""
          <div class="table-scroll">
          <table class="data-table">
            <thead><tr><th>Time</th><th>Step</th><th>Tag</th><th>Data</th></tr></thead>
            <tbody>{search_rows}</tbody>
          </table>
          </div>"""}
        </section>
        """)

    # ── 3. 에러/예외 타임라인 ────────────────────────────────────────────────
    if errors:
        error_rows = ""
        for e in errors:
            ts_str = datetime.fromtimestamp(e.get("ts", 0)).strftime("%H:%M:%S.%f")[:-3]
            patterns = ", ".join(e.get("_detected_patterns", [])) or e.get("level", "")
            exc_data = e.get("data", {})
            exc_type = exc_data.get("exc_type", "") if isinstance(exc_data, dict) else ""
            exc_msg = str(exc_data.get("exc_msg", ""))[:200] if isinstance(exc_data, dict) else ""
            source = e.get("source", "")
            step = e.get("step", "")
            row_class = "error-row" if e.get("level") in ERROR_LEVELS else "warning-row"
            error_rows += f"""
            <tr class="{row_class}">
              <td>{ts_str}</td>
              <td>{e.get('level','')}</td>
              <td>{e.get('tag','')}</td>
              <td>{step}</td>
              <td>{exc_type}</td>
              <td>{exc_msg}</td>
              <td>{patterns}</td>
              <td>{source}</td>
            </tr>"""

        sections_html.append(f"""
        <section>
          <h2>Error / Exception Timeline <span class="badge badge-red">{len(errors)}</span></h2>
          <div class="table-scroll">
          <table class="data-table">
            <thead><tr>
              <th>Time</th><th>Level</th><th>Tag</th><th>Step</th>
              <th>Type</th><th>Message</th><th>Patterns</th><th>Source</th>
            </tr></thead>
            <tbody>{error_rows}</tbody>
          </table>
          </div>
        </section>
        """)
    else:
        sections_html.append("""
        <section>
          <h2>Error / Exception Timeline</h2>
          <p class="no-data">No errors detected.</p>
        </section>
        """)

    # ── 4. 위치 궤적 시각화 ──────────────────────────────────────────────────
    if position_trajectory and PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        import plotly.io as pio

        xs = [p["x"] for p in position_trajectory if p.get("x") is not None]
        ys = [p["y"] for p in position_trajectory if p.get("y") is not None]
        steps = [p.get("step", i) for i, p in enumerate(position_trajectory) if p.get("x") is not None]

        if xs and ys:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name="Agent Position",
                marker=dict(size=5, color=steps, colorscale="Viridis", showscale=True,
                           colorbar=dict(title="Step")),
                line=dict(width=1.5, color="rgba(100,149,237,0.5)"),
                hovertemplate="step=%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                text=steps,
            ))
            if xs:
                fig.add_trace(go.Scatter(x=[xs[0]], y=[ys[0]], mode="markers",
                    name="Start", marker=dict(size=12, color="green", symbol="star")))
                fig.add_trace(go.Scatter(x=[xs[-1]], y=[ys[-1]], mode="markers",
                    name="End", marker=dict(size=12, color="red", symbol="x")))

            fig.update_layout(
                title="Agent Position Trajectory",
                xaxis_title="X", yaxis_title="Y",
                template="plotly_white", height=450,
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )
            sections_html.append(f"""
            <section>
              <h2>Position Trajectory</h2>
              {pio.to_html(fig, full_html=False, include_plotlyjs=False)}
            </section>
            """)

    # ── 5. Loss / Metric 곡선 ────────────────────────────────────────────────
    metric_series = {
        k: v for k, v in series.items()
        if any(kw in k.lower() for kw in ["loss", "metric", "score", "reward", "acc", "error", "dist", "value"])
    }

    if metric_series and PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        import plotly.io as pio
        fig = go.Figure()
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]
        for i, (tag, points) in enumerate(metric_series.items()):
            xs = [p["step"] if p["step"] is not None else p["ts"] for p in points]
            ys = [p["value"] for p in points]
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=tag, line=dict(color=color),
                marker=dict(size=4),
                hovertemplate=f"<b>{tag}</b><br>step=%{{x}}<br>value=%{{y:.6f}}<extra></extra>",
            ))

            if tag in anomalies:
                ax = [a["step"] if a.get("step") is not None else a.get("ts") for a in anomalies[tag]]
                ay = [a["value"] for a in anomalies[tag]]
                labels = [a["anomaly_type"] for a in anomalies[tag]]
                fig.add_trace(go.Scatter(
                    x=ax, y=ay, mode="markers",
                    name=f"{tag} anomaly",
                    marker=dict(color="red", size=10, symbol="x"),
                    text=labels,
                    hovertemplate="<b>ANOMALY</b><br>%{text}<br>step=%{x}<br>value=%{y:.6f}<extra></extra>",
                ))

        fig.update_layout(
            title="Loss / Metric Curves",
            xaxis_title="Step / Epoch",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        sections_html.append(f"""
        <section>
          <h2>Loss / Metric Curves</h2>
          {chart_html}
        </section>
        """)

    # ── 6. 텐서 상태 진단 ────────────────────────────────────────────────────
    if tensor_stats:
        problematic = [t for t in tensor_stats if (t.get("nan_count") or 0) > 0 or (t.get("inf_count") or 0) > 0]

        tensor_rows = ""
        for t in tensor_stats[:200]:
            nan_c = t.get("nan_count", 0) or 0
            inf_c = t.get("inf_count", 0) or 0
            row_class = "error-row" if (nan_c > 0 or inf_c > 0) else ""
            tensor_rows += f"""
            <tr class="{row_class}">
              <td>{t.get('step','')}</td>
              <td>{t.get('tag','')}</td>
              <td>{t.get('label','')}</td>
              <td>{t.get('shape','')}</td>
              <td class="{'cell-error' if nan_c>0 else ''}">{nan_c}</td>
              <td class="{'cell-error' if inf_c>0 else ''}">{inf_c}</td>
              <td>{f"{t.get('mean'):.4f}" if t.get('mean') is not None else ''}</td>
              <td>{f"{t.get('std'):.4f}" if t.get('std') is not None else ''}</td>
              <td>{f"{t.get('norm'):.4f}" if t.get('norm') is not None else ''}</td>
            </tr>"""

        problem_summary = (
            f'<div class="alert alert-error">{len(problematic)} tensors with NaN/Inf detected.</div>'
            if problematic else
            '<div class="alert alert-ok">No NaN/Inf tensors detected.</div>'
        )

        sections_html.append(f"""
        <section>
          <h2>Tensor State Diagnostics</h2>
          {problem_summary}
          <div class="table-scroll">
          <table class="data-table">
            <thead><tr>
              <th>Step</th><th>Tag</th><th>Label</th><th>Shape</th>
              <th>NaN</th><th>Inf</th><th>Mean</th><th>Std</th><th>Norm</th>
            </tr></thead>
            <tbody>{tensor_rows}</tbody>
          </table>
          </div>
        </section>
        """)

    # ── 7. 성능 프로파일링 ────────────────────────────────────────────────────
    if timing and PLOTLY_AVAILABLE:
        import plotly.graph_objects as go
        import plotly.io as pio
        timing_by_tag: Dict[str, List] = defaultdict(list)
        for t in timing:
            timing_by_tag[t["tag"]].append(t)

        avg_times = {tag: sum(p["value_ms"] for p in pts)/len(pts) for tag, pts in timing_by_tag.items()}
        bar_fig = go.Figure(go.Bar(
            x=list(avg_times.keys()),
            y=list(avg_times.values()),
            text=[f"{v:.1f}ms" for v in avg_times.values()],
            textposition="auto",
        ))
        bar_fig.update_layout(
            title="Average Latency per Stage",
            xaxis_title="Stage",
            yaxis_title="Avg Time (ms)",
            template="plotly_white",
            height=300,
        )

        sections_html.append(f"""
        <section>
          <h2>Performance Profile</h2>
          {pio.to_html(bar_fig, full_html=False, include_plotlyjs=False)}
        </section>
        """)

    # ── 8. 목적 기반 커스텀 섹션 ─────────────────────────────────────────────
    if purpose_relevant:
        for kw, rel_records in purpose_relevant.items():
            if not rel_records:
                continue
            rows = ""
            for r in rel_records[:100]:
                data_preview = json.dumps(r.get("data", {}), default=str)[:150]
                rows += f"""
                <tr>
                  <td>{r.get('step','')}</td>
                  <td>{r.get('tag','')}</td>
                  <td>{r.get('level','')}</td>
                  <td>{data_preview}</td>
                  <td>{r.get('source','')}</td>
                </tr>"""

            sections_html.append(f"""
            <section class="purpose-section">
              <h2>Purpose-Aware: <em>{purpose}</em> → <em>{kw}</em>
                <span class="badge badge-blue">{len(rel_records)} records</span>
              </h2>
              <div class="table-scroll">
              <table class="data-table">
                <thead><tr><th>Step</th><th>Tag</th><th>Level</th><th>Data</th><th>Source</th></tr></thead>
                <tbody>{rows}</tbody>
              </table>
              </div>
            </section>
            """)

    # ── 9. 이상치 요약 ───────────────────────────────────────────────────────
    if anomalies:
        anomaly_rows = ""
        for tag, anom_list in anomalies.items():
            for a in anom_list:
                val = a.get('value', '')
                val_str = f"{val:.6f}" if isinstance(val, float) else str(val)
                anomaly_rows += f"""
                <tr>
                  <td>{tag}</td>
                  <td>{a.get('step','')}</td>
                  <td><span class="badge badge-yellow">{a.get('anomaly_type','')}</span></td>
                  <td>{val_str}</td>
                  <td>{a.get('detail', a.get('z_score', ''))}</td>
                </tr>"""

        sections_html.append(f"""
        <section>
          <h2>Anomaly Summary <span class="badge badge-yellow">{anomaly_count}</span></h2>
          <table class="data-table">
            <thead><tr><th>Tag</th><th>Step</th><th>Type</th><th>Value</th><th>Detail</th></tr></thead>
            <tbody>{anomaly_rows}</tbody>
          </table>
        </section>
        """)

    # ── 10. 전체 로그 레코드 (검색 가능) ────────────────────────────────────
    all_rows = ""
    for r in records[:500]:
        ts_str = datetime.fromtimestamp(r.get("ts", 0)).strftime("%H:%M:%S")
        data_preview = json.dumps(r.get("data", {}), default=str)[:200]
        level = r.get("level", "")
        row_class = "error-row" if level in ERROR_LEVELS else ("warning-row" if level == "WARNING" else "")
        all_rows += f"""
        <tr class="{row_class}">
          <td>{ts_str}</td>
          <td>{r.get('step','')}</td>
          <td>{r.get('phase','')}</td>
          <td>{level}</td>
          <td>{r.get('tag','')}</td>
          <td>{data_preview}</td>
          <td>{r.get('source','')}</td>
        </tr>"""

    sections_html.append(f"""
    <section>
      <h2>All Log Records (first 500) <span class="badge badge-blue">{len(records)} total</span></h2>
      <input type="text" id="logSearch" placeholder="Filter by tag, data, source..."
             style="width:100%;padding:8px 12px;margin-bottom:12px;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:6px;font-size:13px;"
             onkeyup="filterTable(this.value)">
      <div class="table-scroll">
      <table class="data-table" id="logTable">
        <thead><tr><th>Time</th><th>Step</th><th>Phase</th><th>Level</th><th>Tag</th><th>Data</th><th>Source</th></tr></thead>
        <tbody>{all_rows}</tbody>
      </table>
      </div>
    </section>
    <script>
    function filterTable(query) {{
      var q = query.toLowerCase();
      var rows = document.querySelectorAll('#logTable tbody tr');
      rows.forEach(function(row) {{
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
      }});
    }}
    </script>
    """)

    # ── HTML 조립 ─────────────────────────────────────────────────────────────
    plotly_cdn = (
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>'
        if PLOTLY_AVAILABLE else ""
    )
    log_stem = log_path.stem
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Log Analysis: {log_stem}</title>
  {plotly_cdn}
  <style>
    :root {{
      --bg: #0f1117; --surface: #1a1d27; --border: #2e3347;
      --text: #e2e8f0; --text-muted: #94a3b8;
      --accent: #6366f1; --error: #f87171; --warning: #fbbf24;
      --success: #34d399; --info: #60a5fa;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, monospace; font-size: 14px; line-height: 1.6; }}
    header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 20px 32px; display: flex; justify-content: space-between; align-items: center; }}
    header h1 {{ font-size: 1.4rem; font-weight: 600; color: var(--accent); }}
    header .meta {{ color: var(--text-muted); font-size: 12px; }}
    main {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}
    .summary-banner {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 32px; }}
    .summary-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
    .summary-label {{ display: block; font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
    .summary-value {{ font-size: 1.2rem; font-weight: 600; }}
    section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px 24px; margin-bottom: 20px; }}
    section h2 {{ font-size: 1.05rem; font-weight: 600; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 10px; }}
    section h3 {{ font-size: 0.95rem; font-weight: 600; }}
    .purpose-section {{ border-left: 3px solid var(--accent); }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .data-table th {{ background: rgba(255,255,255,0.05); padding: 8px 12px; text-align: left; font-weight: 600; color: var(--text-muted); text-transform: uppercase; font-size: 11px; letter-spacing: 0.04em; border-bottom: 1px solid var(--border); }}
    .data-table td {{ padding: 7px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: top; word-break: break-word; max-width: 300px; }}
    .data-table tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .error-row td {{ color: var(--error); }}
    .warning-row td {{ color: var(--warning); }}
    .cell-error {{ color: var(--error); font-weight: 600; }}
    .table-scroll {{ overflow-x: auto; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 600; }}
    .badge-red {{ background: rgba(248,113,113,0.15); color: var(--error); }}
    .badge-yellow {{ background: rgba(251,191,36,0.15); color: var(--warning); }}
    .badge-green {{ background: rgba(52,211,153,0.15); color: var(--success); }}
    .badge-blue {{ background: rgba(96,165,250,0.15); color: var(--info); }}
    .alert {{ padding: 10px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; }}
    .alert-error {{ background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: var(--error); }}
    .alert-ok {{ background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: var(--success); }}
    .no-data {{ color: var(--text-muted); font-style: italic; padding: 12px 0; }}
    footer {{ text-align: center; color: var(--text-muted); font-size: 12px; padding: 24px; border-top: 1px solid var(--border); margin-top: 32px; }}
  </style>
</head>
<body>
  <header>
    <h1>Log Analysis Report</h1>
    <div class="meta">
      <div>Source: <code>{log_stem}.jsonl</code></div>
      <div>Generated: {generated_at}</div>
    </div>
  </header>
  <main>
    {''.join(sections_html)}
  </main>
  <footer>Generated by log-analysis skill &middot; Structured Log Protocol v1.0</footer>
</body>
</html>"""

    return html


# ══════════════════════════════════════════════════════════════════════════════
# 5. CLI 진입점
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Analyze AI research experiment logs")
    parser.add_argument("--log-file", required=True, help="Path to jsonl log file")
    parser.add_argument("--output-dir", default="reports", help="Output directory for HTML report")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze_logs] Parsing: {log_path}")
    records, is_slp = parse_jsonl(log_path)
    print(f"[analyze_logs] Records: {len(records)} ({'SLP' if is_slp else 'Heuristic'})")

    if not records:
        print("[analyze_logs] Warning: no records parsed. Empty report will be generated.")

    print("[analyze_logs] Extracting run metadata...")
    run_meta = extract_run_meta(records)

    print("[analyze_logs] Detecting errors...")
    errors = extract_errors(records)
    print(f"[analyze_logs] Errors found: {len(errors)}")

    print("[analyze_logs] Extracting numeric series...")
    series = extract_numeric_series(records)
    print(f"[analyze_logs] Series tags: {len(series)}")

    print("[analyze_logs] Detecting anomalies...")
    anomalies = detect_anomalies(series)
    total_anomalies = sum(len(v) for v in anomalies.values())
    print(f"[analyze_logs] Anomalies: {total_anomalies}")

    print("[analyze_logs] Extracting tensor stats...")
    tensor_stats = extract_tensor_stats(records)

    print("[analyze_logs] Extracting timing info...")
    timing = extract_timing(records)

    purpose = run_meta.get("purpose", "")
    print(f"[analyze_logs] Purpose: {purpose or 'general_monitoring'}")
    purpose_relevant = extract_purpose_relevant(records, purpose)

    print("[analyze_logs] Running MCTD-specific analysis...")
    mctd_analysis = extract_mctd_analysis(records)
    position_trajectory = extract_position_trajectory(records)
    print(f"[analyze_logs] MCTD: {len(mctd_analysis.get('search_events',[]))} search events, "
          f"{len(mctd_analysis.get('loop_outcomes',[]))} outcomes, "
          f"{len(position_trajectory)} position points")

    print("[analyze_logs] Building HTML report...")
    html = build_html_report(
        log_path=log_path,
        records=records,
        is_slp=is_slp,
        run_meta=run_meta,
        errors=errors,
        series=series,
        anomalies=anomalies,
        tensor_stats=tensor_stats,
        timing=timing,
        purpose_relevant=purpose_relevant,
        mctd_analysis=mctd_analysis,
        position_trajectory=position_trajectory,
    )

    output_path = output_dir / f"{log_path.stem}_analysis.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"[analyze_logs] Report saved: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
