from __future__ import annotations

import argparse
import json
import os
from typing import Any

METRIC_TITLES = {
    "agent_success": "Agent Success",
    "hamiltonian_path_success": "Hamiltonian Success",
    "postprocessed_plan_length_rel_error": "Postprocessed RelErr",
}

METRIC_ORDER = [
    "agent_success",
    "hamiltonian_path_success",
    "postprocessed_plan_length_rel_error",
]

BETTER_LABEL = {
    None: "n/a",
    "online": "online",
    "baseline": "fixed",
    "tie": "tie",
}


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _render_metric_rows(scope: dict[str, Any]) -> list[str]:
    rows = [
        "| Metric | Online | Fixed | Delta (O-F) | Better |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for metric_name in METRIC_ORDER:
        metric = scope.get(metric_name, {})
        rows.append(
            "| "
            f"{METRIC_TITLES[metric_name]} | "
            f"{_fmt(metric.get('online_mean'))} | "
            f"{_fmt(metric.get('baseline_mean'))} | "
            f"{_fmt(metric.get('delta_online_minus_baseline'))} | "
            f"{BETTER_LABEL.get(metric.get('better_variant'), 'n/a')} |"
        )
    return rows


def _render_scope_section(title: str, scope: dict[str, Any]) -> list[str]:
    lines = [f"## {title}", ""]
    lines.extend(_render_metric_rows(scope))
    lines.append("")
    return lines


def _render_named_scope_section(
    title: str,
    items: list[dict[str, Any]],
    key_name: str,
) -> list[str]:
    lines = [f"## {title}", ""]
    for item in items:
        if key_name == "task_id":
            heading = f"Task {item[key_name]}"
        elif key_name == "difficulty":
            heading = f"Difficulty {item.get('difficulty_label', item[key_name])}"
        else:
            heading = f"{key_name.capitalize()} {item[key_name]}"
        lines.append(f"### {heading}")
        lines.append("")
        lines.extend(_render_metric_rows(item))
        lines.append("")
    return lines


def _render_task_difficulty_table(items: list[dict[str, Any]]) -> list[str]:
    lines = ["## Task × Difficulty Comparison", ""]
    lines.append(
        "| Task | Difficulty | O Agent | F Agent | Δ Agent | Best Agent | "
        "O Route | F Route | Δ Route | Best Route | "
        "O RelErr | F RelErr | Δ RelErr | Best RelErr |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | --- | "
        "---: | ---: | ---: | --- | "
        "---: | ---: | ---: | --- |"
    )
    for item in items:
        agent = item.get("agent_success", {})
        route = item.get("hamiltonian_path_success", {})
        relerr = item.get("postprocessed_plan_length_rel_error", {})
        lines.append(
            "| "
            f"{item.get('task_id')} | "
            f"{item.get('difficulty_label', item.get('difficulty'))} | "
            f"{_fmt(agent.get('online_mean'))} | "
            f"{_fmt(agent.get('baseline_mean'))} | "
            f"{_fmt(agent.get('delta_online_minus_baseline'))} | "
            f"{BETTER_LABEL.get(agent.get('better_variant'), 'n/a')} | "
            f"{_fmt(route.get('online_mean'))} | "
            f"{_fmt(route.get('baseline_mean'))} | "
            f"{_fmt(route.get('delta_online_minus_baseline'))} | "
            f"{BETTER_LABEL.get(route.get('better_variant'), 'n/a')} | "
            f"{_fmt(relerr.get('online_mean'))} | "
            f"{_fmt(relerr.get('baseline_mean'))} | "
            f"{_fmt(relerr.get('delta_online_minus_baseline'))} | "
            f"{BETTER_LABEL.get(relerr.get('better_variant'), 'n/a')} |"
        )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Render a Hamiltonian benchmark comparison summary JSON as Markdown."
    )
    parser.add_argument("--comparison_summary", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    summary = _load_json(args.comparison_summary)

    lines: list[str] = []
    lines.append("# Hamiltonian Benchmark Comparison")
    lines.append("")
    lines.append(f"- Dataset: `{summary.get('dataset')}`")
    lines.append(f"- Run timestamp: `{summary.get('run_timestamp')}`")
    lines.append(f"- Delta convention: `{summary.get('delta_convention')}`")
    lines.append(
        f"- Online summary: `{summary.get('online_summary')}`"
    )
    lines.append(
        f"- Fixed summary: `{summary.get('baseline_summary')}`"
    )
    lines.append("")

    lines.extend(_render_scope_section("Overall Comparison", summary["overall_comparison"]))
    lines.extend(
        _render_named_scope_section(
            "Difficulty Comparison",
            summary.get("difficulty_comparisons", []),
            "difficulty",
        )
    )
    lines.extend(
        _render_named_scope_section(
            "Task Comparison",
            summary.get("task_comparisons", []),
            "task_id",
        )
    )
    lines.extend(
        _render_task_difficulty_table(
            summary.get("task_difficulty_comparisons", [])
        )
    )

    out_dir = os.path.dirname(args.output_md)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"comparison_markdown: {os.path.abspath(args.output_md)}")


if __name__ == "__main__":
    main()
