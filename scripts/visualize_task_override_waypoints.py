#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logging_utils import make_trajectory_images
from utils.task_override_loader import apply_task_overrides_to_env, resolve_task_override_path


def _repo_root() -> Path:
    return REPO_ROOT


def _make_env(env_id: str, image_size: int):
    import ogbench.locomaze.maze as maze_module

    if "antmaze" in env_id:
        loco_type = "ant"
    elif "pointmaze" in env_id:
        loco_type = "point"
    else:
        raise ValueError(f"Unsupported locomaze env_id: {env_id}")

    maze_type = env_id.split("-")[1]
    return maze_module.make_maze_env(
        loco_type,
        "maze",
        maze_type=maze_type,
        width=image_size,
        height=image_size,
    )


def _load_override_env(override_path: str, image_size: int):
    import yaml

    with open(override_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    env_id = payload.get("env_id")
    if not env_id:
        raise ValueError(f"'env_id' is required in task override file: {override_path}")

    env = _make_env(env_id, image_size=image_size)
    apply_task_overrides_to_env(env, override_path, expected_env_id=env_id, repo_root=str(_repo_root()))
    return env_id, env


def _task_ids_from_arg(task_ids_arg: str | None, num_tasks: int) -> list[int]:
    if not task_ids_arg:
        return list(range(1, num_tasks + 1))
    task_ids = []
    for token in task_ids_arg.split(","):
        token = token.strip()
        if not token:
            continue
        task_ids.append(int(token))
    for task_id in task_ids:
        if not 1 <= task_id <= num_tasks:
            raise ValueError(f"task_id {task_id} is out of range [1, {num_tasks}]")
    return task_ids


def _task_waypoint_groups(task_info: dict, waypoint_group_idx: int | None) -> list[tuple[int | None, np.ndarray]]:
    waypoint_xy_groups = task_info.get("waypoint_xy_groups", None)
    if waypoint_xy_groups:
        normalized_groups = []
        for idx, waypoint_xy_group in enumerate(waypoint_xy_groups):
            waypoint_xy_group = np.asarray(waypoint_xy_group, dtype=np.float32)
            if waypoint_xy_group.size == 0:
                waypoint_xy_group = np.zeros((0, 2), dtype=np.float32)
            else:
                waypoint_xy_group = waypoint_xy_group.reshape(-1, 2)
            normalized_groups.append((idx, waypoint_xy_group))
        if waypoint_group_idx is None:
            return normalized_groups
        if not 0 <= int(waypoint_group_idx) < len(normalized_groups):
            raise ValueError(
                f"waypoint_group_idx {waypoint_group_idx} is out of range [0, {len(normalized_groups) - 1}]"
            )
        return [normalized_groups[int(waypoint_group_idx)]]

    waypoint_xys = np.asarray(task_info.get("waypoint_xys", []), dtype=np.float32)
    if waypoint_xys.size == 0:
        waypoint_xys = np.zeros((0, 2), dtype=np.float32)
    else:
        waypoint_xys = waypoint_xys.reshape(-1, 2)
    return [(task_info.get("active_waypoint_group_idx"), waypoint_xys)]


def _render_task_image(env_id: str, task_info: dict, image_size: int, waypoint_xys: np.ndarray, group_idx: int | None) -> np.ndarray:
    start_xy = np.asarray(task_info["init_xy"], dtype=np.float32)
    goal_xy = np.asarray(task_info["goal_xy"], dtype=np.float32)
    if waypoint_xys.size == 0:
        waypoint_batches: Sequence[np.ndarray] = [np.zeros((0, 2), dtype=np.float32)]
    else:
        waypoint_batches = [waypoint_xys.reshape(-1, 2)]

    rendered = make_trajectory_images(
        env_id,
        {},
        1,
        start=[start_xy],
        goal=[goal_xy],
        plot_end_points=True,
        waypoints=waypoint_batches,
    )[0][:, :, :3]

    img = Image.fromarray(rendered)
    draw = ImageDraw.Draw(img)
    lines = [
        str(task_info["task_name"]),
        f"start: {tuple(round(float(v), 1) for v in start_xy[:2])}",
        f"goal: {tuple(round(float(v), 1) for v in goal_xy[:2])}",
    ]
    if group_idx is not None:
        lines.append(f"group_idx: {group_idx}")
    waypoint_coords = waypoint_batches[0]
    if len(waypoint_coords) == 0:
        lines.append("waypoints: []")
    else:
        lines.append("waypoints:")
        for idx, waypoint in enumerate(waypoint_coords, start=1):
            lines.append(
                f"  wp{idx}: {tuple(round(float(v), 1) for v in waypoint[:2])}"
            )

    text = "\n".join(lines)
    try:
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text, spacing=2)
        text_w = right - left
        text_h = bottom - top
    except Exception:
        text_w = max(len(line) for line in lines) * 7
        text_h = len(lines) * 12
    box = (4, 4, min(img.width - 4, 12 + text_w), min(img.height - 4, 12 + text_h))
    draw.rectangle(box, fill=(0, 0, 0))
    draw.multiline_text((8, 8), text, fill=(255, 255, 255), spacing=2)
    return np.asarray(img)


def _build_contact_sheet(images: list[np.ndarray], labels: list[str], cols: int = 2) -> Image.Image:
    if not images:
        raise ValueError("No images to compose")

    tile_h, tile_w = images[0].shape[:2]
    label_h = 22
    pad = 10
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    sheet_w = cols * tile_w + (cols + 1) * pad
    sheet_h = rows * (tile_h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (img_np, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        x0 = pad + col * tile_w
        y0 = pad + row * (tile_h + label_h)
        sheet.paste(Image.fromarray(img_np), (x0, y0))
        draw.text((x0, y0 + tile_h + 4), label, fill=(0, 0, 0))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Render OGBench task overrides with waypoint markers.")
    parser.add_argument(
        "--override-path",
        default="configurations/task_overrides/antmaze_giant_waypoints.yaml",
        help="Repo-relative or absolute path to task override YAML/JSON",
    )
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task ids to render. Default: all tasks.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug/task_override_waypoints",
        help="Directory to save rendered PNGs",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=200,
        help="Maze render width/height passed to OGBench",
    )
    parser.add_argument(
        "--waypoint-group-idx",
        type=int,
        default=None,
        help="Optional 0-based waypoint group index to render. Default: render all groups for each task.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    override_path = resolve_task_override_path(args.override_path, repo_root=str(repo_root))
    if override_path is None:
        raise ValueError("--override-path must be provided")

    env_id, env = _load_override_env(override_path, image_size=args.image_size)
    try:
        task_ids = _task_ids_from_arg(args.task_ids, env.num_tasks)
        override_stem = Path(override_path).stem
        output_dir = repo_root / args.output_dir / override_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        rendered_images: list[np.ndarray] = []
        labels: list[str] = []
        for task_id in task_ids:
            task_info = env.task_infos[task_id - 1]
            for group_idx, waypoint_xys in _task_waypoint_groups(task_info, args.waypoint_group_idx):
                img_np = _render_task_image(
                    env_id,
                    task_info,
                    image_size=args.image_size,
                    waypoint_xys=waypoint_xys,
                    group_idx=group_idx,
                )
                rendered_images.append(img_np)
                group_label = "active" if group_idx is None else f"g{int(group_idx):02d}"
                labels.append(f"T{task_id:02d} {group_label} wps={len(waypoint_xys)}")
                stem = f"task_{task_id:02d}"
                if group_idx is not None:
                    stem += f"_group_{int(group_idx):02d}"
                out_path = output_dir / f"{stem}.png"
                Image.fromarray(img_np).save(out_path)
                print(f"saved {out_path}")

        if len(rendered_images) > 1 and len(rendered_images) <= 64:
            overview = _build_contact_sheet(rendered_images, labels)
            overview_path = output_dir / "overview.png"
            overview.save(overview_path)
            print(f"saved {overview_path}")
        elif len(rendered_images) > 64:
            overview_path = output_dir / "overview.png"
            if overview_path.exists():
                overview_path.unlink()
                print(f"removed stale {overview_path}")
            print("skipped overview.png because too many images were rendered")
        else:
            overview_path = output_dir / "overview.png"
            if overview_path.exists():
                overview_path.unlink()
                print(f"removed stale {overview_path}")
            print("skipped overview.png because only one task image was rendered")
    finally:
        env.close()


if __name__ == "__main__":
    main()
