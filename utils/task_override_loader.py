from __future__ import annotations

from copy import deepcopy
import json
import os
from typing import Any, Optional

import numpy as np
import yaml


def _default_repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_task_override_path(override_path: Optional[str], repo_root: Optional[str] = None) -> Optional[str]:
    if override_path in (None, ""):
        return None
    repo_root = repo_root or _default_repo_root()
    expanded = os.path.expanduser(str(override_path))
    if not os.path.isabs(expanded):
        expanded = os.path.join(repo_root, expanded)
    resolved = os.path.realpath(expanded)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Task override file not found: {resolved}")
    return resolved


def load_task_override_payload(
    override_path: Optional[str],
    repo_root: Optional[str] = None,
) -> tuple[Optional[str], dict[str, Any]]:
    resolved_path = resolve_task_override_path(override_path, repo_root=repo_root)
    if resolved_path is None:
        return None, {}
    return resolved_path, _load_override_payload(resolved_path)


def _load_override_payload(resolved_path: str) -> dict[str, Any]:
    _, ext = os.path.splitext(resolved_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        if ext.lower() == ".json":
            payload = json.load(f)
        else:
            payload = yaml.safe_load(f)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Task override payload must be a mapping: {resolved_path}")
    return payload


def _normalize_ij_point(value: Any, field_name: str) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")
    if len(value) != 2:
        raise ValueError(f"{field_name} must have exactly 2 entries, got {value!r}")
    return int(value[0]), int(value[1])


def _normalize_waypoint_ij_group(value: Any, field_name: str = "waypoint_ij_group") -> list[tuple[int, int]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of [i, j] pairs")
    return [_normalize_ij_point(point, f"{field_name}[]") for point in value]


def _normalize_waypoint_ij_groups(value: Any) -> list[list[tuple[int, int]]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("waypoint_ij_groups must be a list of waypoint groups")
    return [
        _normalize_waypoint_ij_group(group, "waypoint_ij_groups[]")
        for group in value
    ]


def _normalize_waypoint_group_index(value: Any, *, allow_none: bool = True) -> Optional[int]:
    if value is None:
        if allow_none:
            return None
        raise ValueError("active_waypoint_group_idx cannot be null")
    return int(value)


def _waypoint_xy_groups_from_ij_groups(env, waypoint_ij_groups: list[list[tuple[int, int]]]) -> list[list[tuple[float, float]]]:
    return [
        [tuple(float(v) for v in env.ij_to_xy(waypoint_ij)) for waypoint_ij in waypoint_ij_group]
        for waypoint_ij_group in waypoint_ij_groups
    ]


def _select_active_waypoint_group_index(
    waypoint_ij_groups: list[list[tuple[int, int]]],
    *,
    task_info: dict[str, Any],
    override: dict[str, Any],
    global_active_waypoint_group_idx: Optional[int],
    waypoint_group_idx_override: Optional[int],
) -> Optional[int]:
    if not waypoint_ij_groups:
        return None
    if waypoint_group_idx_override is not None:
        selected_idx = int(waypoint_group_idx_override)
    elif "active_waypoint_group_idx" in override:
        selected_idx = _normalize_waypoint_group_index(override["active_waypoint_group_idx"], allow_none=False)
    elif global_active_waypoint_group_idx is not None:
        selected_idx = int(global_active_waypoint_group_idx)
    else:
        selected_idx = _normalize_waypoint_group_index(task_info.get("active_waypoint_group_idx", 0), allow_none=False)

    if not 0 <= int(selected_idx) < len(waypoint_ij_groups):
        raise ValueError(
            f"active_waypoint_group_idx {selected_idx} is out of range [0, {len(waypoint_ij_groups) - 1}]"
        )
    return int(selected_idx)


def _task_info_xy_from_ij(env, task_info: dict[str, Any], ij_key: str, xy_key: str) -> None:
    if ij_key not in task_info:
        return
    task_info[xy_key] = tuple(float(v) for v in env.ij_to_xy(task_info[ij_key]))


def _normalize_task_info(task_info: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(task_info)
    for key in ("init_ij", "goal_ij"):
        if key in normalized:
            normalized[key] = list(normalized[key])
    for key in ("init_xy", "goal_xy"):
        if key in normalized:
            normalized[key] = [float(v) for v in normalized[key]]
    if "waypoint_ij_group" in normalized:
        normalized["waypoint_ij_group"] = [list(point) for point in normalized["waypoint_ij_group"]]
    if "waypoint_ij_groups" in normalized:
        normalized["waypoint_ij_groups"] = [
            [list(point) for point in waypoint_ij_group]
            for waypoint_ij_group in normalized["waypoint_ij_groups"]
        ]
    if "waypoint_xys" in normalized:
        normalized["waypoint_xys"] = [[float(v) for v in point] for point in normalized["waypoint_xys"]]
    if "waypoint_xy_groups" in normalized:
        normalized["waypoint_xy_groups"] = [
            [[float(v) for v in point] for point in waypoint_xy_group]
            for waypoint_xy_group in normalized["waypoint_xy_groups"]
        ]
    if "active_waypoint_group_idx" in normalized and normalized["active_waypoint_group_idx"] is not None:
        normalized["active_waypoint_group_idx"] = int(normalized["active_waypoint_group_idx"])
    return normalized


def apply_task_overrides_to_env(
    env,
    override_path: Optional[str],
    *,
    expected_env_id: Optional[str] = None,
    repo_root: Optional[str] = None,
    waypoint_group_idx_override: Optional[int] = None,
) -> Optional[str]:
    resolved_path = resolve_task_override_path(override_path, repo_root=repo_root)
    if resolved_path is None:
        return None

    payload = _load_override_payload(resolved_path)
    payload_env_id = payload.get("env_id")
    global_active_waypoint_group_idx = _normalize_waypoint_group_index(
        payload.get("active_waypoint_group_idx", None),
        allow_none=True,
    )
    if payload_env_id and expected_env_id and str(payload_env_id) != str(expected_env_id):
        raise ValueError(
            f"Task override env_id mismatch: expected {expected_env_id}, got {payload_env_id} in {resolved_path}"
        )

    overrides = payload.get("tasks", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError(f"'tasks' must be a mapping of task_id -> overrides: {resolved_path}")

    base_task_infos = deepcopy(getattr(env, "task_infos", None))
    if not isinstance(base_task_infos, list) or not base_task_infos:
        raise ValueError("Env does not expose a non-empty task_infos list")

    updated_task_infos = deepcopy(base_task_infos)
    for raw_task_id, override in overrides.items():
        task_id = int(raw_task_id)
        if not 1 <= task_id <= len(updated_task_infos):
            raise ValueError(
                f"Task override id {task_id} is out of range [1, {len(updated_task_infos)}] for {resolved_path}"
            )
        if override is None:
            override = {}
        if not isinstance(override, dict):
            raise ValueError(f"Task override for task {task_id} must be a mapping")

        task_info = deepcopy(updated_task_infos[task_id - 1])
        if "task_name" in override and override["task_name"] is not None:
            task_info["task_name"] = str(override["task_name"])
        if "init_ij" in override:
            task_info["init_ij"] = _normalize_ij_point(override["init_ij"], "init_ij")
        if "goal_ij" in override:
            task_info["goal_ij"] = _normalize_ij_point(override["goal_ij"], "goal_ij")

        if "waypoint_ij_groups" in override:
            waypoint_ij_groups = _normalize_waypoint_ij_groups(override["waypoint_ij_groups"])
        elif "waypoint_ijs" in override:
            waypoint_ij_groups = [_normalize_waypoint_ij_group(override["waypoint_ijs"], "waypoint_ijs")]
        elif "waypoint_ij_groups" in task_info:
            waypoint_ij_groups = _normalize_waypoint_ij_groups(task_info["waypoint_ij_groups"])
        elif "waypoint_ijs" in task_info:
            waypoint_ij_groups = [_normalize_waypoint_ij_group(task_info["waypoint_ijs"], "waypoint_ijs")]
        else:
            waypoint_ij_groups = []

        _task_info_xy_from_ij(env, task_info, "init_ij", "init_xy")
        _task_info_xy_from_ij(env, task_info, "goal_ij", "goal_xy")

        if waypoint_ij_groups:
            task_info["waypoint_ij_groups"] = waypoint_ij_groups
            task_info["waypoint_xy_groups"] = _waypoint_xy_groups_from_ij_groups(env, waypoint_ij_groups)
            active_group_idx = _select_active_waypoint_group_index(
                waypoint_ij_groups,
                task_info=task_info,
                override=override,
                global_active_waypoint_group_idx=global_active_waypoint_group_idx,
                waypoint_group_idx_override=waypoint_group_idx_override,
            )
            task_info["active_waypoint_group_idx"] = active_group_idx
            task_info["waypoint_ij_group"] = list(waypoint_ij_groups[int(active_group_idx)])
            task_info["waypoint_xys"] = list(task_info["waypoint_xy_groups"][int(active_group_idx)])
            task_info["waypoint_ijs"] = [tuple(point) for point in task_info["waypoint_ij_group"]]
        else:
            task_info["waypoint_ij_groups"] = []
            task_info["waypoint_xy_groups"] = []
            task_info["active_waypoint_group_idx"] = None
            task_info["waypoint_ij_group"] = []
            task_info["waypoint_xys"] = []
            task_info["waypoint_ijs"] = []

        updated_task_infos[task_id - 1] = task_info

    env.task_infos = updated_task_infos
    env.num_tasks = len(updated_task_infos)
    cur_task_id = getattr(env, "cur_task_id", None)
    if cur_task_id is not None and 1 <= int(cur_task_id) <= env.num_tasks:
        env.cur_task_info = env.task_infos[int(cur_task_id) - 1]
    return resolved_path


def inject_task_metadata_into_reset_info(ob: Any, info: Optional[dict[str, Any]], task_info: Optional[dict[str, Any]]) -> dict[str, Any]:
    info_out = dict(info or {})
    ob_xy = np.asarray(ob, dtype=np.float32).reshape(-1)[:2]
    info_out["start_xy"] = ob_xy.copy()

    goal_ob = info_out.get("goal")
    if goal_ob is not None:
        goal_xy = np.asarray(goal_ob, dtype=np.float32).reshape(-1)[:2]
    elif task_info is not None and "goal_xy" in task_info:
        goal_xy = np.asarray(task_info["goal_xy"], dtype=np.float32).reshape(-1)[:2]
    else:
        goal_xy = np.zeros((2,), dtype=np.float32)
    info_out["goal_xy"] = goal_xy.copy()

    if task_info is None:
        info_out["waypoints_xy"] = np.zeros((0, 2), dtype=np.float32)
        info_out["waypoint_xy_groups"] = []
        info_out["active_waypoint_group_idx"] = None
        return info_out

    waypoint_xys = task_info.get("waypoint_xys", [])
    waypoints_xy = np.asarray(waypoint_xys, dtype=np.float32)
    if waypoints_xy.size == 0:
        waypoints_xy = np.zeros((0, 2), dtype=np.float32)
    else:
        waypoints_xy = waypoints_xy.reshape(-1, 2)
    info_out["waypoints_xy"] = waypoints_xy.copy()
    waypoint_xy_groups = []
    for waypoint_xy_group in task_info.get("waypoint_xy_groups", []):
        waypoint_xy_group_np = np.asarray(waypoint_xy_group, dtype=np.float32)
        if waypoint_xy_group_np.size == 0:
            waypoint_xy_groups.append(np.zeros((0, 2), dtype=np.float32))
        else:
            waypoint_xy_groups.append(waypoint_xy_group_np.reshape(-1, 2).copy())
    info_out["waypoint_xy_groups"] = waypoint_xy_groups
    info_out["active_waypoint_group_idx"] = task_info.get("active_waypoint_group_idx")
    if "task_name" in task_info:
        info_out["task_name"] = task_info["task_name"]
    info_out["task_info"] = _normalize_task_info(task_info)
    return info_out
