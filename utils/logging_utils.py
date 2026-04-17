from typing import Optional
import wandb
import numpy as np
import torch

import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
from tqdm import trange, tqdm
import matplotlib.animation as animation
from pathlib import Path

plt.set_loglevel("warning")

from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
)
from algorithms.common.metrics import (
    FrechetVideoDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)

OGBENCH_ENVS = [
    "pointmaze-medium-v0",
    "pointmaze-large-v0",
    "pointmaze-giant-v0",
    "pointmaze-teleport-v0",
    "antmaze-medium-v0",
    "antmaze-large-v0",
    "antmaze-giant-v0",
    "antmaze-teleport-v0",
]

GUIDANCE_COMPONENT_SPECS = {
    "score": {
        "color": "yellow",
        "label_xt": "Prior score (x_t)",
        "label_x0": "Prior score (x̂_0)",
    },
    "anchor": {
        "color": "deepskyblue",
        "label_xt": "Anchor grad (x_t)",
        "label_x0": "Anchor grad (x̂_0)",
    },
    "goal": {
        "color": "lime",
        "label_xt": "Goal grad (x_t)",
        "label_x0": "Goal grad (x̂_0)",
    },
    "rdf": {
        "color": "magenta",
        "label_xt": "RDF grad (x_t)",
        "label_x0": "RDF grad (x̂_0)",
    },
    "particle": {
        "color": "cyan",
        "label_xt": "Particle grad (x_t)",
        "label_x0": "Particle grad (x̂_0)",
    },
}


def _normalize_vector_field(U: np.ndarray, V: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Normalize each 2D vector to unit length while keeping zero vectors zero."""
    mag = np.sqrt(U ** 2 + V ** 2)
    safe_mag = np.where(mag > eps, mag, 1.0)
    U_n = np.where(mag > eps, U / safe_mag, 0.0)
    V_n = np.where(mag > eps, V / safe_mag, 0.0)
    return U_n, V_n


# FIXME: clean up & check this util
def log_video(
    observation_hat,
    observation_gt=None,
    step=0,
    namespace="train",
    prefix="video",
    context_frames=0,
    color=(255, 0, 0),
    logger=None,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    """
    if not logger:
        logger = wandb
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hat)
    observation_hat[:context_frames] = observation_gt[:context_frames]
    # Add red border of 1 pixel width to the context frames
    for i, c in enumerate(color):
        c = c / 255.0
        observation_hat[:context_frames, :, i, [0, -1], :] = c
        observation_hat[:context_frames, :, i, :, [0, -1]] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    video = torch.cat([observation_hat, observation_gt], -1).detach().cpu().numpy()
    video = np.transpose(np.clip(video, a_min=0.0, a_max=1.0) * 255, (1, 0, 2, 3, 4)).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)
    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    for i in range(n_samples):
        logger.log(
            {
                f"{namespace}/{prefix}_{i}": wandb.Video(video[i], fps=24),
                f"trainer/global_step": step,
            }
        )


def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
    fvd_model: Optional[FrechetVideoDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :param fvd_model: a FrechetVideoDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    if frame < 9:
        fvd_model = None  # FVD requires at least 9 frames

    if fvd_model is not None:
        output_dict["fvd"] = fvd_model.compute(
            torch.clamp(observation_hat, -1.0, 1.0),
            torch.clamp(observation_gt, -1.0, 1.0),
        )

    # reshape to (frame * batch, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0)
    output_dict["ssim"] = structural_similarity_index_measure(observation_hat, observation_gt, data_range=2.0)
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


def is_grid_env(env_id):
    return "maze2d" in env_id or "diagonal2d" in env_id or "pointmaze" in env_id or "antmaze" in env_id


def get_maze_grid(env_id):
    # import gym
    # maze_string = gym.make(env_id).str_maze_spec
    if "medium" in env_id:
        maze_string = "########\\#OO#OOO#\\#OOOO#O#\\###O#OO#\\##OOOO##\\#OO#O#O#\\#OO#OOO#\\########"
    elif "large" in env_id:
        maze_string = "#########\\#OOOOO#O#\\#O#O#OOO#\\#O#O#####\\#OOO#OOO#\\###O###O#\\#OOOOOOO#\\#O###O###\\#OOO#OOO#\\#O#O#O#O#\\#OOOOO#O#\\#########"
    elif "giant" in env_id:
        maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    elif "teleport" in env_id:
        maze_string = "#########\\#O##OOOO#\\#OOOO##O#\\#O##O##O#\\#OO#OOOO#\\#OO######\\##OOOOOO#\\#O#O###O#\\##OOOOOO#\\#OOO#####\\#O#OOOOO#\\#########"
    #if "large" in env_id:
    #    maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    #if "teleport" in env_id:
    #    maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    #if "medium" in env_id:
    #    maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    #if "umaze" in env_id:
    #    maze_string = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    #if "giant" in env_id:
    #    maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def get_random_start_goal(env_id, batch_size):
    maze_grid = get_maze_grid(env_id)
    s2i = {"O": 0, "#": 1, "G": 2}
    maze_grid = [[s2i[s] for s in r] for r in maze_grid]
    maze_grid = np.array(maze_grid)
    x, y = np.nonzero(maze_grid == 0)
    indices = np.random.randint(len(x), size=batch_size)
    start = np.stack([x[indices], y[indices]], -1) + 1
    x, y = np.nonzero(maze_grid == 2)
    goal = np.concatenate([x, y], -1)
    goal = np.tile(goal[None, :], (batch_size, 1)) + 1
    return start, goal


def plot_maze_layout(ax, maze_grid):
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def _select_waypoints_for_batch(waypoints, batch_idx):
    if waypoints is None:
        return None

    if isinstance(waypoints, np.ndarray):
        if waypoints.size == 0:
            return None
        if waypoints.ndim == 2:
            selected = waypoints
        elif waypoints.ndim == 3:
            if batch_idx >= waypoints.shape[0]:
                return None
            selected = waypoints[batch_idx]
        else:
            return None
    elif isinstance(waypoints, (list, tuple)):
        if len(waypoints) == 0:
            return None
        first = np.asarray(waypoints[0])
        if first.ndim >= 2:
            if batch_idx >= len(waypoints):
                return None
            selected = np.asarray(waypoints[batch_idx])
        else:
            selected = np.asarray(waypoints)
    else:
        selected = np.asarray(waypoints)

    if selected.size == 0:
        return None
    if selected.ndim == 1:
        selected = selected[None, :]
    return np.asarray(selected, dtype=np.float32)


def plot_start_goal(ax, start_goal: None, waypoints=None):
    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    start_x, start_y = start_goal[0][:2]
    start_outer_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1][:2]
    goal_outer_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)

    if waypoints is not None and len(waypoints) > 0:
        for idx, waypoint in enumerate(np.asarray(waypoints)):
            waypoint_x, waypoint_y = waypoint[:2]
            ax.scatter(
                waypoint_x,
                waypoint_y,
                marker="D",
                s=110,
                facecolors="white",
                edgecolors="seagreen",
                linewidths=1.5,
                zorder=14,
                label="Waypoint" if idx == 0 else "_nolegend_",
            )
            ax.scatter(
                waypoint_x,
                waypoint_y,
                marker="o",
                s=20,
                c="seagreen",
                zorder=15,
                label="_nolegend_",
            )


def _maybe_add_batch_dim(points):
    if points is None:
        return None
    points = np.asarray(points)
    if points.ndim == 2:
        return points[:, None, :]
    return points


def _to_plot_coords(env_id, points):
    points = np.asarray(points)
    if env_id in OGBENCH_ENVS:
        return points / 4 + 1
    return points


def _sample_frame_indices(num_frames, max_frames):
    if num_frames <= 0:
        return np.array([], dtype=int)
    if max_frames is None or max_frames <= 0 or num_frames <= max_frames:
        return np.arange(num_frames, dtype=int)
    return np.unique(np.linspace(0, num_frames - 1, num=max_frames, dtype=int))


def _sample_history_until(frame_idx, path_stride):
    if frame_idx < 0:
        return np.array([], dtype=int)
    stride = max(int(path_stride or 1), 1)
    hist_idx = np.arange(0, frame_idx + 1, stride, dtype=int)
    if hist_idx.size == 0 or hist_idx[-1] != frame_idx:
        hist_idx = np.append(hist_idx, frame_idx)
    return hist_idx


def _parse_trajectory_plot_inputs(trajectory):
    if isinstance(trajectory, dict):
        plot_data = dict(trajectory)
    else:
        plot_data = {"plan": trajectory}

    for key in [
        "plan",
        "node_path",
        "target_node_path",
        "best_node_target",
        "guidance_targets",
        "goal_grad_vectors",
        "prior_grad_vectors",
        "pred_x_start_pos",
        "pred_x_start_guidance_grad",
        "pred_x_start_prior_grad",
        "rollout_agent_history",
        "rollout_current_agent",
        "rollout_current_subgoal",
        "postprocessed_plan",
        "expanded_node_pos",
    ]:
        if key in plot_data and plot_data[key] is not None:
            plot_data[key] = np.asarray(plot_data[key])

    for key in [
        "guidance_component_vectors",
        "pred_x_start_guidance_component_vectors",
    ]:
        if key in plot_data and plot_data[key] is not None:
            plot_data[key] = {
                comp: np.asarray(vecs)
                for comp, vecs in plot_data[key].items()
                if vecs is not None
            }

    plot_data["plan"] = _maybe_add_batch_dim(plot_data.get("plan"))
    plot_data["rollout_agent_history"] = _maybe_add_batch_dim(plot_data.get("rollout_agent_history"))
    plot_data["rollout_current_agent"] = _maybe_add_batch_dim(plot_data.get("rollout_current_agent"))
    plot_data["rollout_current_subgoal"] = _maybe_add_batch_dim(plot_data.get("rollout_current_subgoal"))
    plot_data["postprocessed_plan"] = _maybe_add_batch_dim(plot_data.get("postprocessed_plan"))
    return plot_data


def _render_trajectory_plot(fig, ax, env_id, plot_data, batch_idx, start, goal, plot_end_points=True, waypoints=None):
    plan_trajectory = plot_data.get("plan")
    node_trajectory = plot_data.get("node_path")
    target_node_trajectory = plot_data.get("target_node_path")
    is_forward_tree = plot_data.get("is_forward_tree", True)
    best_node_target = plot_data.get("best_node_target")
    hilp_heatmap = plot_data.get("hilp_heatmap")
    hilp_grad_field = plot_data.get("hilp_grad_field")
    guidance_targets = plot_data.get("guidance_targets")
    goal_grad_vectors = plot_data.get("goal_grad_vectors")
    prior_grad_vectors = plot_data.get("prior_grad_vectors")
    pred_x_start_pos = plot_data.get("pred_x_start_pos")
    pred_x_start_guidance_grad = plot_data.get("pred_x_start_guidance_grad")
    pred_x_start_prior_grad = plot_data.get("pred_x_start_prior_grad")
    guidance_component_vectors = plot_data.get("guidance_component_vectors")
    pred_x_start_guidance_component_vectors = plot_data.get("pred_x_start_guidance_component_vectors")
    rollout_agent_history = plot_data.get("rollout_agent_history")
    rollout_current_agent = plot_data.get("rollout_current_agent")
    rollout_current_subgoal = plot_data.get("rollout_current_subgoal")
    postprocessed_plan = plot_data.get("postprocessed_plan")
    unc_plans_list = plot_data.get("unc_plans_list")          # list of (T, pos_dim) arrays or None
    unc_guidance_targets = plot_data.get("unc_guidance_targets")  # (G*K, pos_dim) or None
    unc_scale_colors = plot_data.get("unc_scale_colors")          # (G*K, 3) RGB per sample or None
    unc_guidance_legend = plot_data.get("unc_guidance_legend")    # list of (label, rgb) per scale or None
    unc_cluster_labels = plot_data.get("unc_cluster_labels")      # (G*K,) int cluster indices or None
    tail_dot_color = plot_data.get("tail_dot_color", (0.2, 0.85, 0.2))  # RGB for x_t/pred_x0 dots

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None
    plot_maze_layout(ax, maze_grid)

    if hilp_heatmap is not None:
        X_w = hilp_heatmap["X"]
        Y_w = hilp_heatmap["Y"]
        vals = hilp_heatmap["values"]
        if env_id in OGBENCH_ENVS:
            X_p = X_w / 4 + 1
            Y_p = Y_w / 4 + 1
        else:
            X_p, Y_p = X_w, Y_w
        vmin, vmax = float(vals.min()), float(vals.max())
        mesh = ax.pcolormesh(X_p, Y_p, vals, shading="auto", cmap="viridis", alpha=0.5,
                             zorder=1, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, label="HILP V(s,g)", fraction=0.03, pad=0.02)

    if plan_trajectory is not None:
        plan_xy = _to_plot_coords(env_id, plan_trajectory[:, batch_idx, :2])
        ax.scatter(plan_xy[:, 0], plan_xy[:, 1], c=np.arange(len(plan_xy)), cmap="Reds",
                   alpha=0.8, label="Plan", s=50)

    if postprocessed_plan is not None and len(postprocessed_plan) > 0:
        pp_xy = _to_plot_coords(env_id, postprocessed_plan[:, batch_idx, :2])
        ax.scatter(pp_xy[:, 0], pp_xy[:, 1], c=np.arange(len(pp_xy)), cmap="Reds",
                   alpha=0.7, label="PP Plan", s=30, zorder=6)

    if node_trajectory is not None and len(node_trajectory) > 0:
        node_xy = _to_plot_coords(env_id, node_trajectory[:, :2])
        ax.plot(node_xy[:, 0], node_xy[:, 1], "b-", linewidth=2, alpha=0.6, label="Expanding Path")
        ax.scatter(node_xy[:, 0], node_xy[:, 1], c="blue", marker="o", s=100, alpha=0.7,
                   edgecolors="darkblue", linewidth=2, zorder=5)

    if unc_plans_list is not None and len(unc_plans_list) > 0:
        # unc_plans_list: list of G*K arrays (T, pos_dim) — uncertainty fast-sampled trajectories.
        # Each is rendered as a red-gradient scatter (same style as the main plan).
        # unc_guidance_targets: (G*K, pos_dim) — endpoint positions.
        # unc_scale_colors: (G*K, 3) — per-sample RGB color from plasma cmap, keyed by guidance scale.
        for j, traj_j in enumerate(unc_plans_list):
            if traj_j is None or len(traj_j) == 0:
                continue
            traj_xy = _to_plot_coords(env_id, traj_j[:, :2])
            ax.scatter(traj_xy[:, 0], traj_xy[:, 1], c=np.arange(len(traj_xy)), cmap="Reds",
                       alpha=0.5, s=20, zorder=4)

        if unc_guidance_targets is not None and len(unc_guidance_targets) > 0:
            _unc_fixed_alpha = 0.40
            # tab10 palette for cluster border dots (up to 10 clusters, then wraps).
            _tab10 = [
                (0.122, 0.467, 0.706),  # blue
                (1.000, 0.498, 0.055),  # orange
                (0.173, 0.627, 0.173),  # green
                (0.839, 0.153, 0.157),  # red
                (0.580, 0.404, 0.741),  # purple
                (0.549, 0.337, 0.294),  # brown
                (0.890, 0.467, 0.761),  # pink
                (0.498, 0.498, 0.498),  # gray
                (0.737, 0.741, 0.133),  # olive
                (0.090, 0.745, 0.812),  # cyan
            ]
            for j, ep in enumerate(unc_guidance_targets):
                ep_xy = _to_plot_coords(env_id, np.asarray(ep)[:2])
                # Draw cluster-border dot first (larger, behind the main dot).
                # Border size ~242 gives 2x the border width vs main dot size 80
                # (border_width ∝ sqrt(s_border) - sqrt(s_main)).
                if unc_cluster_labels is not None and j < len(unc_cluster_labels):
                    _cl_idx = int(unc_cluster_labels[j]) % len(_tab10)
                    _border_color = _tab10[_cl_idx]
                    ax.scatter(ep_xy[0], ep_xy[1], color=_border_color, marker="o", s=242,
                               zorder=8, edgecolors="none", linewidth=0, alpha=_unc_fixed_alpha,
                               label="_nolegend_")
                dot_color = tuple(unc_scale_colors[j]) if unc_scale_colors is not None and j < len(unc_scale_colors) else (0.2, 0.85, 0.2)
                ax.scatter(ep_xy[0], ep_xy[1], color=dot_color, marker="o", s=80, zorder=9,
                           edgecolors="none", linewidth=0, alpha=_unc_fixed_alpha,
                           label="_nolegend_")

            # Add one proxy legend entry per unique guidance scale.
            if unc_guidance_legend is not None:
                import matplotlib.lines as mlines
                proxy_handles = []
                for label, leg_color in unc_guidance_legend:
                    proxy = mlines.Line2D([], [], color=leg_color, marker="o", linestyle="None",
                                         markersize=7, markeredgecolor="none",
                                         markeredgewidth=0, alpha=_unc_fixed_alpha, label=label)
                    proxy_handles.append(proxy)
                unc_leg = ax.legend(handles=proxy_handles, loc="lower right", framealpha=0.6,
                                    fontsize=4.5, title="unc gs", title_fontsize=4.5)
                ax.add_artist(unc_leg)  # pin so the main ax.legend() call below doesn't overwrite it

    if target_node_trajectory is not None and len(target_node_trajectory) > 0:
        tgt_xy = _to_plot_coords(env_id, target_node_trajectory[:, :2])
        ax.plot(tgt_xy[:, 0], tgt_xy[:, 1], color="orange", linestyle="-", linewidth=2,
                alpha=0.6, label="Target Path")
        ax.scatter(tgt_xy[:, 0], tgt_xy[:, 1], c="orange", marker="o", s=100, alpha=0.7,
                   edgecolors="darkorange", linewidth=2, zorder=5)

    if hilp_grad_field is not None:
        X_w = hilp_grad_field["x_grid"]
        Y_w = hilp_grad_field["y_grid"]
        hilp_grads = hilp_grad_field["hilp_grads"]
        far_mask_grid = hilp_grad_field.get("far_mask_grid", None)
        if env_id in OGBENCH_ENVS:
            X_p = X_w / 4 + 1
            Y_p = Y_w / 4 + 1
        else:
            X_p, Y_p = X_w, Y_w
        U = hilp_grads[:, :, 0]
        V = hilp_grads[:, :, 1]
        U_n, V_n = _normalize_vector_field(U, V)

        if far_mask_grid is None:
            far_mask_grid = np.zeros_like(U_n, dtype=bool)

        flat_hilp_mask = (~far_mask_grid).reshape(-1)
        flat_far_mask = far_mask_grid.reshape(-1)
        flat_colors = np.empty((U_n.size, 4), dtype=float)
        flat_colors[flat_hilp_mask] = to_rgba("crimson", alpha=0.6)
        flat_colors[flat_far_mask] = to_rgba("steelblue", alpha=0.6)

        ax.quiver(
            X_p.reshape(-1), Y_p.reshape(-1),
            U_n.reshape(-1), V_n.reshape(-1),
            color=flat_colors,
            angles="xy",
            scale_units="xy",
            scale=2.0,
            pivot="mid",
            width=0.004,
            zorder=3,
        )

        if flat_hilp_mask.any():
            ax.scatter([], [], c="crimson", marker=r"$\rightarrow$", s=120, alpha=0.6, label="HILP grad")
        if flat_far_mask.any():
            ax.scatter([], [], c="steelblue", marker=r"$\rightarrow$", s=120, alpha=0.6, label="RMSE grad (far)")

    if plot_end_points:
        start_goal = (_to_plot_coords(env_id, np.asarray(start[batch_idx])[:2]),
                      _to_plot_coords(env_id, np.asarray(goal[batch_idx])[:2]))
        batch_waypoints = _select_waypoints_for_batch(waypoints, batch_idx)
        if batch_waypoints is not None:
            batch_waypoints = _to_plot_coords(env_id, batch_waypoints[:, :2])
        plot_start_goal(ax, start_goal, batch_waypoints)

    if best_node_target is not None:
        pos = _to_plot_coords(env_id, np.asarray(best_node_target).flatten()[:2])
        ax.scatter(pos[0], pos[1], c="green", marker="*", s=300, zorder=10,
                   edgecolors="darkgreen", linewidth=1.5, label="Target")

    has_guidance_targets = guidance_targets is not None and len(guidance_targets) > 0
    if has_guidance_targets:
        for idx, gpos in enumerate(guidance_targets):
            gxy = _to_plot_coords(env_id, np.asarray(gpos)[:2])
            ax.scatter(gxy[0], gxy[1], color=tail_dot_color, marker="o", s=120, zorder=9,
                       edgecolors="none", linewidth=0, alpha=0.9,
                       label="x_t tail" if idx == 0 else "_nolegend_")

    has_component_guidance = (
        has_guidance_targets
        and guidance_component_vectors is not None
        and any(len(vecs) > 0 for vecs in guidance_component_vectors.values())
    )
    if has_component_guidance:
        component_legend_added = set()
        for comp_name in ["anchor", "goal", "rdf", "particle"]:
            vecs = guidance_component_vectors.get(comp_name)
            if vecs is None or len(vecs) == 0:
                continue
            spec = GUIDANCE_COMPONENT_SPECS.get(comp_name, {})
            color = spec.get("color", "lime")
            label_xt = spec.get("label_xt", f"{comp_name} grad (x_t)")
            for idx, (gpos, gvec) in enumerate(zip(guidance_targets, vecs)):
                gxy = _to_plot_coords(env_id, np.asarray(gpos)[:2])
                if env_id in OGBENCH_ENVS:
                    gdx, gdy = float(gvec[0]) / 4, float(gvec[1]) / 4
                else:
                    gdx, gdy = float(gvec[0]), float(gvec[1])
                ax.quiver(
                    gxy[0], gxy[1], gdx, gdy,
                    color=color, angles="xy", scale_units="xy",
                    scale=1.0, width=0.006, zorder=12,
                    label=label_xt if comp_name not in component_legend_added else "_nolegend_",
                )
                component_legend_added.add(comp_name)
    else:
        has_goal_grad = has_guidance_targets and goal_grad_vectors is not None and len(goal_grad_vectors) > 0
        if has_goal_grad:
            for idx, (gpos, gvec) in enumerate(zip(guidance_targets, goal_grad_vectors)):
                gxy = _to_plot_coords(env_id, np.asarray(gpos)[:2])
                if env_id in OGBENCH_ENVS:
                    gdx, gdy = gvec[0] / 4, gvec[1] / 4
                else:
                    gdx, gdy = float(gvec[0]), float(gvec[1])
                ax.quiver(gxy[0], gxy[1], gdx, gdy, color="lime", angles="xy", scale_units="xy",
                          scale=1.0, width=0.006, zorder=12,
                          label="Guidance grad" if idx == 0 else "_nolegend_")
    has_goal_grad = has_component_guidance or (
        has_guidance_targets and goal_grad_vectors is not None and len(goal_grad_vectors) > 0
    )

    has_prior_grad = has_guidance_targets and prior_grad_vectors is not None and len(prior_grad_vectors) > 0
    if has_prior_grad:
        for idx, (gpos, pvec) in enumerate(zip(guidance_targets, prior_grad_vectors)):
            gxy = _to_plot_coords(env_id, np.asarray(gpos)[:2])
            if env_id in OGBENCH_ENVS:
                pdx, pdy = pvec[0] / 4, pvec[1] / 4
            else:
                pdx, pdy = float(pvec[0]), float(pvec[1])
            ax.quiver(gxy[0], gxy[1], pdx, pdy, color="yellow", angles="xy", scale_units="xy",
                      scale=1.0, width=0.006, zorder=12,
                      label="Prior score" if idx == 0 else "_nolegend_")

    # --- pred_x_start (x̂_0) visualization ---
    # pred_x_start (x̂_0) dot: same color as x_t tail but lower alpha (alpha 0.35 vs 0.9)
    has_xs_pos = pred_x_start_pos is not None and len(pred_x_start_pos) > 0
    if has_xs_pos:
        for idx, xspos in enumerate(pred_x_start_pos):
            xsxy = _to_plot_coords(env_id, np.asarray(xspos)[:2])
            ax.scatter(xsxy[0], xsxy[1], color=tail_dot_color, marker="o", s=80, zorder=8,
                       alpha=0.35, edgecolors="none", linewidth=0,
                       label="x̂_0 tail" if idx == 0 else "_nolegend_")

    has_xs_component_guidance = (
        has_xs_pos
        and pred_x_start_guidance_component_vectors is not None
        and any(len(vecs) > 0 for vecs in pred_x_start_guidance_component_vectors.values())
    )
    if has_xs_component_guidance:
        xs_component_legend_added = set()
        for comp_name in ["anchor", "goal", "rdf", "particle"]:
            vecs = pred_x_start_guidance_component_vectors.get(comp_name)
            if vecs is None or len(vecs) == 0:
                continue
            spec = GUIDANCE_COMPONENT_SPECS.get(comp_name, {})
            color = spec.get("color", "lime")
            label_x0 = spec.get("label_x0", f"{comp_name} grad (x̂_0)")
            for xspos, gvec in zip(pred_x_start_pos, vecs):
                xsxy = _to_plot_coords(env_id, np.asarray(xspos)[:2])
                if env_id in OGBENCH_ENVS:
                    gdx, gdy = float(gvec[0]) / 4, float(gvec[1]) / 4
                else:
                    gdx, gdy = float(gvec[0]), float(gvec[1])
                arrow = FancyArrowPatch(
                    posA=(xsxy[0], xsxy[1]),
                    posB=(xsxy[0] + gdx, xsxy[1] + gdy),
                    arrowstyle='-|>', mutation_scale=10,
                    color=color, linestyle="dashed", linewidth=1.5,
                    transform=ax.transData, zorder=13,
                )
                ax.add_patch(arrow)
            if comp_name not in xs_component_legend_added:
                ax.plot([], [], color=color, linestyle="dashed", linewidth=1.5, label=label_x0)
                xs_component_legend_added.add(comp_name)
    else:
        has_xs_gg = has_xs_pos and pred_x_start_guidance_grad is not None and len(pred_x_start_guidance_grad) > 0
        if has_xs_gg:
            _xs_gg_legend_added = False
            for idx, (xspos, gvec) in enumerate(zip(pred_x_start_pos, pred_x_start_guidance_grad)):
                xsxy = _to_plot_coords(env_id, np.asarray(xspos)[:2])
                if env_id in OGBENCH_ENVS:
                    gdx, gdy = float(gvec[0]) / 4, float(gvec[1]) / 4
                else:
                    gdx, gdy = float(gvec[0]), float(gvec[1])
                arrow = FancyArrowPatch(
                    posA=(xsxy[0], xsxy[1]),
                    posB=(xsxy[0] + gdx, xsxy[1] + gdy),
                    arrowstyle='-|>', mutation_scale=10,
                    color="lime", linestyle="dashed", linewidth=1.5,
                    transform=ax.transData, zorder=13,
                )
                ax.add_patch(arrow)
                if not _xs_gg_legend_added:
                    ax.plot([], [], color="lime", linestyle="dashed", linewidth=1.5, label="Guidance grad (x̂_0)")
                    _xs_gg_legend_added = True
    has_xs_gg = has_xs_component_guidance or (
        has_xs_pos and pred_x_start_guidance_grad is not None and len(pred_x_start_guidance_grad) > 0
    )

    has_xs_pg = has_xs_pos and pred_x_start_prior_grad is not None and len(pred_x_start_prior_grad) > 0
    if has_xs_pg:
        _xs_pg_legend_added = False
        for idx, (xspos, pvec) in enumerate(zip(pred_x_start_pos, pred_x_start_prior_grad)):
            xsxy = _to_plot_coords(env_id, np.asarray(xspos)[:2])
            if env_id in OGBENCH_ENVS:
                pdx, pdy = float(pvec[0]) / 4, float(pvec[1]) / 4
            else:
                pdx, pdy = float(pvec[0]), float(pvec[1])
            arrow = FancyArrowPatch(
                posA=(xsxy[0], xsxy[1]),
                posB=(xsxy[0] + pdx, xsxy[1] + pdy),
                arrowstyle='-|>', mutation_scale=10,
                color="yellow", linestyle="dashed", linewidth=1.5,
                transform=ax.transData, zorder=13,
            )
            ax.add_patch(arrow)
            if not _xs_pg_legend_added:
                ax.plot([], [], color="yellow", linestyle="dashed", linewidth=1.5, label="Prior score (x̂_0)")
                _xs_pg_legend_added = True

    has_rollout_history = rollout_agent_history is not None and len(rollout_agent_history) > 0
    if has_rollout_history:
        hist_xy = _to_plot_coords(env_id, rollout_agent_history[:, batch_idx, :2])
        ax.plot(hist_xy[:, 0], hist_xy[:, 1], color="royalblue", linewidth=2.0, alpha=0.75,
                label="Agent Path")
        ax.scatter(hist_xy[:, 0], hist_xy[:, 1], c=np.arange(len(hist_xy)), cmap="Blues",
                   s=40, alpha=0.85, zorder=8)

    if rollout_current_agent is not None:
        agent_xy = _to_plot_coords(env_id, rollout_current_agent[0, batch_idx, :2])
        ax.scatter(agent_xy[0], agent_xy[1], c="deepskyblue", marker="o", s=180, zorder=13,
                   edgecolors="navy", linewidth=1.5, label="Agent")

    if rollout_current_subgoal is not None:
        subgoal_xy = _to_plot_coords(env_id, rollout_current_subgoal[0, batch_idx, :2])
        ax.scatter(subgoal_xy[0], subgoal_xy[1], c="lime", marker="o", s=180, zorder=14,
                   edgecolors="darkgreen", linewidth=1.5, label="Sub-goal")

    # --- Expanded-node value & uncertainty text labels ---
    # expanded_node_pos: (2,) world coords of the newly expanded node (vnode.obs[:2])
    # node_value_label: compact scientific string shown ABOVE the node position (all viz types)
    # node_unc_label:   multi-line string shown BELOW the node position (uncertainty viz only)
    expanded_node_pos = plot_data.get("expanded_node_pos")
    node_value_label  = plot_data.get("node_value_label")
    node_unc_label    = plot_data.get("node_unc_label")

    if expanded_node_pos is not None:
        epos = np.asarray(expanded_node_pos).flatten()[:2]
        epos_p = _to_plot_coords(env_id, epos)  # (2,) plot coords

        if node_value_label is not None:
            ax.annotate(
                node_value_label,
                xy=(epos_p[0], epos_p[1]),
                xytext=(0, +12),
                textcoords="offset points",
                fontsize=8,
                color="yellow",
                ha="center",
                va="bottom",
                zorder=20,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65, edgecolor="none"),
            )

        if node_unc_label is not None:
            ax.annotate(
                node_unc_label,
                xy=(epos_p[0], epos_p[1]),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=7,
                color="cyan",
                ha="center",
                va="top",
                zorder=20,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65, edgecolor="none"),
            )

    if ((node_trajectory is not None and len(node_trajectory) > 0)
            or (target_node_trajectory is not None and len(target_node_trajectory) > 0)
            or best_node_target is not None
            or hilp_grad_field is not None
            or has_guidance_targets
            or has_goal_grad
            or has_prior_grad
            or has_xs_pos
            or has_xs_gg
            or has_xs_pg
            or has_rollout_history
            or rollout_current_agent is not None
            or rollout_current_subgoal is not None
            or postprocessed_plan is not None):
        ax.legend(loc="upper right", fontsize=5)


def make_trajectory_images(env_id, trajectory, batch_size, start, goal, plot_end_points=True, waypoints=None):
    """
    Create trajectory visualization images.

    Args:
        trajectory: Can be either:
            - numpy array of shape (T, batch_size, 2) for backward compatibility (red)
            - dict with keys 'plan' (red) and 'node_path' (blue, optional)
    """
    images = []

    plot_data = _parse_trajectory_plot_inputs(trajectory)

    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        _render_trajectory_plot(
            fig,
            ax,
            env_id,
            plot_data,
            batch_idx,
            start,
            goal,
            plot_end_points,
            waypoints=waypoints,
        )
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
        images.append(img)

        plt.close()
    return images


def make_trajectory_videos(env_id, trajectory, batch_size, start, goal, rollout_agent_history,
                           rollout_subgoal_history, plot_end_points=True, max_frames=200,
                           path_stride=4, postprocessed_plan_per_frame=None, waypoints=None):
    plot_data = _parse_trajectory_plot_inputs(trajectory)
    rollout_agent_history = _maybe_add_batch_dim(rollout_agent_history)
    rollout_subgoal_history = _maybe_add_batch_dim(rollout_subgoal_history)
    videos = []

    num_frames = 0 if rollout_agent_history is None else rollout_agent_history.shape[0]
    frame_indices = _sample_frame_indices(num_frames, max_frames)
    for batch_idx in range(batch_size):
        frames = []
        fig, ax = plt.subplots()
        frame_pbar = tqdm(
            frame_indices,
            desc=f"validation video frames ({batch_idx})",
            leave=True,
            dynamic_ncols=True,
        )
        for frame_idx in frame_pbar:
            frame_plot_data = dict(plot_data)
            hist_idx = _sample_history_until(int(frame_idx), path_stride)
            frame_plot_data["rollout_agent_history"] = rollout_agent_history[hist_idx]
            frame_plot_data["rollout_current_agent"] = rollout_agent_history[frame_idx : frame_idx + 1]
            subgoal_frame = rollout_subgoal_history[frame_idx : frame_idx + 1]
            if np.isnan(subgoal_frame).all():
                frame_plot_data["rollout_current_subgoal"] = None
            else:
                frame_plot_data["rollout_current_subgoal"] = subgoal_frame
            if postprocessed_plan_per_frame is not None and frame_idx < len(postprocessed_plan_per_frame):
                frame_plot_data["postprocessed_plan"] = postprocessed_plan_per_frame[frame_idx]
            else:
                frame_plot_data["postprocessed_plan"] = None

            _render_trajectory_plot(
                fig,
                ax,
                env_id,
                frame_plot_data,
                batch_idx,
                start,
                goal,
                plot_end_points,
                waypoints=waypoints,
            )
            fig.tight_layout()
            fig.canvas.draw()
            img_shape = fig.canvas.get_width_height()[::-1] + (4,)
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
            frames.append(img[:, :, :3])

        frame_pbar.close()
        plt.close(fig)

        if frames:
            videos.append(np.stack(frames, axis=0).transpose(0, 3, 1, 2))
        else:
            videos.append(np.zeros((0, 3, 1, 1), dtype=np.uint8))
    return videos




def make_convergence_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    if env_id in OGBENCH_ENVS: # OGBench envs
        start, goal = np.array(start[batch_idx])/4+1, np.array(goal[batch_idx])/4+1
    else:
        start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the first plan
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[0][frame]
        plan_history_m = plan_history_m.numpy()
        if env_id in OGBENCH_ENVS: # OGBench envs
            ax.scatter(
                plan_history_m[:, 0]/4+1,
                plan_history_m[:, 1]/4+1,
                c=np.arange(len(plan_history_m))[::-1],
                cmap="Reds",
            )
        else:
            ax.scatter(
                plan_history_m[:, 0],
                plan_history_m[:, 1],
                c=np.arange(len(plan_history_m))[::-1],
                cmap="Reds",
            )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    frames = tqdm(range(len(plan_history[0])), desc="Making convergence animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_convergence.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)
    return filename


def prune_history(plan_history, trajectory, goal, open_loop_horizon):
    dist = np.linalg.norm(
        trajectory[:, :2] - np.array(goal)[None],
        axis=-1,
    )
    reached = dist < 0.2
    if reached.any():
        cap_idx = np.argmax(reached)
        trajectory = trajectory[: cap_idx + open_loop_horizon + 1]
        plan_history = plan_history[: cap_idx // open_loop_horizon + 2]

    pruned_plan_history = []
    for plans in plan_history:
        pruned_plan_history.append([])
        for m in range(len(plans)):
            plan = plans[m]
            pruned_plan_history[-1].append(plan)
        plan = pruned_plan_history[-1][-1]
        dist = np.linalg.norm(plan.numpy()[:, :2] - np.array(goal)[None], axis=-1)
        reached = dist < 0.2
        if reached.any():
            cap_idx = np.argmax(reached) + 1
            pruned_plan_history[-1] = [p[:cap_idx] for p in pruned_plan_history[-1]]
    return trajectory, pruned_plan_history


def make_mpc_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    if env_id in OGBENCH_ENVS: # OGBench envs
        start, goal = np.array(start[batch_idx])/4+1, np.array(goal[batch_idx])/4+1
    else:
        start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(plan_history, trajectory, goal, open_loop_horizon)

    # animate the convergence of the plans
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    trajectory_colors = np.linspace(0, 1, len(trajectory))

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        control_time_step = 0
        while frame >= 0:
            frame -= len(plan_history[control_time_step])
            control_time_step += 1
        control_time_step -= 1
        m = frame + len(plan_history[control_time_step])
        num_steps_taken = 1 + open_loop_horizon * control_time_step
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[control_time_step][m]
        plan_history_m = plan_history_m.numpy()
        if env_id in OGBENCH_ENVS:
            ax.scatter(
                trajectory[:num_steps_taken, 0]/4+1,
                trajectory[:num_steps_taken, 1]/4+1,
                c=trajectory_colors[:num_steps_taken],
                cmap="Blues",
            )
            ax.scatter(
                plan_history_m[:, 0]/4+1,
                plan_history_m[:, 1]/4+1,
                c=np.arange(len(plan_history_m))[::-1],
                cmap="Reds",
            )
        else:
            ax.scatter(
                trajectory[:num_steps_taken, 0],
                trajectory[:num_steps_taken, 1],
                c=trajectory_colors[:num_steps_taken],
                cmap="Blues",
            )
            ax.scatter(
                plan_history_m[:, 0],
                plan_history_m[:, 1],
                c=np.arange(len(plan_history_m))[::-1],
                cmap="Reds",
            )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    num_frames = sum([len(p) for p in plan_history])
    frames = tqdm(range(num_frames), desc="Making MPC animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_mpc.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)

    return filename
