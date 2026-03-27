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


def plot_start_goal(ax, start_goal: None):
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
    ]:
        if key in plot_data and plot_data[key] is not None:
            plot_data[key] = np.asarray(plot_data[key])

    plot_data["plan"] = _maybe_add_batch_dim(plot_data.get("plan"))
    plot_data["rollout_agent_history"] = _maybe_add_batch_dim(plot_data.get("rollout_agent_history"))
    plot_data["rollout_current_agent"] = _maybe_add_batch_dim(plot_data.get("rollout_current_agent"))
    plot_data["rollout_current_subgoal"] = _maybe_add_batch_dim(plot_data.get("rollout_current_subgoal"))
    plot_data["postprocessed_plan"] = _maybe_add_batch_dim(plot_data.get("postprocessed_plan"))
    return plot_data


def _render_trajectory_plot(fig, ax, env_id, plot_data, batch_idx, start, goal, plot_end_points=True):
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
    rollout_agent_history = plot_data.get("rollout_agent_history")
    rollout_current_agent = plot_data.get("rollout_current_agent")
    rollout_current_subgoal = plot_data.get("rollout_current_subgoal")
    postprocessed_plan = plot_data.get("postprocessed_plan")

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
        mag = np.sqrt(U ** 2 + V ** 2) + 1e-8
        U_n = U / mag
        V_n = V / mag

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
        plot_start_goal(ax, start_goal)

    if best_node_target is not None:
        pos = _to_plot_coords(env_id, np.asarray(best_node_target).flatten()[:2])
        ax.scatter(pos[0], pos[1], c="green", marker="*", s=300, zorder=10,
                   edgecolors="darkgreen", linewidth=1.5, label="Target")

    has_guidance_targets = guidance_targets is not None and len(guidance_targets) > 0
    if has_guidance_targets:
        for idx, gpos in enumerate(guidance_targets):
            gxy = _to_plot_coords(env_id, np.asarray(gpos)[:2])
            ax.scatter(gxy[0], gxy[1], c="lime", marker="o", s=120, zorder=9,
                       edgecolors="green", linewidth=1.5,
                       label="Guidance Tgt" if idx == 0 else "_nolegend_")

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
    # Lighter green dot + dashed arrows for ∂V/∂x̂_0 (green) and prior score (yellow)
    has_xs_pos = pred_x_start_pos is not None and len(pred_x_start_pos) > 0
    if has_xs_pos:
        for idx, xspos in enumerate(pred_x_start_pos):
            xsxy = _to_plot_coords(env_id, np.asarray(xspos)[:2])
            ax.scatter(xsxy[0], xsxy[1], c="lime", marker="o", s=80, zorder=8,
                       alpha=0.5, edgecolors="green", linewidth=1.0,
                       label="x̂_0 (pred_x_start)" if idx == 0 else "_nolegend_")

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
        ax.legend(loc="upper right", fontsize=8)


def make_trajectory_images(env_id, trajectory, batch_size, start, goal, plot_end_points=True):
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
        _render_trajectory_plot(fig, ax, env_id, plot_data, batch_idx, start, goal, plot_end_points)
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
        images.append(img)

        plt.close()
    return images


def make_trajectory_videos(env_id, trajectory, batch_size, start, goal, rollout_agent_history,
                           rollout_subgoal_history, plot_end_points=True, max_frames=200,
                           path_stride=4, postprocessed_plan_per_frame=None):
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

            _render_trajectory_plot(fig, ax, env_id, frame_plot_data, batch_idx, start, goal, plot_end_points)
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
