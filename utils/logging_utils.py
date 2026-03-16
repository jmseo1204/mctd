from typing import Optional
import wandb
import numpy as np
import torch

import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
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


def make_trajectory_images(env_id, trajectory, batch_size, start, goal, plot_end_points=True):
    """
    Create trajectory visualization images.

    Args:
        trajectory: Can be either:
            - numpy array of shape (T, batch_size, 2) for backward compatibility (red)
            - dict with keys 'plan' (red) and 'node_path' (blue, optional)
    """
    images = []

    # Handle both dict and array inputs for backward compatibility
    if isinstance(trajectory, dict):
        plan_trajectory = trajectory.get('plan')
        node_trajectory = trajectory.get('node_path')
        best_node_target = trajectory.get('best_node_target')  # single (2,) pos or None
        hilp_heatmap    = trajectory.get('hilp_heatmap')       # dict {X, Y, values} or None
    else:
        plan_trajectory = trajectory
        node_trajectory = None
        best_node_target = None
        hilp_heatmap = None

    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        if is_grid_env(env_id):
            maze_grid = get_maze_grid(env_id)
        else:
            maze_grid = None
        plot_maze_layout(ax, maze_grid)

        # Plot HILP value heatmap as a low-alpha background layer
        if hilp_heatmap is not None:
            X_w = hilp_heatmap['X']
            Y_w = hilp_heatmap['Y']
            vals = hilp_heatmap['values']
            if env_id in OGBENCH_ENVS:
                X_p = X_w / 4 + 1
                Y_p = Y_w / 4 + 1
            else:
                X_p, Y_p = X_w, Y_w
            ax.pcolormesh(X_p, Y_p, vals, shading='auto', cmap='viridis', alpha=0.5, zorder=1)

        # Plot plan trajectory (red)
        if plan_trajectory is not None:
            if env_id in OGBENCH_ENVS:  # OGBench envs
                ax.scatter(plan_trajectory[:, batch_idx, 0]/4+1, plan_trajectory[:, batch_idx, 1]/4+1,
                          c=np.arange(len(plan_trajectory)), cmap="Reds", alpha=0.8, label="Plan", s=50),
            else:
                ax.scatter(plan_trajectory[:, batch_idx, 0], plan_trajectory[:, batch_idx, 1],
                          c=np.arange(len(plan_trajectory)), cmap="Reds", alpha=0.8, label="Plan", s=50),

        # Plot node trajectory (blue) - tree path from root to leaf
        if node_trajectory is not None and len(node_trajectory) > 0:
            if env_id in OGBENCH_ENVS:  # OGBench envs
                ax.plot(node_trajectory[:, 0]/4+1, node_trajectory[:, 1]/4+1,
                       'b-', linewidth=2, alpha=0.6, label="Tree Path"),
                ax.scatter(node_trajectory[:, 0]/4+1, node_trajectory[:, 1]/4+1,
                          c='blue', marker='o', s=100, alpha=0.7, edgecolors='darkblue', linewidth=2, zorder=5),
            else:
                ax.plot(node_trajectory[:, 0], node_trajectory[:, 1],
                       'b-', linewidth=2, alpha=0.6, label="Tree Path"),
                ax.scatter(node_trajectory[:, 0], node_trajectory[:, 1],
                          c='blue', marker='o', s=100, alpha=0.7, edgecolors='darkblue', linewidth=2, zorder=5),

        if plot_end_points:
            if env_id in OGBENCH_ENVS:  # OGBench envs
                start_goal = (np.array(start[batch_idx])/4+1, np.array(goal[batch_idx])/4+1)
            else:
                start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)

        # Plot best_node's target_node obs_pos (single green star)
        if best_node_target is not None:
            pos = np.asarray(best_node_target).flatten()
            if env_id in OGBENCH_ENVS:
                px, py = pos[0] / 4 + 1, pos[1] / 4 + 1
            else:
                px, py = pos[0], pos[1]
            ax.scatter(px, py, c='green', marker='*', s=300, zorder=10,
                       edgecolors='darkgreen', linewidth=1.5, label="Target")

        # Add legend if node_trajectory or target is present
        if (node_trajectory is not None and len(node_trajectory) > 0) or best_node_target is not None:
            ax.legend(loc='upper right', fontsize=10)

        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
        images.append(img)

        plt.close()
    return images


def make_combined_grad_field_image(
    env_id: str,
    grad_field: dict,
    target_pos,
    start,
    goal,
) -> np.ndarray:
    """
    Draw RMSE and HILP gradient fields on a single figure.

    - RMSE arrows: blue, all same length (unit vectors toward target).
    - HILP arrows: red, unit-vector direction, alpha ∝ log(hilp_norm+1)
      so near-zero gradient regions are transparent.
    - Arrow length is normalized for both types (direction only), magnitude
      is encoded via color/alpha to avoid invisibility when raw magnitudes differ.

    Args:
        env_id: Environment ID string.
        grad_field: Dict from _compute_guidance_grad_fields.
        target_pos: (2,) world coords of the fixed target (green star).
        start: (batch, 2) world coords for start marker.
        goal: (batch, 2) world coords for goal marker.

    Returns:
        RGBA uint8 numpy image.
    """
    X_w = grad_field['x_grid']
    Y_w = grad_field['y_grid']

    if env_id in OGBENCH_ENVS:
        X_p = X_w / 4 + 1
        Y_p = Y_w / 4 + 1
        tpx = float(target_pos[0]) / 4 + 1
        tpy = float(target_pos[1]) / 4 + 1
    else:
        X_p, Y_p = X_w, Y_w
        tpx, tpy = float(target_pos[0]), float(target_pos[1])

    def _normalize_grad(grads: np.ndarray):
        """
        Normalize each component (U, V) of the gradient field using
        the global mean and std of the raw magnitudes across all grid points.
        Returns (U_norm, V_norm, mean_scalar, std_scalar).
        """
        U_raw = grads[:, :, 0]
        V_raw = grads[:, :, 1]
        mag = np.sqrt(U_raw ** 2 + V_raw ** 2)   # (H, W) per-point magnitude
        m = float(mag.mean())
        s = float(mag.std()) + 1e-8
        # Normalise each vector by (mag - mean) / std so typical arrows ≈ 1 unit
        scale = (mag - m) / s + 1.0   # shift so mean → 1.0
        # Avoid negative scales (can flip arrow direction)
        scale = np.clip(scale, 0.0, None)
        denom = mag + 1e-8
        U_norm = U_raw / denom * scale
        V_norm = V_raw / denom * scale
        return U_norm, V_norm, m, s

    fig, ax = plt.subplots(figsize=(7, 7))
    maze_grid = get_maze_grid(env_id) if is_grid_env(env_id) else None
    plot_maze_layout(ax, maze_grid)

    # --- RMSE arrows (blue) — normalize using mean/std of raw magnitudes ---
    rmse_grads = grad_field['rmse_grads']   # (H, W, 2)
    U_r, V_r, rmse_mean, rmse_std = _normalize_grad(rmse_grads)
    ax.quiver(
        X_p, Y_p, U_r, V_r,
        color='royalblue',
        alpha=0.55,
        angles='xy',
        pivot='mid',
        scale=None,
        zorder=3,
        label=f'RMSE  (mean={rmse_mean:.2e}, std={rmse_std:.2e})',
    )

    # --- HILP arrows (red) — normalize using mean/std of raw magnitudes ---
    hilp_grads = grad_field['hilp_grads']   # (H, W, 2)
    U_h, V_h, hilp_mean, hilp_std = _normalize_grad(hilp_grads)
    ax.quiver(
        X_p, Y_p, U_h, V_h,
        color='crimson',
        alpha=0.55,
        angles='xy',
        pivot='mid',
        scale=None,
        zorder=4,
        label=f'HILP  (mean={hilp_mean:.2e}, std={hilp_std:.2e})',
    )

    # Diagnostics
    import sys as _sys
    print(
        f"[TD_field DIAG] RMSE mag mean={rmse_mean:.4e} std={rmse_std:.4e} | "
        f"HILP mag mean={hilp_mean:.4e} std={hilp_std:.4e}",
        file=_sys.stderr, flush=True,
    )

    ax.scatter(tpx, tpy, c='green', marker='*', s=300, zorder=10,
               edgecolors='darkgreen', linewidth=1.5, label='Target')

    if start is not None and goal is not None:
        if env_id in OGBENCH_ENVS:
            start_goal = (np.array(start[0]) / 4 + 1, np.array(goal[0]) / 4 + 1)
        else:
            start_goal = (start[0], goal[0])
        plot_start_goal(ax, start_goal)

    ax.set_title('TD Gradient Field  (blue=RMSE, red=HILP)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    img_shape = fig.canvas.get_width_height()[::-1] + (4,)
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape)
    plt.close()
    return img


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
