"""plan_viz.py
Plan visualization utilities for MCTD trajectory inspection.

Provides PlanVizMixin — a mixin class whose methods handle:
  - HILP value heatmap over the maze (_compute_hilp_heatmap)
  - HILP gradient field over the maze (_compute_guidance_grad_fields)
  - MCTS node-path trajectory extraction (_extract_node_trajectory)
  - Per-candidate denoising video logging (_log_candidate_plan_video)

Intended to be inherited by DiffusionForcingPlanning alongside DiffusionForcingBase.
All methods reference `self.*` attributes/methods provided by the base class or other mixins.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from utils.logging_utils import get_maze_grid, is_grid_env, make_trajectory_images
from . import guidance


class PlanVizMixin:
    """Mixin providing plan visualization methods."""

    # ------------------------------------------------------------------
    # HILP heatmap & gradient field
    # ------------------------------------------------------------------

    def _compute_hilp_heatmap(self, goal_obs_np: np.ndarray, grid_res: int = 40) -> dict:
        """
        Compute a HILP value heatmap over the maze for visualization.

        Args:
            goal_obs_np: Full goal observation (1D, obs_dim) in world (unnormalized) coords.
                         First 2 dims are x, y. Used as the fixed goal for V(s, goal).
            grid_res: Grid resolution per axis (default 40 → 1600 total points).

        Returns:
            dict with 'X', 'Y' (world-coord meshgrids, shape grid_res×grid_res) and
            'values' (HILP values, same shape).
        """
        # --- Determine world-coordinate extent from maze dimensions ---
        if is_grid_env(self.env_id):
            maze_grid = get_maze_grid(self.env_id)
            H = len(maze_grid)       # rows  → x axis in plot_maze_layout
            W = len(maze_grid[0])    # cols  → y axis in plot_maze_layout
            # Plot coord p = world / 4 + 1  →  world = (p - 1) * 4
            # Plot range: [0.5, H+0.5] for x, [0.5, W+0.5] for y
            x_min, x_max = (0.5 - 1) * 4, (H + 0.5 - 1) * 4   # ≈ -2 to (H-0.5)*4
            y_min, y_max = (0.5 - 1) * 4, (W + 0.5 - 1) * 4
        else:
            import torch
            obs_mean_np = self.data_mean[self.pos_dim_indices].cpu().numpy() if isinstance(self.data_mean, torch.Tensor) else np.array(self.data_mean)[self.pos_dim_indices]
            obs_std_np  = self.data_std[self.pos_dim_indices].cpu().numpy()  if isinstance(self.data_std,  torch.Tensor) else np.array(self.data_std)[self.pos_dim_indices]
            x_min, x_max = obs_mean_np[0] - 3 * obs_std_np[0], obs_mean_np[0] + 3 * obs_std_np[0]
            y_min, y_max = obs_mean_np[1] - 3 * obs_std_np[1], obs_mean_np[1] + 3 * obs_std_np[1]

        xs = np.linspace(x_min, x_max, grid_res)
        ys = np.linspace(y_min, y_max, grid_res)
        X, Y = np.meshgrid(xs, ys)             # both shape (grid_res, grid_res)
        grid_xy = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (N, 2)
        N = grid_xy.shape[0]

        # Build obs_batch: use goal_obs as template, swap in grid x,y
        obs_batch  = np.tile(goal_obs_np, (N, 1)).astype(np.float32)
        obs_batch[:, self.pos_dim_indices] = grid_xy

        # goal_batch: same goal obs repeated for each grid point
        goal_batch = np.tile(goal_obs_np, (N, 1)).astype(np.float32)

        values = self._compute_hilp_values(obs_batch, goal_batch)   # (N,)
        values = values.cpu().numpy().reshape(X.shape)               # (grid_res, grid_res)

        return {'X': X, 'Y': Y, 'values': values}

    def _compute_guidance_grad_fields(
        self,
        target_pos: np.ndarray,
        grid_step: float = 2.0,
    ) -> Optional[dict]:
        """
        Compute HILP gradient field over the maze for visualization.

        Args:
            target_pos: (2,) world coordinates of the fixed target (green star).
            grid_step: Grid spacing in world coordinates.

        Returns:
            dict with keys:
                'x_grid', 'y_grid': (H_g, W_g) world-coord meshgrids
                'hilp_grads': (H_g, W_g, 2) JAX-grad or finite-diff gradients
                'hilp_norms': (H_g, W_g) ||hilp_grad|| per grid point
            or None if target_pos is None or HILP model not loaded.
        """
        if target_pos is None:
            return None
        if not (hasattr(self, '_hilp_value_fn_instance') and self._hilp_value_fn_instance is not None):
            return None

        # --- 1. Grid extent ---
        if is_grid_env(self.env_id):
            maze_grid = get_maze_grid(self.env_id)
            H = len(maze_grid)
            W = len(maze_grid[0])
            x_min, x_max = (0.5 - 1) * 4, (H + 0.5 - 1) * 4
            y_min, y_max = (0.5 - 1) * 4, (W + 0.5 - 1) * 4
        else:
            import torch
            obs_mean_np = self.data_mean[self.pos_dim_indices].cpu().numpy() if isinstance(self.data_mean, torch.Tensor) else np.array(self.data_mean)[self.pos_dim_indices]
            obs_std_np  = self.data_std[self.pos_dim_indices].cpu().numpy()  if isinstance(self.data_std,  torch.Tensor) else np.array(self.data_std)[self.pos_dim_indices]
            x_min, x_max = obs_mean_np[0] - 3 * obs_std_np[0], obs_mean_np[0] + 3 * obs_std_np[0]
            y_min, y_max = obs_mean_np[1] - 3 * obs_std_np[1], obs_mean_np[1] + 3 * obs_std_np[1]

        xs = np.arange(x_min, x_max, grid_step)
        ys = np.arange(y_min, y_max, grid_step)
        X, Y = np.meshgrid(xs, ys)   # (H_g, W_g)
        N = X.size
        grid_x_flat = X.ravel().astype(np.float32)
        grid_y_flat = Y.ravel().astype(np.float32)

        # --- 2. Build grid observations padded with reference state ---
        _hilp_ref = getattr(self, '_hilp_ref_obs', None)
        if _hilp_ref is not None:
            obs_np = np.tile(_hilp_ref, (N, 1)).astype(np.float32)
        else:
            obs_np = np.zeros((N, self.hilp_obs_dim), dtype=np.float32)
        obs_np[:, 0] = grid_x_flat
        obs_np[:, 1] = grid_y_flat

        target_np = np.zeros((1, self.hilp_obs_dim), dtype=np.float32)
        if _hilp_ref is not None:
            target_np[0] = _hilp_ref
        target_np[0, self.pos_dim_indices] = target_pos[self.pos_dim_indices]

        # --- 3. Guidance gradients via shared helper (same logic as goal_guidance) ---
        # Any changes to guidance.compute_guidance_grad_np propagate here automatically.
        hilp_fn = self._hilp_value_fn_instance
        if not hasattr(hilp_fn, 'compute_grads') or not hasattr(hilp_fn, 'compute_values_np'):
            return None

        try:
            combined_flat, values_flat, _far_mask_flat, _hilp_flat, _rmse_flat, _, _ = guidance.compute_guidance_grad_np(
                self, obs_np, target_np, hilp_fn
            )
        except Exception:
            return None

        combined_grads = combined_flat.reshape(*X.shape, 2)
        hilp_values = values_flat.reshape(X.shape)

        # far_mask_grid: True where RMSE guidance was used (V < TD_thres)
        TD_thres = getattr(self, 'TD_thres_for_far_target', None)
        far_mask_grid = (hilp_values < float(TD_thres)).reshape(X.shape) if TD_thres is not None \
            else np.zeros(X.shape, dtype=bool)

        print(
            f"[grad_field] HILP V range: [{values_flat.min():.3f}, {values_flat.max():.3f}]"
            f"  TD_thres={TD_thres}  far_pts={far_mask_grid.sum()}/{far_mask_grid.size}",
            flush=True,
        )

        return {
            'x_grid': X,
            'y_grid': Y,
            'hilp_grads': combined_grads,
            'hilp_norms': np.linalg.norm(combined_grads, axis=-1),
            'hilp_values': hilp_values,       # (H_g, W_g) HILP value per grid point
            'far_mask_grid': far_mask_grid,   # (H_g, W_g) True → RMSE region
        }

    # ------------------------------------------------------------------
    # Node trajectory extraction
    # ------------------------------------------------------------------

    def _extract_node_trajectory(self, best_node) -> np.ndarray:
        """
        Recursively extract trajectory from best_node to root via sim_state.
        Returns array of shape (num_nodes, 2) with position at each node level.
        """
        trajectory = []
        current_node = best_node

        while current_node is not None:
            # Extract position from sim_state if available
            if current_node.sim_state is not None:
                pos = current_node.sim_state["qpos"][:2]  # physical: qpos[:2] = world (x,y)
                trajectory.append(pos)
            elif current_node.obs_pos is not None:
                trajectory.append(current_node.obs_pos)

            # Move to parent
            current_node = current_node._parent_node

        # Reverse so root is first
        if trajectory:
            trajectory = np.array(trajectory[::-1])
        else:
            trajectory = np.array([])

        return trajectory

    # ------------------------------------------------------------------
    # Per-candidate plan video logging
    # ------------------------------------------------------------------

    def _log_candidate_plan_video(
        self,
        vname: str,
        vinfo: dict,
        active_tree,
        start_np: np.ndarray,
        goal_np: np.ndarray,
        hilp_fn,
        tgt_vis_cache: dict,
        sc_by_name: dict,
        obs_std_np: np.ndarray,
        loops: int,
        log_prefix: str = "plan",
        plan_hist_override=None,
    ) -> None:
        """Build and log a per-candidate denoising video to WandB.

        Iterates over every denoising step stored in the candidate's
        ``plan_history`` and renders one maze-overlay frame per step.
        The resulting video (K×3×H×W) is logged via ``self.log_video``.

        Args:
            vname: Candidate name key from ``expanded_node_infos``.
            vinfo: Candidate info dict (must contain ``"node"``).
            active_tree: The MCTSTreeState that was just expanded.
            start_np: Start position array ``(1, pos_dim)``.
            goal_np: Goal position array ``(1, pos_dim)``.
            hilp_fn: HILP value function instance (may be None).
            tgt_vis_cache: Mutable cache dict mapping target-node name →
                ``(heatmap, grad_field)``.  Updated in-place.
            sc_by_name: Per-candidate denoising-step captures keyed by name.
            obs_std_np: Observation std in world coordinates ``(pos_dim,)``.
            loops: Current MPC loop index (used for WandB step alignment).
        """
        vnode = vinfo.get("node")
        if vnode is None or not vnode.plan_history:
            return

        seg_size = active_tree.plan_tokens // self.sequence_dividing_factor
        alen = self._get_prefix_len_frames_from_depth(vnode.depth, seg_size)

        # Single guidance-target index: end of the currently denoised segment.
        tail_vis = np.array([alen - 1]) if alen > 0 else np.array([], dtype=int)

        node_traj = self._extract_node_trajectory(vnode)
        is_fwd = active_tree.is_tree1
        tgt_node_traj = (
            self._extract_node_trajectory(vnode.target_node)
            if vnode.target_node is not None
            else None
        )
        tgt_pos = (
            vnode.target_node.obs_pos
            if vnode.target_node is not None and getattr(vnode.target_node, 'obs_pos', None) is not None
            else None
        )

        # Heatmap & grad-field for this candidate's target, cached by target name.
        cand_heatmap = None
        cand_grad_field = None
        if tgt_pos is not None and hilp_fn is not None:
            tgt_name = getattr(vnode.target_node, 'name', None)
            if tgt_name is not None and tgt_name in tgt_vis_cache:
                cand_heatmap, cand_grad_field = tgt_vis_cache[tgt_name]
            else:
                try:
                    cand_heatmap = self._compute_hilp_heatmap(
                        tgt_pos[:self.observation_dim].astype(np.float32)
                    )
                except Exception:
                    pass
                try:
                    cand_grad_field = self._compute_guidance_grad_fields(
                        tgt_pos[self.pos_dim_indices]
                    )
                except Exception:
                    pass
                if tgt_name is not None:
                    tgt_vis_cache[tgt_name] = (cand_heatmap, cand_grad_field)

        hist = plan_hist_override if plan_hist_override is not None else vnode.plan_history[-1]  # (K, plan_tokens*fs, c)
        frames = []
        step_caps = sc_by_name.get(vname)
        _dn_step_indices = getattr(self, '_parallel_plan_step_indices', None)
        _dn_step_total = getattr(self, '_parallel_plan_step_total', None)

        for m in range(hist.shape[0]):
            pu = self._unnormalize_x(hist[m].unsqueeze(1))  # (plan_tokens*fs, 1, c)
            plan = pu[:alen, :, self.pos_dim_indices].detach().cpu().numpy() if alen > 0 else None

            gpos = xs_gpos = gg = pg = xs_gg = xs_pg = None

            if len(tail_vis) > 0:
                gpos = pu[tail_vis, 0, self.pos_dim_indices].detach().cpu().numpy()

                if step_caps is not None and m < len(step_caps):
                    sc = step_caps[m]
                    pn = sc.get('prior_pred_noise')
                    gg_dict = sc.get('guidance_grads', {})

                    # DPS-theory-correct scales:
                    # sigma_t = sqrt(1-α_t); prior_scale = sqrt(α_t).
                    sigma_t = 1.0
                    prior_scale = 0.0
                    nl_data = sc.get('noise_level')
                    if nl_data is not None:
                        tail_tok = min(int(tail_vis[0]) // self.frame_stack, len(nl_data) - 1)
                        sigma_t = float(nl_data[tail_tok].clamp(0.0, 1.0).item())
                        prior_scale = float((1.0 - sigma_t ** 2) ** 0.5)

                    # Prior: -ε_θ × sqrt(α_t) — grows as denoising progresses
                    if pn is not None:
                        pg = -pn[tail_vis][:, self.pos_dim_indices].numpy() * obs_std_np * prior_scale

                    # Guidance: +grad × σ_t — x̂_0 moves toward goal
                    if gg_dict:
                        gg = (
                            sum(v[tail_vis][:, self.pos_dim_indices].numpy() for v in gg_dict.values())
                            * obs_std_np
                        ) * sigma_t

                    # pred_x_start (x̂_0) visualization
                    pxs_cpu = sc.get('pred_x_start_pos')
                    if pxs_cpu is not None:
                        pxs_world = self._unnormalize_x(pxs_cpu.unsqueeze(1))
                        xs_gpos = pxs_world[tail_vis, 0, self.pos_dim_indices].detach().cpu().numpy()

                        gg_clean_dict = sc.get('guidance_grads_clean', {})
                        if gg_clean_dict:
                            xs_gg = (
                                sum(v[tail_vis][:, self.pos_dim_indices].numpy() for v in gg_clean_dict.values())
                                * obs_std_np
                            ) * sigma_t

                        if pn is not None:
                            xs_pg = -pn[tail_vis][:, self.pos_dim_indices].numpy() * obs_std_np * prior_scale

            td = {
                'plan': plan,
                'is_forward_tree': is_fwd,
                'node_path': node_traj if node_traj is not None and len(node_traj) > 0 else None,
                'target_node_path': tgt_node_traj if tgt_node_traj is not None and len(tgt_node_traj) > 0 else None,
                'best_node_target': tgt_pos,
                'guidance_targets': gpos,
                'goal_grad_vectors': gg,
                'prior_grad_vectors': pg,
                'pred_x_start_pos': xs_gpos,
                'pred_x_start_guidance_grad': xs_gg,
                'pred_x_start_prior_grad': xs_pg,
                'hilp_heatmap': cand_heatmap,
                'hilp_grad_field': cand_grad_field,
            }
            img = make_trajectory_images(
                self.env_id, td, 1, start_np, goal_np, self.plot_end_points
            )[0]
            frame_rgb = img[:, :, :3].copy()
            if _dn_step_indices is not None and m < len(_dn_step_indices):
                _step_num = int(_dn_step_indices[m])
                _step_label = f"step {_step_num}"
                if _dn_step_total is not None:
                    _step_label += f"/{_dn_step_total - 1}"
                _pil_img = Image.fromarray(frame_rgb)
                _draw = ImageDraw.Draw(_pil_img)
                _draw.text((5, 5), _step_label, fill=(255, 255, 0))
                frame_rgb = np.array(_pil_img)
            frames.append(frame_rgb)

        video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (K, 3, H, W)
        label = self._node_path_label(vnode, active_tree.is_tree1, vnode.target_node)
        video_step = self.get_safe_wandb_step(min_step=loops)
        print(
            f"[wandb-video-debug] key={log_prefix}/plan_at_{label} shape={video.shape} "
            f"dtype={video.dtype} fps=4 step={video_step}",
            flush=True,
        )
        self.log_video(f"{log_prefix}/plan_at_{label}", video, fps=4, step=video_step)
