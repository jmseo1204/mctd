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

import math
import re
from typing import Optional

import matplotlib.cm as _mpl_cm
import numpy as np
from PIL import Image, ImageDraw

from utils.logging_utils import get_maze_grid, is_grid_env, make_trajectory_images
from . import guidance


def _fmt_sci(v: float) -> str:
    """Format float as compact scientific notation: 1.23e2, -4.56e-3.

    Strips leading zeros from exponent and the '+' sign:
        1.23e+02  →  1.23e2
       -4.56e-03  → -4.56e-3
    """
    s = f"{v:.2e}"
    s = re.sub(r'e\+0*(\d+)', r'e\1', s)
    s = re.sub(r'e-0*(\d+)', r'e-\1', s)
    return s


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
            elif current_node.obs is not None:
                trajectory.append(current_node.obs)

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
        is_uncertainty_viz: bool = False,
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
            obs_std_np: Observation std in world coordinates ``(obs_dim,)``.
            loops: Current MPC loop index (used for WandB step alignment).
            plan_hist_override: If provided, used instead of vnode.plan_history[-1].
                In normal mode: (fast_steps+1, plan_tokens*fs, c).
                In uncertainty mode: (fast_steps+1, plan_tokens*fs, G*K, c).
            is_uncertainty_viz: If True, plan_hist_override carries G*K fast-sampled
                trajectories. Each is rendered as a red-gradient scatter; green endpoint
                dots with alpha proportional to relative guidance-scale intensity.
        """
        vnode = vinfo.get("node")
        if vnode is None:
            return
        if plan_hist_override is None and not vnode.plan_history:
            return

        hist = plan_hist_override if plan_hist_override is not None else vnode.plan_history[-1]  # (steps, plan_tokens*fs, [G*K,] c)
        seg_size = self._require_current_plan_tokens() // self.sequence_dividing_factor
        alen = self._get_denoised_frame_prefix_len(vnode.depth, seg_size, hist.shape[1])
        target_node = getattr(vnode, "target_node", None) or vinfo.get("target_node")

        if is_uncertainty_viz and plan_hist_override is not None:
            # In uncertainty mode the endpoint is the NEXT segment boundary.
            unc_alen = self._get_denoised_frame_prefix_len(
                vnode.depth + 1, seg_size, plan_hist_override.shape[1]
            )
            tail_vis = np.array([unc_alen - 1]) if unc_alen > 0 else np.array([], dtype=int)

            # Precompute per-sample colors based on guidance scale (hue differentiation).
            # Batch ordering within each candidate: for g in G_scales: for k in K_copies.
            K = self.fast_sampling_multiple
            gk_total = plan_hist_override.shape[2]  # should equal G*K
            scales = self.mctd_guidance_scales
            G = len(scales)
            _cmap = _mpl_cm.get_cmap('plasma')
            # Evenly space G colors across the plasma colormap (use 0.15–0.85 to avoid extremes).
            if G == 1:
                _scale_rgbs = [_cmap(0.65)[:3]]
            else:
                _scale_rgbs = [_cmap(0.15 + 0.70 * i / (G - 1))[:3] for i in range(G)]
            # For sample index j: g_idx = j // K → pick color from _scale_rgbs[g_idx].
            unc_scale_colors = np.array([
                _scale_rgbs[j // K]
                for j in range(gk_total)
            ], dtype=np.float32)
            # Legend entries: one per unique guidance scale (G total), keyed by RGB color.
            unc_guidance_legend = [
                (f"gs={s:.3g}", _scale_rgbs[i])
                for i, s in enumerate(scales)
            ]
            # Per-sample cluster labels from unc_diagnostics (computed once, same across frames).
            _unc_diag = vinfo.get("unc_diagnostics")
            unc_cluster_labels = (
                np.asarray(_unc_diag["cluster_labels"], dtype=int)
                if _unc_diag is not None and "cluster_labels" in _unc_diag
                else None
            )
        else:
            unc_alen = None
            unc_scale_colors = None
            unc_guidance_legend = None
            unc_cluster_labels = None
            # Single guidance-target index: end of the currently denoised segment.
            tail_vis = np.array([alen - 1]) if alen > 0 else np.array([], dtype=int)

        node_traj = self._extract_node_trajectory(vnode)
        is_fwd = active_tree.is_tree1
        tgt_node_traj = (
            self._extract_node_trajectory(target_node)
            if target_node is not None
            else None
        )
        tgt_pos = (
            target_node.obs
            if target_node is not None and getattr(target_node, 'obs', None) is not None
            else None
        )

        # Heatmap & grad-field for this candidate's target, cached by target name.
        cand_heatmap = None
        cand_grad_field = None
        if tgt_pos is not None and hilp_fn is not None:
            tgt_name = getattr(target_node, 'name', None)
            if tgt_name is not None and tgt_name in tgt_vis_cache:
                cand_heatmap, cand_grad_field = tgt_vis_cache[tgt_name]
            else:
                try:
                    cand_heatmap = self._compute_hilp_heatmap(
                        tgt_pos.astype(np.float32)  # tgt_pos already indexed by obs_dim_indices
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

        # ------------------------------------------------------------------
        # Prepare per-frame text labels
        # ------------------------------------------------------------------
        # Value label: shown above vnode.obs for ALL viz types (expand/replan/uncertainty).
        _raw_value = vinfo.get("value")
        node_value_label = _fmt_sci(float(_raw_value)) if _raw_value is not None else None

        # Uncertainty diagnostics label: shown below vnode.obs only for uncertainty viz.
        node_unc_label = None
        if is_uncertainty_viz:
            unc_diag = vinfo.get("unc_diagnostics")
            if unc_diag is not None:
                _umode = getattr(self, 'uncertainty_mode', 'entropy')
                if _umode == 'variance':
                    node_unc_label = (
                        f"V_int={_fmt_sci(unc_diag.get('V_int', 0.0))}\n"
                        f"V_eg={_fmt_sci(unc_diag.get('V_ext_g', 0.0))}\n"
                        f"V_ec={_fmt_sci(unc_diag.get('V_ext_c', 0.0))}\n"
                        f"U={_fmt_sci(unc_diag.get('U', 0.0))}\n"
                        f"n_cl={unc_diag.get('n_clusters', '?')}"
                    )
                elif _umode == 'cluster':
                    node_unc_label = (
                        f"U={_fmt_sci(unc_diag.get('U', 0.0))}\n"
                        f"n_cl={unc_diag.get('n_clusters', '?')}"
                    )
                else:
                    node_unc_label = (
                        f"H_R={_fmt_sci(unc_diag.get('H_R', 0.0))}\n"
                        f"H_A={_fmt_sci(unc_diag.get('H_A', 0.0))}\n"
                        f"C_g={_fmt_sci(unc_diag.get('C_geom_hat', 0.0))}\n"
                        f"T_curr={_fmt_sci(unc_diag.get('T_curr', 0.0))}\n"
                        f"n_cl={unc_diag.get('n_clusters', '?')}"
                    )

        # Expanded node position (world coords) used as anchor for labels.
        expanded_node_pos = (
            vnode.obs[:2].copy() if vnode is not None and getattr(vnode, 'obs', None) is not None
            else None
        )

        # Compute per-candidate dot color based on guidance scale (hue differentiation).
        # For normal (non-uncertainty) viz, maps this candidate's guidance scale to the same
        # plasma colormap used by uncertainty viz, so colors are consistent across both modes.
        tail_dot_color = (0.2, 0.85, 0.2)  # default lime-like green (RGB)
        if not is_uncertainty_viz:
            _vgs = vinfo.get('guidance_scale')
            _all_scales = getattr(self, 'mctd_guidance_scales', None)
            if _vgs is not None and _all_scales is not None and len(_all_scales) > 0:
                _cmap2 = _mpl_cm.get_cmap('plasma')
                G2 = len(_all_scales)
                if G2 == 1:
                    tail_dot_color = _cmap2(0.65)[:3]
                else:
                    try:
                        _gs_idx = list(_all_scales).index(_vgs)
                        tail_dot_color = _cmap2(0.15 + 0.70 * _gs_idx / (G2 - 1))[:3]
                    except ValueError:
                        pass

        frames = []
        step_caps = sc_by_name.get(vname)
        _dn_step_indices = getattr(self, '_parallel_plan_step_indices', None)
        _dn_step_total = getattr(self, '_parallel_plan_step_total', None)

        for m in range(hist.shape[0]):
            gg_components = {}
            xs_gg_components = {}
            if is_uncertainty_viz:
                # hist[m]: (plan_tokens*fs, G*K, c) — unnormalize all samples at once
                pu_all = self._unnormalize_x(hist[m])  # (plan_tokens*fs, G*K, c)
                gk_total = pu_all.shape[1]
                plan = None  # main plan slot unused in uncertainty mode
                gpos = xs_gpos = gg = pg = xs_gg = xs_pg = None

                # Build G*K trajectories and their endpoints for this denoising step.
                unc_plans_list = []
                unc_guidance_targets = []
                for j in range(gk_total):
                    traj_j = (
                        pu_all[:unc_alen, j, self.pos_dim_indices].detach().cpu().numpy()
                        if unc_alen > 0
                        else np.zeros((0, len(self.pos_dim_indices)))
                    )  # (unc_alen, pos_dim)
                    unc_plans_list.append(traj_j)
                    if len(traj_j) > 0:
                        unc_guidance_targets.append(traj_j[-1])  # (pos_dim,)
                    else:
                        unc_guidance_targets.append(np.zeros(len(self.pos_dim_indices)))
                unc_guidance_targets = np.stack(unc_guidance_targets, axis=0)  # (G*K, pos_dim)
                unc_scale_colors_frame = unc_scale_colors  # (G*K, 3) precomputed RGB
            else:
                pu = self._unnormalize_x(hist[m].unsqueeze(1))  # (plan_tokens*fs, 1, c)
                plan = pu[:alen, :, self.pos_dim_indices].detach().cpu().numpy() if alen > 0 else None
                unc_plans_list = unc_guidance_targets = unc_scale_colors_frame = None

                gpos = xs_gpos = gg = pg = xs_gg = xs_pg = None

                if len(tail_vis) > 0:
                    gpos = pu[tail_vis][:, 0][:, self.pos_dim_indices].detach().cpu().numpy()

                    if step_caps is not None and m < len(step_caps):
                        sc = step_caps[m]
                        pn = sc.get('prior_pred_noise')
                        gg_dict = sc.get('guidance_grads', {})
                        gg_clean_dict = sc.get('guidance_grads_clean', {})
                        gg_xs_disp_dict = sc.get('guidance_xstart_displacements', {})
                        _use_direct_x0 = bool(getattr(self, 'use_directly_inject_guidance_to_x0', False))

                        # DPS-theory-correct scales:
                        # sigma_t = sqrt(1-α_t); prior_scale = sqrt(α_t).
                        sigma_t = 1.0
                        prior_scale = 0.0
                        nl_data = sc.get('noise_level')
                        if nl_data is not None:
                            tail_tok = min(int(tail_vis[0]) // self.frame_stack, len(nl_data) - 1)
                            sigma_t = float(nl_data[tail_tok].clamp(0.0, 1.0).item())
                            prior_scale = float((1.0 - sigma_t ** 2) ** 0.5)

                        # x_t score arrow: -ε_θ × σ_t
                        # -ε_θ/σ_t is the true score ∇log p_t(x_t), multiplied by σ_t²
                        # so arrows shrink naturally as denoising completes (σ_t → 0).
                        if pn is not None:
                            pg = -pn[tail_vis][:, self.pos_dim_indices].numpy() * obs_std_np * sigma_t

                        # x_t guidance arrow:
                        #   xt mode  : +grad × (σ_t / √α_t) — effective Δx̂_0 from xt guidance
                        #   x0 mode  : +η_t · ∂V/∂x̂_0, with η_t = κ · (σ_t / √α_t)
                        _gg_scale = sigma_t / prior_scale if prior_scale > 1e-8 else sigma_t
                        _direct_coeff = float(getattr(self, 'direct_x0_guidance_scale', 0.2))
                        if gg_dict:
                            for _gk in ("anchor", "goal", "rdf", "particle"):
                                _gv = gg_dict.get(_gk)
                                _gv_clean = gg_clean_dict.get(_gk)
                                if _gv is None and _gv_clean is None:
                                    continue
                                if _use_direct_x0:
                                    _src = _gv_clean
                                    _vec = (
                                        _src[tail_vis][:, self.pos_dim_indices].numpy() * obs_std_np
                                    ) * (_direct_coeff * _gg_scale)
                                else:
                                    _src = _gv
                                    _vec = (
                                        _src[tail_vis][:, self.pos_dim_indices].numpy() * obs_std_np
                                    ) * _gg_scale
                                gg_components[_gk] = _vec

                        # x̂_0 visualization: before-guidance position at this step
                        pxs_cpu = sc.get('pred_x_start_pos')       # x̂_0_before^(m)
                        pxs_after_cpu = sc.get('pred_x_start_pos_after')  # x̂_0_after^(m)
                        if pxs_cpu is not None:
                            pxs_world = self._unnormalize_x(pxs_cpu.unsqueeze(1))
                            xs_gpos = pxs_world[tail_vis][:, 0][:, self.pos_dim_indices].detach().cpu().numpy()

                            if _use_direct_x0 and gg_xs_disp_dict:
                                for _gk in ("anchor", "goal", "rdf", "particle"):
                                    _gv_disp = gg_xs_disp_dict.get(_gk)
                                    if _gv_disp is None:
                                        continue
                                    xs_gg_components[_gk] = (
                                        _gv_disp[tail_vis][:, self.pos_dim_indices].numpy() * obs_std_np
                                    )
                            else:
                                # xt mode: reuse the effective guidance-induced Δx̂_0 approximation,
                                # but anchor it at x̂_0_before for dashed visualization.
                                for _gk, _vec in gg_components.items():
                                    xs_gg_components[_gk] = _vec.copy()

                            # x̂_0 score arrow: x̂_0_before^(m) - x̂_0_after^(m-1)
                            # = how much the model (score) moved x̂_0 from where guidance
                            #   left it in the previous step to current before-guidance estimate.
                            # For m==0 (null capture), pxs_after_prev_cpu is None → xs_pg stays None.
                            sc_prev = step_caps[m - 1] if m > 0 else None
                            if sc_prev is not None:
                                pxs_after_prev_cpu = sc_prev.get('pred_x_start_pos_after')
                                if pxs_after_prev_cpu is not None:
                                    pxs_after_prev_world = self._unnormalize_x(pxs_after_prev_cpu.unsqueeze(1))
                                    xs_pg = (
                                        pxs_world[tail_vis][:, 0][:, self.pos_dim_indices]
                                        - pxs_after_prev_world[tail_vis][:, 0][:, self.pos_dim_indices]
                                    ).detach().cpu().numpy()

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
                'guidance_component_vectors': gg_components if gg_components else None,
                'pred_x_start_guidance_component_vectors': xs_gg_components if xs_gg_components else None,
                'hilp_heatmap': cand_heatmap,
                'hilp_grad_field': cand_grad_field,
                'unc_plans_list': unc_plans_list,
                'unc_guidance_targets': unc_guidance_targets,
                'unc_scale_colors': unc_scale_colors_frame,
                'unc_guidance_legend': unc_guidance_legend,
                'unc_cluster_labels': unc_cluster_labels,
                'tail_dot_color': tail_dot_color,
                # Value & uncertainty labels (same for every denoising-step frame)
                'expanded_node_pos': expanded_node_pos,
                'node_value_label': node_value_label,
                'node_unc_label': node_unc_label,
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

        if len(frames) <= 1:
            return  # skip single-frame (zero-length) video upload
        video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)  # (K, 3, H, W)
        label = self._node_path_label(vnode, active_tree.is_tree1, target_node)
        selection_count = vinfo.get("selection_count", getattr(vnode, "selection_count", None))
        label_with_count = (
            f"[{selection_count}]_{label}"
            if selection_count is not None
            else label
        )
        video_step = self.get_safe_wandb_step(min_step=loops)
        print(
            f"[wandb-video-debug] key={log_prefix}/plan_at_{label_with_count} shape={video.shape} "
            f"dtype={video.dtype} fps=4 step={video_step}",
            flush=True,
        )
        self.log_video(f"{log_prefix}/plan_at_{label_with_count}", video, fps=4, step=video_step)
