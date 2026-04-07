from typing import Optional, Union, Dict, Tuple
import time
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat, reduce


def _kde_score_np(query_xy: np.ndarray, data_xy: np.ndarray, sigma: float) -> np.ndarray:
    """∇log p(query) — Gaussian KDE score function.

    p(s) ∝ Σᵢ exp(-||s - sᵢ||² / (2σ²))
    ∇log p(s) = Σᵢ wᵢ·(sᵢ - s)/σ²   where wᵢ ∝ exp(-||s-sᵢ||²/(2σ²))

    Always points toward higher training-data density (inside corridors).

    Args:
        query_xy: (N, 2) world coordinates to evaluate at
        data_xy:  (M, 2) training data positions (pre-subsampled)
        sigma:    KDE bandwidth

    Returns:
        (N, 2) score vectors
    """
    diff = data_xy[None, :, :] - query_xy[:, None, :]   # (N, M, 2)
    sq   = (diff ** 2).sum(-1)                           # (N, M)
    logw = -sq / (2.0 * sigma ** 2)
    logw -= logw.max(-1, keepdims=True)                  # numerical stability
    w    = np.exp(logw)                                  # (N, M)
    return (w[:, :, None] * diff).sum(1) / (w.sum(-1, keepdims=True) * sigma ** 2)  # (N, 2)

def weighted_loss(
    planner,
    dist: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: tuple = (0, 2),
) -> torch.Tensor:
    """
    Helper function to compute weighted loss from distance tensor.

    Args:
        planner: The DiffusionForcingPlanning instance
        dist: (t*fs, b, c) or (t, b, n) distance tensor
        weight: (t,) or (t, n) weight tensor for temporal weighting
        dim: tuple of dimensions to reduce over

    Returns:
        weighted_loss: (b,) scalar loss per batch element
    """
    dist_o, dist_a, _ = planner.split_bundle(
        dist
    )  # guidance observation and action with separate weights
    # dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
    dist_o = dist_o[:, :, planner.pos_dim_indices]
    dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=1)
    dist_o = (dist_o + 1e-6).sqrt()
    # dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
    dist = dist_o
    if weight is None:
        weight = torch.ones_like(dist)
    else:
        assert len(weight.shape) == 1, f"weight shape {weight.shape} is not 1D"
        weight = repeat(weight, "t -> t n", n=dist.shape[-1])
        weight = torch.ones_like(dist) * weight[:, None]  #  t b n
    # Divide by number of active (non-zero weight) positions, not total T.
    # This ensures the effective guidance scale equals anchor_guidance_scale
    # regardless of sequence length, preventing the 3/1000 dilution.
    weighted_sum = (dist * weight).sum(dim=dim)
    active_count = (weight > 0).float().sum(dim=dim).clamp(min=1)
    return weighted_sum / active_count  # * dist.shape[1] DO NOT DELETE THIS COMMENT

def prepare_pred(planner, x: torch.Tensor) -> torch.Tensor:
    """
    Helper to rearrange and unnormalize predictions.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor

    Returns:
        pred: (t*fs, b, c) unnormalized prediction tensor
    """
    # x is a tensor of shape [t b (fs c)]
    pred = rearrange(
        x, "t b (fs c) -> (t fs) b c", fs=planner.frame_stack
    )  # (t*fs, b, c)
    return planner._unnormalize_x(pred)

def compute_guidance_grad_np(
    planner,
    obs_np: np.ndarray,
    target_np: np.ndarray,
    hilp_fn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Core guidance gradient logic shared by goal_guidance() and _compute_guidance_grad_fields().

    Given pre-padded obs_np and target_np (both in hilp_obs_dim space):
      1. Compute ∂V/∂obs via JAX grad (HILP)
      2. Normalize: ∇V/|∇V|
      3. Optionally add KDE score:  ∇V/|∇V| + kde_lam·∇log p  (use_score_func_with_TD)
      4. Compute HILP values V(obs, target) for TD threshold check
      5. For elements with V < TD_thres_for_far_target → replace with RMSE direction

    Changing logic here automatically propagates to both the diffusion guidance and
    the plan/plan_at_* visualization arrows.

    Args:
        planner:    DiffusionForcingPlanning instance (reads config attrs)
        obs_np:     (N, hilp_obs_dim) float32 — observations padded to HILP input dim
        target_np:  (1, hilp_obs_dim) or (N, hilp_obs_dim) float32 — target padded to HILP input dim
        hilp_fn:    HILPJax instance with compute_grads / compute_values_np

    Returns:
        combined_np:    (N, 2) final guidance direction per observation
        hilp_values_np: (N,)  HILP value V(obs, target) — for diagnostics / threshold coloring
    """
    if target_np.shape[0] == 1:
        target_xy = np.broadcast_to(target_np[:, :2], (len(obs_np), 2))
        goal_rep_np = np.broadcast_to(target_np[:1], (len(obs_np), target_np.shape[-1])).copy()
    else:
        assert target_np.shape[0] == len(obs_np), (
            f"target_np.shape[0]={target_np.shape[0]} must be 1 or len(obs_np)={len(obs_np)}"
        )
        target_xy = target_np[:, :2]
        goal_rep_np = target_np.copy()

    # 1. HILP gradient ∂V/∂obs[:2] via JAX
    _t_grad0 = time.time()
    hilp_grad_np = hilp_fn.compute_grads(obs_np, goal_rep_np)  # (N, 2)
    hilp_grad_ms = (time.time() - _t_grad0) * 1000

    # 2. Normalize: ∇V/|∇V|
    grad_mag = np.linalg.norm(hilp_grad_np, axis=-1, keepdims=True).clip(min=1e-8)
    hilp_grad_normalized = hilp_grad_np / grad_mag

    # 3. Optionally add KDE score:  ∇V/|∇V| + kde_lam·∇log p
    use_score_func = getattr(planner, 'use_score_func_with_TD', True)
    kde_lam = getattr(planner, 'kde_lam', 0.0)
    if use_score_func and kde_lam > 0.0:
        query_xy = obs_np[:, :2]
        if hasattr(planner, "_kde_grid_cache"):
            kde_score = planner._get_kde_score_grid(query_xy)
        else:
            kde_data_xy = planner._get_kde_data_xy()
            kde_score = _kde_score_np(query_xy, kde_data_xy, sigma=planner.kde_sigma)
        
        # Filter: set score to 0 where ||kde_score|| < μ + k * sigma
        kde_score_norm = np.linalg.norm(kde_score, axis=-1)  # (N,)
        score_mean = float(kde_score_norm.mean())
        score_std = float(kde_score_norm.std()) + 1e-8
        threshold = score_mean + planner.kde_grad_thres_sigma_coeff * score_std
        mask_below_threshold = kde_score_norm < threshold  # (N,)
        kde_score_filtered = kde_score.copy()
        kde_score_filtered[mask_below_threshold] = 0.0
        
        hilp_combined_np = hilp_grad_normalized + kde_lam * kde_score_filtered
    else:
        hilp_combined_np = hilp_grad_normalized

    # 3b. Optionally re-normalize the combined vector to unit length
    if getattr(planner, 'regularize_goal_guidance', False):
        combined_mag = np.linalg.norm(hilp_combined_np, axis=-1, keepdims=True).clip(min=1e-8)
        hilp_combined_np = hilp_combined_np / combined_mag

    # 4. HILP values V(obs, target) for threshold-based switching
    # Use planner._compute_hilp_values for consistency with heatmap (same pessimistic
    # min(v1,v2) path, same _hilp_ref_obs padding). For HILPJax v1==v2 so min is a no-op,
    # but this keeps the computation identical across heatmap / grad-field / guidance.
    _t_val0 = time.time()
    hilp_values_np = planner._compute_hilp_values(obs_np, goal_rep_np).cpu().numpy()  # (N,)
    hilp_value_ms = (time.time() - _t_val0) * 1000

    # 5. TD threshold: switch to RMSE direction for temporally-far elements (V < TD_thres)
    TD_thres = getattr(planner, 'TD_thres_for_far_target', None)
    N = len(obs_np)
    if TD_thres is not None:
        far_mask = hilp_values_np < float(TD_thres)  # small V ↔ far from target
        delta = target_xy - obs_np[:, :2]  # (N, 2) vector toward target
        delta_mag = np.linalg.norm(delta, axis=-1, keepdims=True).clip(min=1e-8)
        rmse_grad_np = delta / delta_mag           # unit vector toward target
        combined_np = np.where(far_mask[:, None], rmse_grad_np, hilp_combined_np)
    else:
        far_mask = np.zeros(N, dtype=bool)
        rmse_grad_np = np.zeros((N, 2), dtype=np.float32)
        combined_np = hilp_combined_np

    return combined_np, hilp_values_np, far_mask, hilp_combined_np, rmse_grad_np, hilp_grad_ms, hilp_value_ms


def goal_guidance(
    planner, x: torch.Tensor, goal: torch.Tensor, horizon: int,
    active_tail_per_batch: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
    """
    Target guidance to reach goal/start.

    Returns separate losses for HILP (near target) and RMSE (far target) components
    so that combined_guidance can apply TD_guidance_scale and rmse_guidance_scale independently.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor from diffusion model
          where t=plan_tokens, b=batch_size, fs*c flattened observation
        goal: (b, obs_dim) normalized goal observations
        horizon: planning horizon in timesteps
        active_tail_per_batch: When provided (smooth noise mode), list of length B with
            per-batch frame-space indices of the single tail to apply guidance to.
            None → apply to all segment tails (causal mode default).

    Returns:
        (hilp_loss, rmse_loss, hilp_per_batch_list, rmse_per_batch_list):
            hilp_loss: scalar pseudo-loss for near-target (HILP) positions; scale with TD_guidance_scale
            rmse_loss: scalar pseudo-loss for far-target (RMSE) positions; scale with rmse_guidance_scale
            hilp_per_batch_list: per-batch HILP pseudo-loss values (Python list)
            rmse_per_batch_list: per-batch RMSE pseudo-loss values (Python list)
    """
    pred = prepare_pred(planner, x)

    if not planner.use_reward:
        # Temporal consistency guidance via shifted predictions
        # pred: (t fs) b c

        target = goal

        # Unnormalize target to match the scale of pred (which is unnormalized above)
        target = planner._unnormalize_x(target)

        target_guidance = torch.stack([target] * pred.shape[0])  # (t*fs, b, c)

        # Compute distance at tail positions of each segment only
        T, B = pred.shape[0], pred.shape[1]
        segment_size = horizon // planner.sequence_dividing_factor

        if active_tail_per_batch is not None:
            # ── smooth mode: per-batch single active tail ──────────────────────
            # Each candidate has exactly one segment being denoised this turn;
            # restrict guidance to that tail to avoid perturbing already-denoised
            # past segments that still participate in the smooth forward pass.
            _atail = torch.tensor(active_tail_per_batch, device=pred.device)  # (B,)
            _bidx  = torch.arange(B, device=pred.device)

            _obs_idx = planner.obs_bundle_indices  # model-space obs indices (Method A)
            hilp_obs_dim = planner.hilp_obs_dim
            _ref = getattr(planner, '_hilp_ref_obs', None)

            obs_tail_raw = pred[_atail, _bidx][..., _obs_idx].detach().cpu().numpy().astype(np.float32)  # (B, n_obs)
            if _ref is not None:
                obs_tail_np = np.tile(_ref, (B, 1)).astype(np.float32)
            else:
                obs_tail_np = np.zeros((B, hilp_obs_dim), np.float32)
            obs_tail_np[:, _obs_idx] = obs_tail_raw

            target_raw = target[:, _obs_idx].detach().cpu().numpy().astype(np.float32)  # (B, n_obs)
            if _ref is not None:
                target_np = np.tile(_ref, (B, 1)).astype(np.float32)
            else:
                target_np = np.zeros((B, hilp_obs_dim), np.float32)
            target_np[:, _obs_idx] = target_raw

            hilp_fn = getattr(planner, '_hilp_value_fn_instance', None)
            assert hilp_fn is not None and hasattr(hilp_fn, 'compute_grads'), \
                "active_tail_per_batch requires HILP guidance to be configured"

            _t0 = time.time()
            combined_np, hilp_values_np, far_mask_np, hilp_combined_np, rmse_grad_np, _hilp_grad_ms, _hilp_value_ms = compute_guidance_grad_np(
                planner, obs_tail_np, target_np, hilp_fn
            )
            _t_hilp_ms = (time.time() - _t0) * 1000

            # shapes: (B, 1) and (B, 2) — one tail per batch item
            far_mask_t = torch.from_numpy(far_mask_np).reshape(B, 1).to(pred.device)   # (B, 1)
            hilp_t     = torch.from_numpy(hilp_combined_np).reshape(B, 2).to(pred.device).detach()  # (B, 2)
            rmse_t     = torch.from_numpy(rmse_grad_np).reshape(B, 2).to(pred.device).detach()      # (B, 2)

            TD_thres = getattr(planner, 'TD_thres_for_far_target', None)
            from utils.tracer import get_tracer as _get_tracer
            _tracer = _get_tracer()
            if _tracer:
                _tracer.log("timing.guidance_breakdown", {
                    "hilp_ms": round(_t_hilp_ms, 2),
                    "hilp_grad_ms": round(_hilp_grad_ms, 2),
                    "hilp_value_ms": round(_hilp_value_ms, 2),
                    "batch_size": B,
                    "use_score_func": getattr(planner, 'use_score_func_with_TD', True),
                    "far_count": int(far_mask_np.sum()),
                    "TD_thres": TD_thres,
                    "hilp_values": hilp_values_np.tolist(),
                    "active_tail_mode": "per_batch",
                })

            # pred[_atail, _bidx, :2]: (B, 2) — gather active tail per batch item
            _pos = pred[_atail, _bidx][:, planner.pos_dim_indices]  # (B, pos_dim)
            hilp_pseudo_loss = (_pos * hilp_t * (~far_mask_t)).sum(dim=1)  # (B,)
            rmse_pseudo_loss = (_pos * rmse_t * far_mask_t).sum(dim=1)     # (B,)
            return hilp_pseudo_loss.mean(), rmse_pseudo_loss.mean(), hilp_pseudo_loss.tolist(), rmse_pseudo_loss.tolist()

        # ── causal mode (default): all segment tails ──────────────────────────
        # Tail positions: last frame of each segment
        tail_pos = torch.arange(
            planner.frame_stack + segment_size - 1,
            planner.frame_stack + horizon,
            segment_size,
            device=pred.device,
        )
        tail_pos = tail_pos[tail_pos < T]

        # --- HILP unit-vector guidance (pseudo-loss trick for JAX backend) ---
        # HILPJax breaks PyTorch autograd, so we compute gradients via JAX and
        # inject them using a pseudo-loss: d/d(pred) = hilp_grad_unit at the
        # segment tail positions. This matches the older RMSE-based guidance,
        # which applied guidance to every segment tail instead of only the last one.
        hilp_fn = getattr(planner, '_hilp_value_fn_instance', None)

        if hilp_fn is not None and hasattr(hilp_fn, 'compute_grads'):
            _obs_idx = planner.obs_bundle_indices  # model-space obs indices (Method A)
            _n_obs = len(_obs_idx)
            hilp_obs_dim = planner.hilp_obs_dim
            _ref = getattr(planner, '_hilp_ref_obs', None)

            # Build padded obs at all tail positions (len(tail_pos) * B rows)
            obs_tail_raw = pred[tail_pos][..., _obs_idx].reshape(-1, _n_obs).detach().cpu().numpy().astype(np.float32)
            if _ref is not None:
                obs_tail_np = np.tile(_ref, (len(obs_tail_raw), 1)).astype(np.float32)
            else:
                obs_tail_np = np.zeros((len(obs_tail_raw), hilp_obs_dim), np.float32)
            obs_tail_np[:, _obs_idx] = obs_tail_raw

            # Build padded target obs for each tail position and batch item
            target_raw = target[:, _obs_idx].detach().cpu().numpy().astype(np.float32)
            target_tail_raw = np.repeat(target_raw[None, :, :], len(tail_pos), axis=0).reshape(-1, _n_obs)
            if _ref is not None:
                target_np = np.tile(_ref, (len(target_tail_raw), 1)).astype(np.float32)
            else:
                target_np = np.zeros((len(target_tail_raw), hilp_obs_dim), np.float32)
            target_np[:, _obs_idx] = target_tail_raw

            _t0 = time.time()
            # Delegate all normalization / KDE / TD-threshold logic to shared helper.
            # Any changes to compute_guidance_grad_np automatically apply here AND
            # to _compute_guidance_grad_fields (plan/plan_at_* visualization).
            combined_np, hilp_values_np, far_mask_np, hilp_combined_np, rmse_grad_np, _hilp_grad_ms, _hilp_value_ms = compute_guidance_grad_np(
                planner, obs_tail_np, target_np, hilp_fn
            )
            _t_hilp_ms = (time.time() - _t0) * 1000

            # Shape: (len(tail_pos), B, 2) and (len(tail_pos), B, 1) for masking
            far_mask_t = torch.from_numpy(far_mask_np).reshape(len(tail_pos), B, 1).to(pred.device)  # bool
            hilp_t = torch.from_numpy(hilp_combined_np).reshape(len(tail_pos), B, 2).to(pred.device).detach()
            rmse_t = torch.from_numpy(rmse_grad_np).reshape(len(tail_pos), B, 2).to(pred.device).detach()

            # far_mask for logging
            far_mask = far_mask_np  # (len(tail_pos)*B,) — keep for tracer below
            TD_thres = getattr(planner, 'TD_thres_for_far_target', None)

            # [TIMING] Log latency per guidance call
            from utils.tracer import get_tracer as _get_tracer
            _tracer = _get_tracer()
            if _tracer:
                _tracer.log("timing.guidance_breakdown", {
                    "hilp_ms": round(_t_hilp_ms, 2),
                    "hilp_grad_ms": round(_hilp_grad_ms, 2),
                    "hilp_value_ms": round(_hilp_value_ms, 2),
                    "batch_size": B,
                    "use_score_func": getattr(planner, 'use_score_func_with_TD', True),
                    "far_count": int(far_mask.sum()),
                    "TD_thres": TD_thres,
                    "hilp_values": hilp_values_np.tolist(),
                })

            # Separate pseudo-losses for HILP (near) and RMSE (far) positions.
            # Each is a unit-direction dot-product; scaling is applied externally in combined_guidance.
            _pos = pred[tail_pos][:, :, planner.pos_dim_indices]  # (n_tails, B, pos_dim)
            hilp_pseudo_loss = (_pos * hilp_t * (~far_mask_t)).sum(dim=(0, 2))  # (B,)
            rmse_pseudo_loss = (_pos * rmse_t * far_mask_t).sum(dim=(0, 2))    # (B,)

            return hilp_pseudo_loss.mean(), rmse_pseudo_loss.mean(), hilp_pseudo_loss.tolist(), rmse_pseudo_loss.tolist()

        # --- Fallback: RMSE distance to goal ---
        assert 0, "HILP guidance is not recognized for some reason"
        dist_mse = nn.functional.mse_loss(pred, target_guidance, reduction="none")  # (T, B, C)
        dist_rmse = torch.sqrt(dist_mse + 1e-8)
        dist_target = dist_rmse

        target_weight = torch.zeros(T, device=planner.device)
        target_weight[tail_pos] = 1

        dist_per_batch = weighted_loss(planner, dist_target, target_weight)
        last_token_dist = weighted_loss(planner, dist_target, weight=None, dim=(-1,))[-1]

    else:
        raise NotImplementedError(
            "reward guidance not officially supported yet, although implemented"
        )

    zero = x.sum() * 0.0
    return -(dist_per_batch).mean(), zero, dist_per_batch.tolist(), last_token_dist.tolist()

def anchor_dist_guidance(planner, x: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    Anchor distance regularization: pulls each segment head toward the end of the
    previous segment, enforcing temporal continuity between sub-plans.

    Gradient scale is normalized to match goal_guidance: per-active-position magnitude
    is 1/B regardless of sequence_dividing_factor.  weighted_loss() divides by
    active_count (= n_segs), so we multiply back by n_active to cancel that factor.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor
        horizon: planning horizon

    Returns:
        loss: scalar loss (positive → DPS pushes segment heads toward anchors)
    """
    pred = prepare_pred(planner, x)
    pred_detached = pred.detach()

    segment_size = horizon // planner.sequence_dividing_factor
    head_of_each_segments = pred_detached[
        planner.frame_stack - 1 : planner.frame_stack + horizon - 1 : segment_size
    ]
    anchor_plan = torch.zeros_like(pred_detached)
    anchor_plan[
        planner.frame_stack : planner.frame_stack + horizon : segment_size
    ] = head_of_each_segments
    dist_anchor = nn.functional.mse_loss(pred, anchor_plan, reduction="none")

    anchor_weight = torch.zeros_like(pred_detached[:, 0, 0])
    anchor_weight[
        planner.frame_stack : planner.frame_stack + horizon : segment_size
    ] = 1
    n_active = (anchor_weight > 0).sum().float().clamp(min=1)

    # weighted_loss divides by active_count (n_segs); multiply back to cancel that
    # so the per-position gradient magnitude is 1/B, matching goal_guidance.
    weighted_dist_anchor = weighted_loss(planner, dist_anchor, anchor_weight)
    return -(weighted_dist_anchor * n_active).mean()

def segment_rdf_guidance(planner, x: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    Temporal consistency guidance using RDF kernel with a sliding window.
    Repels current state from states in the window [idx-7-segment_size, idx-7].

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor
        horizon: planning horizon

    Returns:
        loss: scalar negative repulsion loss
    """
    # x is a tensor of shape [t b (fs c)]
    pred = prepare_pred(planner, x)
    total_T = pred.shape[0]

    # Extract observation part (first pos_dim dimensions for position)
    pred_obs = pred[:, :, planner.pos_dim_indices]  # Shape: [T, B, pos_dim]

    # Create indices for pairwise comparison
    indices = torch.arange(total_T, device=x.device)
    j_idx = indices.view(-1, 1)
    k_idx = indices.view(1, -1)

    # Sliding window mask: k is between [j-7-segment_size, j-7]
    ignore_latest = 6* planner.frame_stack
    pair_mask = (k_idx <= j_idx - ignore_latest) # & (k_idx >= j_idx - ignore_latest - segment_size)

    # Only apply to states within the planning horizon (after conditioning frames)
    planning_mask = (j_idx >= planner.frame_stack) & (
        j_idx < planner.frame_stack + horizon
    )
    pair_mask = pair_mask & planning_mask

    if not pair_mask.any():
        return x.sum() * 0.0  # connected to x so autograd.grad doesn't fail

    # Pairwise squared distances [B, T, T]
    pred_obs_b = pred_obs.transpose(0, 1)  # [B, T, 2]
    dist_sq = torch.cdist(pred_obs_b, pred_obs_b, p=2).pow(2)

    # RDF kernel matrix [B, T, T]
    h = 2.0  # bandwidth
    rdf_matrix = torch.exp(-dist_sq / h)

    # Apply mask: set invalid pairs to 0
    masked_rdf = rdf_matrix * pair_mask.unsqueeze(0).float()

    # For each j (dim 1), find mean of top 3 RDF among valid k's (dim 2)
    topk_rdf, _ = torch.topk(masked_rdf, k=3, dim=2)
    topk_rdf_mean_per_j = topk_rdf.mean(dim=2)

    # Average over j's that have at least one valid candidate k
    j_has_candidates = pair_mask.any(dim=1)
    if not j_has_candidates.any():
        return x.sum() * 0.0  # connected to x so autograd.grad doesn't fail

    mean_loss = topk_rdf_mean_per_j[:, j_has_candidates].sum()

    # Return negative loss (gradient descent will minimize repulsion)
    return -mean_loss

def particle_guidance(planner, x: torch.Tensor, horizon: int, group_ids: Optional[list] = None) -> torch.Tensor:
    """
    Particle diversity guidance via pairwise L2 repulsion at segment tail positions.

    Targets the same positions as goal_guidance (segment tail x,y coords), so gradient
    magnitudes are naturally unit-normalized — matching the scale of goal_guidance and
    anchor_dist_guidance.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor
        horizon: planning horizon in timesteps (same as passed to goal_guidance)
        group_ids: Optional list of length b. Elements sharing the same integer id belong
            to the same sibling group and repel each other. Cross-group pairs are ignored.
            If None, all pairs repel (original behaviour).

    Returns:
        loss: scalar negative mean L2 distance (minimizing pushes particles apart)
    """
    b = x.shape[1]
    if b <= 1:
        return x.sum() * 0.0

    pred = prepare_pred(planner, x)  # (t*fs, b, c) unnormalized
    T = pred.shape[0]
    segment_size = horizon // planner.sequence_dividing_factor

    # Tail positions: identical to goal_guidance causal mode
    tail_pos = torch.arange(
        planner.frame_stack + segment_size - 1,
        planner.frame_stack + horizon,
        segment_size,
        device=pred.device,
    )
    tail_pos = tail_pos[tail_pos < T]  # (n_tails,)

    if len(tail_pos) == 0:
        return x.sum() * 0.0

    pos_dim_indices = planner.pos_dim_indices
    # Extract (x,y) at tail positions: (n_tails, b, pos_dim) → (b, n_tails*pos_dim)
    tails = pred[tail_pos][:, :, pos_dim_indices]              # (n_tails, b, pos_dim)
    tails_flat = tails.permute(1, 0, 2).reshape(b, -1)        # (b, n_tails*pos_dim)

    # Pairwise L2 distance — gradient is unit-normalized by construction
    dist = torch.cdist(tails_flat, tails_flat, p=2)  # (b, b)

    if group_ids is not None:
        gids = torch.tensor(group_ids, device=x.device, dtype=torch.long)
        same_group = (gids.unsqueeze(0) == gids.unsqueeze(1)).float()
        pair_mask = same_group * (1.0 - torch.eye(b, device=x.device))
        n_pairs = pair_mask.sum()
        if n_pairs == 0:
            return x.sum() * 0.0
        mean_dist = (dist * pair_mask).sum() / n_pairs
    else:
        off_diag = 1.0 - torch.eye(b, device=x.device)
        mean_dist = (dist * off_diag).sum() / off_diag.sum()

    # Positive: DPS applies pred_noise -= grad, so pred_x_start moves in grad direction.
    # grad(mean_dist) w.r.t. pred_i points AWAY from j → repulsion (diversity).
    return mean_dist

def combined_guidance(planner, x_start, goal, horizon, guidance_scale,
                      particle_guidance_scale: float = 0.0,
                      group_ids: Optional[list] = None,
                      active_tail_per_batch: Optional[list] = None):
    """
    Combined guidance signals for diffusion model.

    Args:
        particle_guidance_scale: Scale for particle diversity guidance (0 = disabled).
            When > 0 and batch_size > 1, a repulsion loss pushes sibling trajectories apart.
        group_ids: Optional sibling group ids of length b (see particle_guidance).
            When provided, repulsion is applied only within groups of same id.
        active_tail_per_batch: When provided (smooth noise mode), list of per-batch
            frame-space indices indicating the single tail position to apply goal
            guidance to.  None → apply to all segment tails (causal mode default).

    Returns:
        guidance_dict: dict of guidance losses
    """
    anchor_guidance_scale = guidance_scale * planner.anchor_guidance_scale_ratio + 0.1
    anchor_loss = anchor_dist_guidance(planner, x_start, horizon) * anchor_guidance_scale 
    # NOTE: goal_guidance() returns HILP and RMSE components separately so each can be
    # scaled independently: TD_guidance_scale (via guidance_scale arg) for HILP,
    # rmse_guidance_scale for the far-target RMSE direction.
    _hilp_inner, _rmse_inner, _, _ = goal_guidance(
        planner, x_start, goal, horizon,
        active_tail_per_batch=active_tail_per_batch,
    )
    rmse_guidance_scale = getattr(planner, 'rmse_guidance_scale', 1.0)
    goal_loss = guidance_scale * _hilp_inner + rmse_guidance_scale * _rmse_inner
    rdf_loss = segment_rdf_guidance(planner, x_start, horizon) * planner.rdf_guidance_scale

    particle_loss = (
        particle_guidance(planner, x_start, horizon=horizon, group_ids=group_ids) * particle_guidance_scale
        if particle_guidance_scale > 0.0
        else x_start.sum() * 0.0  # zero but connected to x_start for autograd
    )

    result = {
        "anchor": anchor_loss,
        "goal": goal_loss,
        "rdf": rdf_loss,
    }
    if particle_guidance_scale > 0.0:
        result["particle"] = particle_loss
    return result
