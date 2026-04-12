from typing import Optional, Union, Dict, Tuple, List
import math
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


def build_active_range_mask(
    candidate_pos: torch.Tensor,
    active_frame_ranges: Optional[List[Tuple[int, int]]],
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return a per-batch mask over candidate frame indices.

    Args:
        candidate_pos: (N,) candidate frame indices in full-sequence frame space.
        active_frame_ranges: optional list of half-open [start_f, end_f) pairs,
            one per batch item.
        batch_size: batch size B.
        device: target device.

    Returns:
        mask: (N, B) bool tensor when active_frame_ranges is provided, otherwise None.
    """
    if active_frame_ranges is None:
        return None

    assert len(active_frame_ranges) == batch_size, (
        f"active_frame_ranges length {len(active_frame_ranges)} != batch_size {batch_size}"
    )
    starts = torch.tensor([s for s, _ in active_frame_ranges], device=device, dtype=torch.long)
    ends = torch.tensor([e for _, e in active_frame_ranges], device=device, dtype=torch.long)
    pos = candidate_pos.to(device=device, dtype=torch.long).unsqueeze(1)  # (N, 1)
    return (pos >= starts.unsqueeze(0)) & (pos < ends.unsqueeze(0))  # (N, B)


def select_active_positions(
    candidate_pos: torch.Tensor,
    active_frame_ranges: Optional[List[Tuple[int, int]]],
    batch_size: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Select one in-range candidate position per batch item.

    Sliding-window guidance is expected to have exactly one semantically-active
    target position of each kind per batch item. When multiple candidates fall
    inside the active range, the right-most one is selected.
    """
    if active_frame_ranges is None:
        return candidate_pos.to(device=device, dtype=torch.long)

    mask = build_active_range_mask(candidate_pos, active_frame_ranges, batch_size, device)
    assert mask is not None
    counts = mask.sum(dim=0)
    if torch.any(counts == 0):
        raise ValueError(f"{name}: no candidate positions inside active_frame_ranges for some batch items")

    expanded = candidate_pos.to(device=device, dtype=torch.long).unsqueeze(1).expand(-1, batch_size)
    selected = torch.where(mask, expanded, torch.full_like(expanded, -1)).max(dim=0).values  # (B,)
    return selected


def get_segment_head_positions(
    planner,
    horizon: int,
    device: torch.device,
    total_frames: int,
) -> torch.Tensor:
    """Frame-space indices of each segment head within the predicted plan."""
    segment_size = horizon // planner.sequence_dividing_factor
    head_pos = torch.arange(
        planner.frame_stack,
        planner.frame_stack + horizon,
        segment_size,
        device=device,
    )
    return head_pos[head_pos < total_frames]


def get_segment_tail_positions(
    planner,
    horizon: int,
    device: torch.device,
    total_frames: int,
) -> torch.Tensor:
    """Frame-space indices of each segment tail within the predicted plan."""
    segment_size = horizon // planner.sequence_dividing_factor
    tail_pos = torch.arange(
        planner.frame_stack + segment_size - 1,
        planner.frame_stack + horizon,
        segment_size,
        device=device,
    )
    return tail_pos[tail_pos < total_frames]

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
    active_frame_ranges: Optional[List[Tuple[int, int]]] = None,
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
        active_frame_ranges: When provided, restrict guidance targets to candidate
            positions that fall inside the current sliding-window frame range for
            each batch item. None → apply to all segment tails (full-seq default).

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
        tail_pos = get_segment_tail_positions(planner, horizon, pred.device, T)

        if active_frame_ranges is not None:
            # Sliding-window mode: select the single in-range tail per batch item.
            _atail = select_active_positions(
                tail_pos, active_frame_ranges, B, pred.device, "goal_guidance/tail_pos"
            )  # (B,)
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
                "active_frame_ranges requires HILP guidance to be configured"

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
                    "active_range_mode": "per_batch",
                })

            # pred[_atail, _bidx, :2]: (B, 2) — gather active tail per batch item
            _pos = pred[_atail, _bidx][:, planner.pos_dim_indices]  # (B, pos_dim)
            hilp_pseudo_loss = (_pos * hilp_t * (~far_mask_t)).sum(dim=1)  # (B,)
            rmse_pseudo_loss = (_pos * rmse_t * far_mask_t).sum(dim=1)     # (B,)
            # Return per-batch (B,) tensors so combined_guidance can apply per-batch scaling.
            # .mean() here would mix batch elements and break guidance_scale=[0.4, 0.0] separation.
            return hilp_pseudo_loss, rmse_pseudo_loss, hilp_pseudo_loss.tolist(), rmse_pseudo_loss.tolist()

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

            # Return per-batch (B,) tensors so combined_guidance can apply per-batch scaling.
            # .mean() here would mix batch elements and break guidance_scale=[0.4, 0.0] separation.
            return hilp_pseudo_loss, rmse_pseudo_loss, hilp_pseudo_loss.tolist(), rmse_pseudo_loss.tolist()

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

def anchor_dist_guidance(
    planner,
    x: torch.Tensor,
    horizon: int,
    active_frame_ranges: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    Anchor distance regularization: pulls each segment head toward the end of the
    previous segment, enforcing temporal continuity between sub-plans.

    Index structure (mirrors goal_guidance's tail_pos pattern):
      head_pos[k]   = frame_stack + k * segment_size        (segment k의 첫 프레임)
      anchor_pos[k] = head_pos[k] - 1                       (직전 segment의 마지막 프레임)

    anchor_pos 값들은 goal_guidance의 tail_pos와 동일하며,
    맨 앞에 frame_stack - 1 (conditioning 마지막 프레임)이 하나 추가된 형태.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor
        horizon: planning horizon

    Returns:
        loss: scalar loss (DPS pushes segment heads toward anchors)
    """
    pred = prepare_pred(planner, x)
    pred_detached = pred.detach()
    T = pred.shape[0]

    segment_size = horizon // planner.sequence_dividing_factor

    # head_pos: 각 segment의 첫 프레임 (guidance 적용 대상)
    head_pos = get_segment_head_positions(planner, horizon, pred.device, T)

    # anchor_pos: head_pos - 1 = 직전 segment의 tail
    anchor_pos = head_pos - 1  # (n_segs,)

    if len(head_pos) == 0:
        return x.sum() * 0.0

    if active_frame_ranges is not None:
        selected_head_pos = select_active_positions(
            head_pos, active_frame_ranges, pred.shape[1], pred.device, "anchor_dist_guidance/head_pos"
        )  # (B,)
        selected_anchor_pos = selected_head_pos - 1
        bidx = torch.arange(pred.shape[1], device=pred.device)
        head_preds = pred[selected_head_pos, bidx][:, planner.pos_dim_indices]             # (B, pos_dim)
        anchor_refs = pred_detached[selected_anchor_pos, bidx][:, planner.pos_dim_indices] # (B, pos_dim)
        dist = ((head_preds - anchor_refs) ** 2).sum(dim=-1)  # (B,)
        dist = (dist + 1e-6).sqrt()
        return -dist  # (B,)

    # 명시적 인덱스로 head/anchor 위치만 추출
    head_preds  = pred[head_pos][:, :, planner.pos_dim_indices]             # (n_segs, B, pos_dim)
    anchor_refs = pred_detached[anchor_pos][:, :, planner.pos_dim_indices]  # (n_segs, B, pos_dim)

    dist = ((head_preds - anchor_refs) ** 2).sum(dim=-1)  # (n_segs, B)
    dist = (dist + 1e-6).sqrt()
    # sum over segments → (B,) per-batch loss.
    # combined_guidance multiplies by per-batch anchor_guidance_scale then .sum() → scalar.
    return -(dist.sum(dim=0))  # (B,)

def segment_rdf_guidance(
    planner,
    x: torch.Tensor,
    horizon: int,
    active_frame_ranges: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    """
    Within-segment repulsion: each segment's tail is repelled from its own head.

    Index structure (mirrors anchor_dist_guidance):
      head_pos[k] = frame_stack + k * segment_size        (first frame of segment k) — detached
      tail_pos[k] = frame_stack + (k+1) * segment_size - 1 (last frame of segment k) — gradient target

    Repulsive potential with short-range singularity and long-range exponential cutoff:
      U(d) = -C · exp(-d² / (2σ²)) / (d + eps)

    Let f(d) = exp(-d² / (2σ²)) / (d + eps).  For d > 0,
      f'(d) = exp(-d² / (2σ²)) · [ -d/(σ²(d+eps)) - 1/(d+eps)² ] < 0
    so
      U'(d) = -C f'(d) > 0   (for C > 0).

    Since ∇_tail d = (tail - head) / d, the tail gradient is
      ∇_tail U = U'(d) · (tail - head) / d
    which always points AWAY from the head.  Therefore the guidance is
    consistently repulsive for all d > 0.

    Additional properties:
      - d → 0: |∇_tail U| grows strongly due to the 1/(d+eps) singularity
      - d > σ : influence decays exponentially via exp(-d² / (2σ²))

    We choose C so that |∇_tail U| = 1 at d = σ.
    The radial derivative magnitude is
      U'(d) = C · exp(-d²/(2σ²)) · [ d/(σ²(d+eps)) + 1/(d+eps)² ]
    so at d = σ:
      1 = C · exp(-1/2) · [ 1/(σ(σ+eps)) + 1/(σ+eps)² ]
    hence
      C = exp(1/2) / [ 1/(σ(σ+eps)) + 1/(σ+eps)² ]
        = exp(1/2) · σ(σ+eps)² / (2σ + eps)

    After combined_guidance multiplies by rdf_guidance_scale, the effective
    gradient magnitude at d = σ becomes rdf_guidance_scale.
    After combined_guidance multiplies by rdf_guidance_scale, the effective
    gradient magnitude at d = σ becomes rdf_guidance_scale.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor
        horizon: planning horizon

    Returns:
        loss: scalar (normalized repulsion loss, to be scaled by rdf_guidance_scale externally)
    """
    pred = prepare_pred(planner, x)
    pred_detached = pred.detach()
    T = pred.shape[0]

    sigma = float(planner.rdf_sigma)
    segment_size = horizon // planner.sequence_dividing_factor

    # head_pos: identical to anchor_dist_guidance
    head_pos = get_segment_head_positions(planner, horizon, pred.device, T)

    # tail_pos: last frame of each segment
    tail_pos = head_pos + segment_size - 1

    n_segs = min(len(head_pos), int((tail_pos < T).sum()))
    if n_segs == 0:
        return x.sum() * 0.0

    head_pos = head_pos[:n_segs]
    tail_pos = tail_pos[:n_segs]

    if active_frame_ranges is not None:
        selected_head_pos = select_active_positions(
            head_pos, active_frame_ranges, pred.shape[1], pred.device, "segment_rdf_guidance/head_pos"
        )  # (B,)
        selected_tail_pos = selected_head_pos + segment_size - 1
        bidx = torch.arange(pred.shape[1], device=pred.device)
        head_refs = pred_detached[selected_head_pos, bidx][:, planner.pos_dim_indices]  # (B, pos_dim)
        tail_preds = pred[selected_tail_pos, bidx][:, planner.pos_dim_indices]           # (B, pos_dim)

        dist_sq = ((tail_preds - head_refs) ** 2).sum(dim=-1)  # (B,)
        dist = (dist_sq + 1e-12).sqrt()

        eps = 1e-6
        norm_factor = math.exp(0.5) * sigma * (sigma + eps) ** 2 / (2.0 * sigma + eps)
        rdf_loss = -norm_factor * torch.exp(-dist_sq / (2.0 * sigma ** 2)) / (dist + eps)  # (B,)
        return rdf_loss.sum()  # scalar

    # head is the fixed reference (detached); gradient flows through tail
    head_refs  = pred_detached[head_pos][:, :, planner.pos_dim_indices]  # (n_segs, B, pos_dim)
    tail_preds = pred[tail_pos][:, :, planner.pos_dim_indices]            # (n_segs, B, pos_dim)

    dist_sq = ((tail_preds - head_refs) ** 2).sum(dim=-1)  # (n_segs, B)
    dist = (dist_sq + 1e-12).sqrt()

    # Inverse-distance repulsion with Gaussian cutoff, normalized so
    # |∇_tail loss| = 1 at d = sigma.
    eps = 1e-6
    norm_factor = math.exp(0.5) * sigma * (sigma + eps) ** 2 / (2.0 * sigma + eps)
    rdf_loss = -norm_factor * torch.exp(-dist_sq / (2.0 * sigma ** 2)) / (dist + eps)  # (n_segs, B)

    # Sum over segments and batch for consistency with other summed guidance terms.
    return rdf_loss.sum()                                  # scalar

def particle_guidance(
    planner,
    x: torch.Tensor,
    horizon: int,
    group_ids: Optional[list] = None,
    active_frame_ranges: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
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
    tail_pos = get_segment_tail_positions(planner, horizon, pred.device, T)

    if len(tail_pos) == 0:
        return x.sum() * 0.0

    pos_dim_indices = planner.pos_dim_indices
    if active_frame_ranges is not None:
        selected_tail_pos = select_active_positions(
            tail_pos, active_frame_ranges, b, pred.device, "particle_guidance/tail_pos"
        )  # (B,)
        tails_flat = pred[selected_tail_pos, torch.arange(b, device=pred.device)][:, pos_dim_indices]  # (B, pos_dim)
    else:
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
        mean_dist = (dist * pair_mask).sum()
    else:
        off_diag = 1.0 - torch.eye(b, device=x.device)
        mean_dist = (dist * off_diag).sum()

    # Positive: DPS applies pred_noise -= grad, so pred_x_start moves in grad direction.
    # grad(mean_dist) w.r.t. pred_i points AWAY from j → repulsion (diversity).
    return mean_dist

def combined_guidance(planner, x_start, goal, horizon, guidance_scale,
                      particle_guidance_scale: float = 0.0,
                      group_ids: Optional[list] = None,
                      active_frame_ranges: Optional[List[Tuple[int, int]]] = None):
    """
    Combined guidance signals for diffusion model.

    Args:
        particle_guidance_scale: Scale for particle diversity guidance (0 = disabled).
            When > 0 and batch_size > 1, a repulsion loss pushes sibling trajectories apart.
        group_ids: Optional sibling group ids of length b (see particle_guidance).
            When provided, repulsion is applied only within groups of same id.
        active_frame_ranges: When provided, restrict all guidance target indices to
            positions inside the current sliding-window frame range for each batch item.

    Returns:
        guidance_dict: dict of guidance losses
    """
    # anchor_dist_guidance returns (B,) per-batch; scale per-batch then sum to scalar.
    # This ensures guidance_scale[b]=0.0 batch elements receive zero anchor gradient.
    anchor_guidance_scale = guidance_scale * planner.anchor_guidance_scale_ratio + 0.1  # (B,)
    anchor_loss = (
        anchor_dist_guidance(
            planner, x_start, horizon, active_frame_ranges=active_frame_ranges
        ) * anchor_guidance_scale
    ).sum()

    # goal_guidance returns (B,) per-batch tensors for hilp and rmse.
    # Scale per-batch so guidance_scale[b]=0.0 zeroes out HILP for that element.
    # rmse_guidance_scale is global (not per-batch) — applies equally to all elements.
    # NOTE: goal_guidance() returns HILP and RMSE components separately so each can be
    # scaled independently: TD_guidance_scale (via guidance_scale arg) for HILP,
    # rmse_guidance_scale for the far-target RMSE direction.
    _hilp_inner, _rmse_inner, _, _ = goal_guidance(
        planner, x_start, goal, horizon,
        active_frame_ranges=active_frame_ranges,
    )  # both (B,) per-batch tensors
    rmse_guidance_scale = getattr(planner, 'rmse_guidance_scale', 1.0)
    goal_loss = (guidance_scale * _hilp_inner + rmse_guidance_scale * _rmse_inner).sum()

    rdf_loss = (
        segment_rdf_guidance(
            planner, x_start, horizon, active_frame_ranges=active_frame_ranges
        ) * planner.rdf_guidance_scale
    )

    particle_loss = (
        particle_guidance(
            planner,
            x_start,
            horizon=horizon,
            group_ids=group_ids,
            active_frame_ranges=active_frame_ranges,
        ) * particle_guidance_scale
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
