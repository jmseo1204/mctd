from typing import Optional, Union, Dict
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat, reduce
from utils.tracer import get_tracer

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
    dist_o = dist_o[:, :, :2]
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
    return (dist * weight).mean(
        dim=dim
    )  # * dist.shape[1] DO NOT DELETE THIS COMMENT

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

def _tstat(t):
    """Return compact NaN/Inf/range stats for a tensor (detached)."""
    d = t.detach().float()
    return {
        "nan": int(torch.isnan(d).sum()),
        "inf": int(torch.isinf(d).sum()),
        "min": float(d.min()) if d.numel() > 0 else None,
        "max": float(d.max()) if d.numel() > 0 else None,
        "norm": float(d.norm()) if d.numel() > 0 else None,
    }

def goal_guidance(planner, x: torch.Tensor, goal: torch.Tensor, horizon: int, guidance_scale: torch.Tensor) -> torch.Tensor:
    """
    Target guidance to reach goal/start.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor from diffusion model
          where t=plan_tokens, b=batch_size, fs*c flattened observation
        goal: (b, obs_dim) normalized goal observations
        horizon: planning horizon in timesteps
        guidance_scale: (b,) guidance scales

    Returns:
        loss: scalar negative guidance loss for gradient ascent
    """
    _tracer = get_tracer()
    pred = prepare_pred(planner, x)
    segment_size = horizon // planner.sequence_dividing_factor
    # h_padded = (
    #     pred.shape[0] - planner.frame_stack
    # )  # include padding when horizon % frame_stack != 0

    if not planner.use_reward:
        # Temporal consistency guidance via shifted predictions
        # pred: (t fs) b c

        target = goal

        # Unnormalize target to match the scale of pred (which is unnormalized above)
        target = planner._unnormalize_x(target)

        target_guidance = torch.stack([target] * pred.shape[0])  # (t*fs, b, c)

        # Compute distance: either HILP value function or MSE
        T, B = pred.shape[0], pred.shape[1]

        # Only run HILP at positions where target_weight=1 to avoid
        # gradient explosion from extreme unnormalized noise in padding region.
        active_idx = torch.arange(
            planner.frame_stack - 1,
            planner.frame_stack + horizon,
            segment_size,
            device=pred.device,
        )  # e.g. [9, 59, 109]
        active_idx = active_idx[active_idx < T]

        obs_active   = pred[active_idx, :, :2].reshape(-1, 2)           # (len*B, 2)
        goal_active  = target[:, :2].unsqueeze(0).expand(
            len(active_idx), -1, -1
        ).reshape(-1, 2)                                                 # (len*B, 2)

        if _tracer is not None:
            with _tracer.scope("goal_guidance_nan_diagnosis", phase="guidance"):
                _tracer.log("goal_guidance.inputs", {
                    "pred":        _tstat(pred),
                    "target":      _tstat(target),
                    "obs_active":  _tstat(obs_active),
                    "goal_active": _tstat(goal_active),
                    "active_idx":  active_idx.tolist(),
                }, depth=1)

        v_active = planner._compute_hilp_values(
            obs_active, goal_active, use_no_grad=False
        ).reshape(len(active_idx), B)                                    # (len, B)

        # Fill full (T, B) tensor — non-active positions get 0 (detached, no grad)
        v = torch.zeros(T, B, device=pred.device, dtype=v_active.dtype)
        v[active_idx] = v_active

        # Convert value to distance: negate since v is negative distance
        TD_values = -v  # (T, B)
        TD_target_hilp = TD_values.unsqueeze(-1)  # (T, B, 1)

        # Replicate to match MSE output shape (T, B, C)
        TD_target = TD_target_hilp.expand(-1, -1, pred.shape[-1])

        # # Use traditional RMSE distance (commented out: causes grad explosion when MSE≈0)
        # RMSE_target = torch.sqrt(nn.functional.mse_loss(
        #     pred, target_guidance, reduction="none"
        # ) + 1e-8)

        dist_target = TD_target  # + RMSE_target

        target_weight = torch.zeros_like(pred.detach().cpu()[:, 0, 0])
        target_weight[
            planner.frame_stack-1 : planner.frame_stack + horizon : segment_size
        ] = 1
        target_weight = target_weight.float().to(planner.device)

        if _tracer is not None:
            with _tracer.scope("goal_guidance_nan_diagnosis", phase="guidance"):
                _tracer.log("goal_guidance.intermediates", {
                    "v_active":  _tstat(v_active),
                    "TD_values": _tstat(TD_values),
                    "TD_target": _tstat(TD_target),
                    # "RMSE_target": _tstat(RMSE_target),
                    "dist_target": _tstat(dist_target),
                }, depth=1)

        # weighted mean over time (dim=0) and feature (dim=2) → (b,)
        weighted_dist_target = weighted_loss(planner, dist_target, target_weight, dim=(0, 2))
        dist_per_batch = guidance_scale * weighted_dist_target

        if _tracer is not None:
            with _tracer.scope("goal_guidance_nan_diagnosis", phase="guidance"):
                _tracer.log("goal_guidance.output", {
                    "weighted_dist": _tstat(weighted_dist_target),
                    "dist_per_batch": _tstat(dist_per_batch),
                    "loss": float(-(dist_per_batch).mean().detach()),
                }, depth=1)

        print(f"Dist per batch: {dist_per_batch.tolist()}")
        print(f"Scales: {guidance_scale.tolist()}")

    else:
        raise NotImplementedError(
            "reward guidance not officially supported yet, although implemented"
        )

    return -(dist_per_batch).mean()

def anchor_dist_guidance(planner, x: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    Anchor distance regularization using segment heads.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor
        horizon: planning horizon

    Returns:
        loss: scalar negative regularization loss
    """
    tracer = get_tracer()

    # x is a tensor of shape [t b (fs c)]
    pred = prepare_pred(planner, x)
    h_padded = (
        pred.shape[0] - planner.frame_stack
    )  # include padding when horizon % frame_stack != 0

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

    tail_of_each_segments = pred_detached[
        planner.frame_stack : planner.frame_stack + horizon : segment_size
    ]
    spread_plan = torch.zeros_like(pred_detached)
    spread_plan[
        planner.frame_stack + segment_size - 1 : planner.frame_stack + horizon : segment_size
    ] = tail_of_each_segments
    dist_spread = nn.functional.mse_loss(pred, spread_plan, reduction="none")


    anchor_weight = torch.zeros_like(pred_detached[:, 0, 0])
    anchor_weight[
        planner.frame_stack : planner.frame_stack + horizon : segment_size
    ] = 1

    anchor_weight[
        planner.frame_stack - 1 : planner.frame_stack + horizon - 1 : segment_size
    ] = 1

    anchor_spread = dist_anchor - dist_spread

    weighted_dist_anchor = weighted_loss(planner, anchor_spread, anchor_weight)

    loss = -(weighted_dist_anchor).mean()

    if tracer is not None:
        with tracer.scope("anchor_guidance_nan_check", phase="guidance"):
            tracer.log_tensor_stats("anchor_guidance.pred",          pred.detach(),                depth=1)
            tracer.log_tensor_stats("anchor_guidance.dist_anchor",   dist_anchor.detach(),         depth=1)
            tracer.log_tensor_stats("anchor_guidance.dist_spread",   dist_spread.detach(),         depth=1)
            tracer.log_tensor_stats("anchor_guidance.anchor_spread", anchor_spread.detach(),       depth=1)
            tracer.log_tensor_stats("anchor_guidance.weighted_dist", weighted_dist_anchor.detach(),depth=1)
            tracer.log_tensor_stats("anchor_guidance.loss",          loss.detach(),                depth=1)
            tracer.log(
                "anchor_guidance.nan_inf_summary",
                {
                    "pred_has_nan":          bool(torch.isnan(pred).any()),
                    "pred_has_inf":          bool(torch.isinf(pred).any()),
                    "dist_anchor_has_nan":   bool(torch.isnan(dist_anchor).any()),
                    "dist_anchor_has_inf":   bool(torch.isinf(dist_anchor).any()),
                    "dist_spread_has_nan":   bool(torch.isnan(dist_spread).any()),
                    "dist_spread_has_inf":   bool(torch.isinf(dist_spread).any()),
                    "anchor_spread_neg_frac": float((anchor_spread < 0).float().mean()),
                    "weighted_dist_has_nan": bool(torch.isnan(weighted_dist_anchor).any()),
                    "weighted_dist_has_inf": bool(torch.isinf(weighted_dist_anchor).any()),
                    "loss_has_nan":          bool(torch.isnan(loss)),
                    "loss_has_inf":          bool(torch.isinf(loss)),
                    "loss_value":            float(loss.detach()),
                    "anchor_guidance_scale": float(planner.anchor_guidance_scale),
                },
                depth=1,
            )

    return loss

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

    # Extract observation part (first 2 dimensions for position)
    pred_obs = pred[:, :, :2]  # Shape: [T, B, 2]

    # Calculate segment size for window width
    segment_size = float("inf")

    # Create indices for pairwise comparison
    indices = torch.arange(total_T, device=x.device)
    j_idx = indices.view(-1, 1)
    k_idx = indices.view(1, -1)

    # Sliding window mask: k is between [j-7-segment_size, j-7]
    ignore_latest = 3* planner.frame_stack
    pair_mask = (k_idx <= j_idx - ignore_latest) # & (k_idx >= j_idx - ignore_latest - segment_size)

    # Only apply to states within the planning horizon (after conditioning frames)
    planning_mask = (j_idx >= planner.frame_stack) & (
        j_idx < planner.frame_stack + horizon
    )
    pair_mask = pair_mask & planning_mask

    if not pair_mask.any():
        return torch.tensor(0.0, device=x.device, requires_grad=True)

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
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    mean_loss = topk_rdf_mean_per_j[:, j_has_candidates].sum()

    # Return negative loss (gradient descent will minimize repulsion)
    return -mean_loss

def particle_guidance(planner, x: torch.Tensor) -> torch.Tensor:
    """
    Particle diversity guidance based on RBF kernel similarity.

    Args:
        planner: The DiffusionForcingPlanning instance
        x: (t, b, fs*c) normalized prediction tensor

    Returns:
        loss: scalar negative diversity loss
    """
    b = x.shape[1]
    if b <= 1:
        return x.sum() * 0.0

    x_flat = rearrange(
        x, "t b (fs c) -> b (t fs c)", fs=planner.frame_stack
    )  # (b, t*fs*c)

    # Shape: [b, b]
    dist_sq = torch.cdist(x_flat, x_flat, p=2).pow(2)

    h = torch.median(dist_sq.detach())
    if h == 0:
        h = 1.0  # Fallback to avoid division by zero

    kernel_matrix = torch.exp(-dist_sq / h)

    similarity = (kernel_matrix.sum() - b) / (b * (b - 1))

    return -similarity

def combined_guidance(planner, x_start, goal, horizon, guidance_scale):
    """
    Combined guidance signals for diffusion model.

    Returns:
        guidance_dict: dict of guidance losses
    """
    return {
        "anchor": anchor_dist_guidance(planner, x_start, horizon) * planner.anchor_guidance_scale,
        "goal": guidance_scale * goal_guidance(planner, x_start, goal, horizon, guidance_scale),
        "rdf": segment_rdf_guidance(planner, x_start, horizon) * planner.rdf_guidance_scale,
    }
