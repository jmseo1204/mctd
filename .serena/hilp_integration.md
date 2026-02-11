# HILP Value Function Integration - UPDATED

## Overview
HILP (Hierarchical Imitation Learning via Preferences) value function integrated into MCTD's goal guidance mechanism with **proper 29-dimensional observation construction**.

## Latest Fix (2026-02-10)

### Problem Identified
- HILP was receiving `pred[:,:,:29]` but `pred` only contains **2 channels** (x,y positions)
- Dimensions 2-28 contained garbage values
- HILP needs full 29-dim AntMaze state (x, y, velocities, joint angles, etc.)

### Solution Implemented
Construct full 29-dimensional observations using **Zero-Padding** for non-x,y dimensions. This was chosen for its simplicity and stability after testing "combining with real state" which led to dimension mismatches in the complex planning pipeline.

```python
# Extract x,y from predicted trajectory
pred_xy = pred[:, :, :2]  # (T, B, 2)

# Zero-padding for remaining 27 dimensions
zeros_rest = torch.zeros((B, 27), device=pred.device)
zeros_rest_expanded = zeros_rest.unsqueeze(0).expand(T, -1, -1)

# Construct full observation: [x, y, 0, ..., 0]
pred_obs = torch.cat([pred_xy, zeros_rest_expanded], dim=-1) # (T, B, 29)
```

## Configuration

```yaml
# configurations/algorithm/df_planning.yaml
use_hilp_guidance: true  # Set to enable
hilp_checkpoint_path: "td_models/hilp_ckpt_latest.pt"
hilp_obs_dim: 29
hilp_skill_dim: 32
```

## Implementation Location

**File**: `algorithms/diffusion_forcing/df_planning.py`

1. **Loader** (L522-550): `get_hilp_value_fn()` lazy loader
2. **Observation Construction** (L570-603): Modified `goal_guidance` function

## Module Location

**cleandiffuser_ex**: `algorithms/cleandiffuser_ex/`
- Automatically imported from algorithms directory
- No external dependencies required

## Model Details

- **Architecture**: GoalConditionedPhiValue with ensemble (2 networks)
- **Value Formula**: `v = -||phi(s) - phi(g)||` (negative distance in phi space)
- **Ensemble**: `v = (v1 + v2) / 2`
- **Checkpoint**: PyTorch-converted from JAX

## Verification

✅ **Observation Construction**: Correctly combines pred x,y + start remaining dims  
✅ **Gradient Flow**: Gradients flow to input (norm: 6.88), HILP params frozen  
✅ **Backward Compatibility**: MSE guidance unchanged when `use_hilp_guidance: false`

## Usage

1. Set `use_hilp_guidance: true` in config
2. Model loads automatically on first guidance call
3. Monitor debug logs prefix: `[HILP DEBUG]`

## Files Modified

- `algorithms/diffusion_forcing/df_planning.py` - Observation construction logic
- `algorithms/cleandiffuser_ex/` - HILP module (copied from scots)
