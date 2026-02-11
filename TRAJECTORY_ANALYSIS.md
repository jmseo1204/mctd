"""
Analysis Report: Trajectory Visualization Issues
===============================================

## Problem:
- mcts_plan_from_start: path doesn't start from start, doesn't reach goal
- mcts_plan_from_goal: same issue
- White (start) and dark red (end) points scattered across map

## Root Causes Identified:

### ✅ FIX 1: Root node current_levels initialization (L1073-1075)
**BUG**: root_current_levels was grounding first and last indices
```python
root_current_levels[:, 0] = 0  # WRONG!
root_current_levels[:, -1] = 0  # WRONG!
```

**REASON**: current_levels only represents middle tokens.
Init/final tokens are handled separately via stabilization noise.

**FIX**: Removed the grounding lines.

### ✅ FIX 2: Noise level reference matrix (L1051-1054)
**BUG**: noise_level matrix was grounding first and last columns
```python
noise_level[:, 0] = 0  # WRONG!
noise_level[:, -1] = 0  # WRONG!
```

**REASON**: Matrix elements only represent middle tokens.

**FIX**: Removed the grounding lines.

## Additional Analysis:

### Plan Structure After parallel_plan():
1. **During processing** (inside parallel_plan):
   - plan shape: (tokens, batch, fs*c)
   - tokens = 1 (init) + plan_tokens (middle) + 1 (final) + padding
   
2. **After rearrange and slicing** (L597):
   ```python
   plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]
   ```
   - This REMOVES the init_token
   - Result: plan_hist starts from FIRST MIDDLE TOKEN

3. **For backward planning** (L601-602):
   ```python
   if self.bidirectional_search and not from_start:
       plan_hist = torch.flip(plan_hist, [1])
   ```
   - After flipping, plan_hist is reversed

### Visualization Code (L782-797):
```python
# Forward planning
forward_image = make_trajectory_images(
    self.env_id, 
    plan[:, :, :2].detach().cpu().numpy(),  # Uses ALL tokens
    1, start_numpy, goal_numpy, 
    self.plot_end_points
)[0]

# Reverse planning
reverse_image = make_trajectory_images(
    self.env_id, 
    reverse_plan[:, :, :2].detach().cpu().numpy(),
    1, goal_numpy[:, :2], start_numpy,  # SWAPPED start/goal
    self.plot_end_points
)[0]
```

### ⚠️ REMAINING ISSUES:

#### Issue 3: Plan includes init_token (maybe wrong)
At L640: `plan = plan_hist[-1]`
- plan_hist has shape (m, t, b, c) where t = horizon (after slicing)
- But does this correctly exclude init_token?
- L597 already removed init_token for plan_hist

#### Issue 4: Backward planning visualization
For `mcts_plan_from_goal` (from_start=False):
- At L767: `tag=\"mcts_plan_from_goal\", from_start=False`
- parallel_plan() flips the sequence during processing
- L602 flips it back
- But at L794, visualization uses: `goal_numpy[:, :2], start_numpy`
  - This SWAPS start and goal for visualization
  
**QUESTION**: After all the flipping, does the final plan have:
- plan[0] = start or goal?
- plan[-1] = goal or start?

## Next Steps for Debugging:

1. Add debug logging to trace actual plan values:
   ```python
   print(f"[DEBUG] plan shape: {plan.shape}")
   print(f"[DEBUG] plan[0]: {plan[0, 0, :2]}")  # First point
   print(f"[DEBUG] plan[-1]: {plan[-1, 0, :2]}")  # Last point
   print(f"[DEBUG] start: {start_numpy}")
   print(f"[DEBUG] goal: {goal_numpy[:, :2]}")
   ```

2. Verify make_trajectory_images() assumptions:
   - Does it assume plan[0] is the starting point?
   - Does it use start_numpy and goal_numpy for coloring only?

3. Test hypothesis: The random scattering might be caused by:
   - Wrong indexing (accessing wrong parts of plan)
   - Coordinate transformation issues
   - Normalization issues in visualization

## Summary:

✅ Fixed 2 confirmed bugs related to endpoint grounding
⚠️  Need to verify plan structure and backward planning visualization
📋 Next: Add debug logging to trace actual plan coordinates
"""

if __name__ == "__main__":
    print(__doc__)
