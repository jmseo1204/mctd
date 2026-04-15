# Parallel Search Regardless of Tree

## Goal

Refactor the `use_uncertainty_as_value=true` bidirectional planning path so that:

- parent nodes are selected globally across both trees by value
- the selected parents are expanded in one mixed batch, regardless of tree
- logic that is tree-agnostic is executed once on the mixed batch
- logic that must remain tree-local is executed afterward per tree
- `tree.search_num` and `tree.p_search_num` are removed entirely
- `mctd_max_search_num` is interpreted as a global parent-selection cap, not a child-count cap

This document is the concrete implementation plan for that refactor.

## Desired Semantics

### Selection / Budget

- `mctd_max_search_num` means the maximum number of selected parents in one planning episode.
- The cap is global, not per tree.
- One selected parent consumes exactly `1` unit of budget.
- Child multiplicity does not affect budget.
- If one selected parent generates 3 children in a round, the global search count still increases by `1`.
- `parallel_search_node` already matches this interpretation and should remain the per-round parent-selection budget.
- Mixed-parent one-shot expansion applies only when `use_uncertainty_as_value=true`.

### Dedup

- Dedup is applied only within the same tree and the same parent group.
- Dedup must never compare siblings that came from different trees.

### Logging

- Mixed-only expansion logic may log at mixed-batch granularity.
- Tree-local rollout / visualization / post-expansion handling remains tree-specific.

## Current State

### Already Implemented

The following pieces already exist and should be reused:

1. Global candidate collection and ranking
   - `_collect_global_expansion_candidates()`
   - `_select_global_expansion_parents()`

2. Parent-count round budget
   - `_parent_selection_budget()`
   - This already returns a parent-selection budget, not a child budget.

3. Uncertainty / cluster expansion primitives
   - `_run_fast_uncertainty_sampling()`
   - `_compute_uncertainty_and_clusters()`
   - `_ensure_uncertainty_roots_initialized()`
   - `_init_root_node_uncertainty()`

4. Existing tree-local post-expansion helpers
   - `_update_expanded_children_state()`
   - `_log_expanded_node_videos()`

5. Most of the intended mixed expansion body already exists inside `_run_mcts_search()`
   - cluster reuse
   - mixed denoising / replanning
   - uncertainty sampling
   - value computation
   - endpoint dedup
   - child allocation

### Missing / Incomplete

The following are currently missing or only half-wired:

1. `_run_global_uncertainty_expansion_round()`
   - called, but not defined

2. `_postprocess_tree_local_expansions()`
   - called, but not defined

3. Global parent budget is not yet the true stopping condition
   - `global_search_num` exists, but stopping logic still depends on tree-local counters

4. `tree.search_num` and `tree.p_search_num` are still deeply wired into `_run_mcts_search()`
   - these must be removed, not merely ignored

5. The mixed path producer does not yet attach the metadata needed by downstream logic
   - `selected_tree`
   - `opposite_tree`
   - `parent_key`

## Refactor Boundary

The key split is:

### A. Tree-Agnostic Mixed Expansion Core

This is the logic that should move into `_run_global_uncertainty_expansion_round()`.

It includes:

- receiving globally selected parents
- building a mixed candidate batch
- choosing dynamic targets per candidate from that candidate's opposite tree
- running cluster-reuse expansion if needed
- running standard denoising expansion otherwise
- replanning / feasibility filtering
- fast uncertainty sampling
- value computation from uncertainty
- endpoint dedup within `(tree, parent)` groups
- child allocation via `parent.expand(...)`
- producing global `expanded_node_infos`
- producing tree-grouped outputs for later tree-local postprocessing

This corresponds roughly to the post-selection body currently inside `_run_mcts_search()`:

- candidate preprocessing
- cluster reuse block
- expansion / replan / value / dedup / allocation
- but **not** backprop, tree counters, or tree-local rollout / video handling

### B. Tree-Local Postprocessing

This is the logic that should move into `_postprocess_tree_local_expansions()`.

It includes:

- grouping results by tree
- backpropagation on parents from that tree
- updating tree-local state such as:
  - `max_depth`
  - `achieved`
- updating child `obs` / `sim_state`
- tree-local visualization and video logging
- tree-local timing / diagnostic logs that genuinely depend on one tree

This helper must not do any mixed candidate generation or mixed denoising work.

## Data Model Changes

### Remove from `MCTSTreeState`

Delete these fields entirely:

- `search_num`
- `p_search_num`
- `max_search_num`

Reason:

- `search_num` and `p_search_num` are no longer valid concepts after the global parent-count interpretation change.
- `max_search_num` is no longer tree-local; it is planner-global.

### Keep in `MCTSTreeState`

Keep tree-local state that still makes sense:

- `root_node`
- `plan_tokens`
- `terminal_depth`
- `children_node_guidance_scales`
- `skip_level_steps`
- `tag`
- `is_tree1`
- `tree_root_obs`
- `max_depth`
- `achieved`
- `pbar` only if still useful for non-uncertainty path
- timing lists, if still used

### Planner-Global State

The global budget should be held only on the planner instance:

- `self.mctd_max_search_num`
- `self.global_search_num`

Interpretation:

- `self.global_search_num` = number of selected parents already consumed in this episode

## New Helper Contracts

### `_run_global_uncertainty_expansion_round(...)`

Suggested signature:

```python
def _run_global_uncertainty_expansion_round(
    self,
    selected_parent_infos: List[dict],
    horizon: int,
    conditions: Optional[Any],
) -> dict:
```

Suggested return shape:

```python
{
    "expanded_node_infos": dict[str, dict],
    "tree_batches": dict[str, dict],
    "mixed_stats": dict,
}
```

Where:

- `expanded_node_infos`
  - global flat map used by `_select_best_leaf()`
  - each info must contain `selected_tree`

- `tree_batches`
  - keyed by tree tag or tree identity
  - contains only the per-tree subset needed by `_postprocess_tree_local_expansions()`

- `mixed_stats`
  - optional mixed-level timing / count diagnostics

Each mixed candidate dict must carry:

- `parent_node`
- `selected_tree`
- `opposite_tree`
- `target_node`
- `parent_key = f"{selected_tree.tag}:{parent_node.name}"`
- `selection_count`

### `_postprocess_tree_local_expansions(...)`

Suggested signature:

```python
def _postprocess_tree_local_expansions(
    self,
    tree_batches: dict[str, dict],
    agent,
    envs,
    start: torch.Tensor,
    goal: torch.Tensor,
    loops: int,
) -> None:
```

Per-tree batch should contain at least:

- `tree`
- `expanded_node_infos`
- `selected_parent_nodes`
- any tree-local stats needed for logging

This helper should:

1. backpropagate only parents from that tree that actually created children
2. update `tree.max_depth`
3. update `tree.achieved`
4. call `_update_expanded_children_state(...)`
5. call `_log_expanded_node_videos(...)`

This helper should **not**:

- do mixed denoising
- recompute values
- touch global budget

## Exact Counter / Stopping Changes

### Global Stop Condition

In the mixed uncertainty path, the planning loop stop condition should become:

- stop if `global_search_num >= mctd_max_search_num`
- stop if no globally selectable parents remain
- stop if both roots become unexpandable
- stop if meeting condition is satisfied
- stop if `val_max_loops` or time limit is reached

### Parent Selection Limit per Round

At each round:

1. compute `round_budget = _parent_selection_budget()`
2. compute `remaining_budget = self.mctd_max_search_num - self.global_search_num`
3. final round selection cap is `min(round_budget, remaining_budget)`
4. `_select_global_expansion_parents()` should respect that cap before returning the final selected parent list

### Remove Child-Based Budgeting

Delete all semantics that treat child count as search budget:

- no `tree.p_search_num += len(expanded_node_candidates)`
- no stopping on a child count
- no logging that presents child count as search budget progress

Child counts may still be logged as a descriptive statistic if needed, but not as a search counter and not via `tree.p_search_num`.

## Detailed Implementation Steps

### Step 1. Remove tree-local search counters from the type and logs

In `MCTSTreeState`:

- remove `search_num`
- remove `p_search_num`
- remove `max_search_num`

Then update all call sites in `df_planning.py` that currently use them, including:

- loop stop condition
- profiler snapshots
- guidance logs
- timing logs
- visualization calls
- final search summary print

Likely replacements:

- mixed path: use `self.global_search_num`
- non-uncertainty path: if a local counter is still needed, compute it as a local variable inside `_run_mcts_search()` rather than storing it on the tree

### Step 2. Split `_run_mcts_search()` into three conceptual layers

Keep `_run_mcts_search()` for the old tree-local path, but restructure it internally:

1. selection layer
2. tree-agnostic expansion core
3. tree-local tail

Target outcome:

- the mixed uncertainty path reuses layer 2 and layer 3 helpers directly
- the legacy tree-local path can still call the same helpers with one-tree input

### Step 3. Extract a tree-agnostic expansion core helper

Create a new internal helper that takes a candidate batch plus tree metadata and performs:

- candidate validation
- cluster-reuse preprocessing
- dynamic goal selection
- expansion
- replanning
- uncertainty sampling
- value calculation
- dedup
- child allocation

This helper should not assume that all candidates belong to one tree.

This helper should output:

- global `expanded_node_infos`
- per-tree grouped `expanded_node_infos`
- per-tree `selected_parent_nodes`
- per-tree `achieved` flags / max depth data

If preferred, this core helper may be the actual implementation body behind `_run_global_uncertainty_expansion_round()`.

### Step 4. Implement `_run_global_uncertainty_expansion_round()`

This function should:

1. receive `selected_parent_infos`
2. convert them into mixed candidates
3. attach:
   - `selected_tree`
   - `opposite_tree`
   - `parent_key`
4. invoke the extracted mixed expansion core
5. annotate each returned leaf info with `selected_tree`
6. return:
   - flat `expanded_node_infos`
   - `tree_batches`
   - optional mixed stats

Important:

- do not split into per-tree `_run_mcts_search()` calls
- do not re-run selection inside this helper
- selection was already done globally before this function is called

### Step 5. Implement `_postprocess_tree_local_expansions()`

This function should iterate over `tree_batches`.

For each tree:

1. find distinct selected parents for that tree
2. backpropagate only parents that actually produced children
3. update `tree.max_depth`
4. update `tree.achieved`
5. call `_update_expanded_children_state(...)`
6. call `_log_expanded_node_videos(...)`
7. emit tree-local timing logs if still needed

This preserves the tree boundary where it still matters, while keeping denoising/value work mixed.

### Step 6. Rewire the main validation loop

Inside the mixed uncertainty path in the main planning loop:

1. select globally ranked parents
2. clamp by remaining global budget
3. increment `self.global_search_num` by the number of selected parents
4. call `_run_global_uncertainty_expansion_round(...)`
5. call `_postprocess_tree_local_expansions(...)`
6. select best leaf from mixed `expanded_node_infos`
7. continue meeting / execution flow as before

The old inlined per-tree expansion block should disappear.

### Step 7. Keep non-uncertainty path behavior stable

The refactor should not change behavior when `use_uncertainty_as_value=false`.

That path may keep `_run_mcts_search()` as a single-tree loop, but it must also stop depending on deleted `tree.search_num` / `tree.p_search_num`.

If iteration numbering is still needed there for logging, keep it as a local variable inside `_run_mcts_search()`.

## Dedup Requirements

The current direction is correct:

- dedup within same tree
- dedup within same parent group

To preserve this in the mixed helper:

- every candidate must have `selected_tree`
- every candidate must have `parent_key`
- `_deduplicate_by_endpoint()` should continue grouping by `parent_key`

No cross-tree dedup should occur.

## Logging / Visualization Migration

Because `tree.search_num` is removed, these logs need updated semantics:

### Mixed logs

Use one of:

- `self.global_search_num`
- per-round mixed index
- explicit parent count in current round

Applicable to:

- mixed guidance logs
- mixed diffusion / replanning timing
- mixed uncertainty timing

### Tree-local logs

Use:

- `loops`
- tree tag
- current global parent count
- local round parent count

Applicable to:

- rollout update logs
- node video logging
- final tree summaries if still kept

### Visualization functions

If a function still takes `search_num`, rename the parameter to something semantically valid, such as:

- `round_idx`
- `global_parent_count`
- `viz_step`

Do not keep the old name once `tree.search_num` is removed.

## Concrete Removal List

The following usages must be removed or rewritten as part of this refactor:

1. `MCTSTreeState.search_num`
2. `MCTSTreeState.p_search_num`
3. `MCTSTreeState.max_search_num`
4. stop condition based on `tree.p_search_num >= tree.max_search_num`
5. all increments of `tree.search_num`
6. all increments of `tree.p_search_num`
7. logs that report those counters as search progress

## Validation Plan

After implementation:

1. Run a reduced eval with `use_uncertainty_as_value=true`.
2. Confirm the previous `AttributeError` is gone.
3. Confirm parent budget is consumed globally, not per tree.
4. Confirm a parent that spawns multiple children still consumes only one unit of global budget.
5. Confirm dedup never kills across different trees.
6. Confirm rollout state update and visualization still happen tree-locally.
7. Confirm the non-uncertainty path still runs.

Suggested checks:

- reduced Hydra eval command with `wandb.mode=offline`
- any focused internal debug script already used for planning / guidance validation

## Recommended Implementation Order

1. delete `tree.search_num`, `tree.p_search_num`, `tree.max_search_num`
2. make non-uncertainty `_run_mcts_search()` compile again using local loop counters only
3. extract tree-agnostic mixed expansion core from `_run_mcts_search()`
4. implement `_run_global_uncertainty_expansion_round()`
5. implement `_postprocess_tree_local_expansions()`
6. rewire the mixed uncertainty path to call the two helpers
7. update logs / visualization parameter names
8. run reduced validation

## Non-Goals

This refactor does not aim to:

- change the semantics of `use_uncertainty_as_value=false`
- redesign uncertainty estimation itself
- redesign cluster subplan generation
- introduce cross-tree dedup

## Summary

The intended design is:

- global value-based parent selection
- one mixed expansion batch for the selected parents
- tree-local rollout / visualization / backprop afterward
- one global parent-count budget
- no tree-local search counters at all

The missing part is not the algorithmic core itself. Most of that core already exists inside `_run_mcts_search()` and needs to be extracted cleanly into the mixed helper and the tree-local helper described above.
