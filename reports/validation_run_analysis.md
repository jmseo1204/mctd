# Log Analysis Report

- **File:** `validation_run.jsonl`
- **Generated:** 2026-03-26 19:46:00
- **Run ID:** unknown
- **Purpose:** bidirectional_mcts_tree_quality
- **Records:** 11639 | **Format:** SLP

---
## Errors

**Total errors:** 87

| Tag | Step | Message |
|-----|------|---------|
| exception.unhandled | None | 'list' object has no attribute 'shape' |
| exception.unhandled | None | too many indices for tensor of dimension 2 |
| exception.unhandled | None | mat1 and mat2 shapes cannot be multiplied (50x28 and 55x256) |
| exception.unhandled | None | name 'loops' is not defined |
| exception.unhandled | None | 'DiffusionForcingPlanning' object has no attribute 'segment_size' |
| exception.unhandled | None | too many indices for tensor of dimension 3 |
| exception.unhandled | None | plan_tokens 5 is not divisible by sequence_dividing_factor 2 |
| exception.unhandled | None | 'NoneType' object is not subscriptable |
| exception.unhandled | None | parent_node.sim_state is None |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | The size of tensor a (116) must match the size of tensor b (100) at non-singleto |
| exception.unhandled | None | expected np.ndarray (got Tensor) |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |
| exception.unhandled | None | CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronou |

_... and 67 more errors_
---
## Latency Analysis

> MPC planning time, MCTS phase breakdown, GPU denoising, guidance overhead, plan execution.

_No `timing.*` records found. Latency logging may be disabled or this log predates timing instrumentation._

---
## Plan-Following Analysis

### Execution Diagnostics (`exec.diag`)

> Tests H1 (DQL can't follow sub-goals) and H2 (open_loop_horizon too short)

**Total execution segments logged:** 20

| Metric | Value |
|--------|-------|
| Avg steps_executed / open_loop_horizon | 1.00 |
| Avg sub_goal_advances per segment | 13.2 |
| Avg mean_dist_to_subgoal | 6.76 |
| Avg mean_tracking_err (agent vs plan) | 7.65 |
| Segments with done_early=True | 0/20 |

#### Hypothesis Verdicts

**H1 (DQL performance issue): LIKELY** ⚠️
  - mean_dist_to_subgoal=6.76 > 3.0 → agent stays far from sub-goal
  - mean_tracking_err=7.65 > 5.0 → agent deviates far from plan

**H2 (open_loop_horizon too short): LIKELY** ⚠️
  - steps_executed/horizon=1.00 (hitting the limit) with sub_goal_advances=13.2 > 0 (agent WAS following)
  - Only 0/20 segments ended early (env done) → most are hitting horizon limit

#### Worst Segments (by mean_dist_to_subgoal)

| steps | advances | mean_dist | tracking_err | done_early |
|-------|----------|-----------|--------------|------------|
| 300/300 | 1 | 19.06 | 16.40 | False |
| 300/300 | 1 | 14.07 | 10.50 | False |
| 300/300 | 1 | 13.76 | 10.26 | False |
| 300/300 | 1 | 13.64 | 10.07 | False |
| 300/300 | 1 | 13.39 | 9.64 | False |
### Plan Start Drift (`mcts.start_drift`)

> Tests H3: does the new plan start from the current agent position?

| Metric | Value |
|--------|-------|
| Avg plan[0]→start dist | 0.50 |
| Max plan[0]→start dist | 0.85 |
| Records | 20 |

#### Hypothesis Verdict

**H3 (plan start drift): NOT SUPPORTED** ✓
  - avg drift=0.50 ≤ 2.0 → plan starts close to agent position
---
## Guidance Analysis

### Per-Tree Guidance Stats (`guidance.combined`)

> Goal guidance effectiveness, anchor/RDF balance, and final token distance to goal.

_No `guidance.combined` records found._

---
## MCTS Search Analysis

### Value / Achieved Rate

| Status | Count |
|--------|-------|
| Achieved | 3 |
| NotReached | 16 |
| Warp | 1 |
| Total | 20 |

**Achieved rate:** 15.8%

⚠️ Low achieved rate — bidir trees may not be meeting. Check `meeting_delta` and tree depth.
### Node Position Progression (`mcts.obs_pos`)

**Total OBS_POS records:** 20

**Tree: bidir_mcts_from_goal** (10 expansions)
  - depth ?→?: [44.0, 16.0] → [32.65505599975586, 23.789649963378906]
  - depth ?→?: [32.65505599975586, 23.789649963378906] → [22.582242965698242, 14.673992156982422]
  - depth ?→?: [22.582242965698242, 14.673992156982422] → [19.7766056060791, 5.59378719329834]
  - depth ?→?: [19.7766056060791, 5.59378719329834] → [28.477222442626953, 0.9040431976318359]
  - depth ?→?: [28.477222442626953, 0.9040431976318359] → [33.5102653503418, 7.8053083419799805]
  - depth ?→?: [28.0, 8.0] → [30.161334991455078, 8.902613639831543]
  - depth ?→?: [30.161334991455078, 8.902613639831543] → [29.089923858642578, 8.248261451721191]
  - depth ?→?: [29.089923858642578, 8.248261451721191] → [28.967723846435547, 8.399953842163086]
  - ... (2 more)

**Tree: bidir_mcts_from_start** (10 expansions)
  - depth ?→?: [8.0, 28.0] → [7.495183944702148, 16.24810791015625]
  - depth ?→?: [7.495183944702148, 16.24810791015625] → [1.9862632751464844, 8.95418643951416]
  - depth ?→?: [1.9862632751464844, 8.95418643951416] → [8.005144119262695, 0.010519027709960938]
  - depth ?→?: [8.005144119262695, 0.010519027709960938] → [26.901412963867188, 0.06621742248535156]
  - depth ?→?: [26.901412963867188, 0.06621742248535156] → [33.797672271728516, 7.826427459716797]
  - depth ?→?: [32.0, 16.0] → [25.550683975219727, 19.110530853271484]
  - depth ?→?: [25.550683975219727, 19.110530853271484] → [29.563383102416992, 17.196346282958984]
  - depth ?→?: [29.563383102416992, 17.196346282958984] → [31.588361740112305, 16.4451961517334]
  - ... (2 more)

---
## Tag Summary

| Tag | Count |
|-----|-------|
| `hilp.debug` | 2060 |
| `anchor.first_frame_diag` | 2000 |
| `gradient.component.anchor` | 2000 |
| `gradient.component.goal` | 2000 |
| `gradient.component.rdf` | 2000 |
| `run.start` | 368 |
| `run.end` | 356 |
| `mcts.phase` | 200 |
| `mcts.build_plan` | 100 |
| `exception.unhandled` | 87 |
| `mcts.bidir_debug` | 40 |
| `mcts.feasibility` | 40 |
| `mcts.values` | 40 |
| `mcts.extract_plan` | 38 |
| `tree.search.start` | 20 |
| `mcts.search_iter` | 20 |
| `mcts.selection` | 20 |
| `plan_start.obs_parent_xy` | 20 |
| `plan_start.before_denoising_xy` | 20 |
| `plan_start.after_denoising_xy` | 20 |
| `mcts.expansion` | 20 |
| `mcts.bidir_values` | 20 |
| `mcts.value_calc` | 20 |
| `mcts.early_term` | 20 |
| `tree.search.complete` | 20 |
| `mcts.obs_pos` | 20 |
| `mcts.start_drift` | 20 |
| `plan_start.plan0_vs_start` | 20 |
| `exec.diag` | 20 |
| `exec.env_dims` | 2 |
