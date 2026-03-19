# Log Analysis Report

- **File:** `20260318_201136_unknown.jsonl`
- **Generated:** 2026-03-19 05:14:18
- **Run ID:** unknown
- **Purpose:** plan_following_diagnosis
- **Records:** 52 | **Format:** SLP

---
## Errors

**Total errors:** 2

| Tag | Step | Message |
|-----|------|---------|
| exception.unhandled | None | cannot import name '_GuidanceJsonlLogger' from 'algorithms.diffusion_forcing.gui |
| exception.unhandled | None | cannot import name '_GuidanceJsonlLogger' from 'algorithms.diffusion_forcing.gui |
---
## Plan-Following Analysis

### Execution Diagnostics (`exec.diag`)

> Tests H1 (DQL can't follow sub-goals) and H2 (open_loop_horizon too short)

_No `exec.diag` records found. Add logging in `_execute_plan_in_env` and re-run._

### Plan Start Drift (`mcts.start_drift`)

> Tests H3: does the new plan start from the current agent position?

_No `mcts.start_drift` records found._

---
## Guidance Analysis

### Per-Tree Guidance Stats (`guidance.combined`)

> Goal guidance effectiveness, anchor/RDF balance, and final token distance to goal.

**Total guidance calls:** 2

#### FORWARD (2 calls)

| Metric | Mean |
|--------|------|
| eff_goal_scale | 4.0000 |
| anchor_loss (abs) | 0.00000 |
| goal_loss (abs) | 14.10736 |
| goal/anchor ratio | 1410735797.8821 |
| rdf_loss (abs) | 0.00000 |
| final_token_dist (last) | 4.40830 |

⚠️ goal/anchor ratio=1410735797.9 — goal dominates; consider increasing anchor_guidance_scale.

---
## MCTS Search Analysis

### Value / Achieved Rate

_No MCTS value records found._

### Node Position Progression (`mcts.obs_pos`)

_No `mcts.obs_pos` records found._

---
## Tag Summary

| Tag | Count |
|-----|-------|
| `mcts.build_plan` | 10 |
| `mcts.phase` | 6 |
| `hilp.debug` | 4 |
| `run.start` | 2 |
| `tree.search.start` | 2 |
| `mcts.search_iter` | 2 |
| `mcts.selection` | 2 |
| `mcts.bidir_debug` | 2 |
| `plan_start.obs_parent_xy` | 2 |
| `plan_start.before_denoising_xy` | 2 |
| `anchor.first_frame_diag` | 2 |
| `guidance.combined` | 2 |
| `gradient.component.anchor` | 2 |
| `gradient.component.goal` | 2 |
| `gradient.component.rdf` | 2 |
| `exception.unhandled` | 2 |
| `run.end` | 2 |
| `exec.env_dims` | 1 |
| `interact.envs_created.memory_stats` | 1 |
| `memory.envs` | 1 |
| `hilp.init` | 1 |
