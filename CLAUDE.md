# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **Diffusion Forcing for Trajectory Planning** (MCTD). It uses a transformer-based diffusion model with MCTS-guided planning in offline RL environments (OGBench: AntMaze, PointMaze).

**Environment note**: The host environment has minimal packages. All project dependencies are in Docker image `mctd:0.1`. Training runs via `train.sh` and evaluation via `eval.sh` (both Docker-based).

## Architecture Overview

### High-Level Flow

```
main.py (Hydra entry point)
  ↓
experiments/build_experiment() → experiment instance (exp_planning, exp_video, etc.)
  ↓
algorithm instance (df_planning, df_video, etc.) with DictConfig
  ↓
dataset instance (OGBench environments) with DictConfig
```

### Core Algorithm: DiffusionForcingPlanning (`df_planning.py`)

This is the primary file (~5500 lines). It inherits from `DiffusionForcingBase` (`df_base.py`) and two mixins: `PlanPostprocMixin` (`plan_postproc.py`) and `PlanVizMixin` (`plan_viz.py`). **Treat it as authoritative — it overrides `df_base.py` behavior.**

Key entry points and call chain:

```
interact() → p_mctd_plan() → _run_mcts_search() → _extract_output_plan()
                                                 → _execute_plan_in_env()
```

- `interact()`: Resets env, captures initial_sim_state, initializes bidirectional MCTS trees, runs search, executes plan
- `_run_mcts_search()`: Iterative tree expansion with diffusion guidance
- `_execute_plan_in_env()`: Loops `open_loop_horizon` steps; tracks sim_state, computes actions, steps env
- `_extract_output_plan()`: In `plan_postproc.py` — concatenates FWD+BWD plan segments based on tree depth and `sequence_dividing_factor`

**Plan segment extraction:**

```
seg_size = plan_tokens // sequence_dividing_factor
used_len = depth * seg_size * frame_stack
plan_a_full[used_len:] = unused (contains undenoised noise)
```

### Mixin Modules

- **`plan_postproc.py` (`PlanPostprocMixin`)**: Plan post-processing — depth-based prefix length, boundary obs extraction (`_extract_obs_at_boundary`), endpoint deduplication (`_deduplicate_by_endpoint`), FWD+BWD gap computation (`_compute_plan_gap`), proximity reordering (`_reorder_plan_by_proximity`), and final output plan construction (`_extract_output_plan`).

- **`plan_viz.py` (`PlanVizMixin`)**: Visualization — HILP value heatmap (`_compute_hilp_heatmap`), guidance gradient field (`_compute_guidance_grad_fields`), MCTS node-path extraction (`_extract_node_trajectory`), and per-candidate denoising video logging (`_log_candidate_plan_video`).

- **`uncertainty_estimator.py`**: Three uncertainty modes — radial-angular entropy (`compute_uncertainty_from_embeddings`), temporal-distance variance decomposition (`compute_uncertainty_variance`), and complete-linkage hierarchical clustering (`cluster_tail_by_temporal_dist`). Selected via `uncertainty_mode` config.

### Key Tensor Dimensions

Always verify when modifying `_construct_sequence()` or value computation:

- Observations: `(batch, n_tokens, obs_dim)` after frame stacking
- Actions: `(batch, n_tokens-1, action_dim)` — **one fewer token than observations**
- Noise levels: `(batch, n_tokens)` aligned with diffusion timesteps

### Bidirectional MCTS

- `tree1` = forward planning, `tree2` = backward planning (uses separate envs)
- `is_unknown_final_token` and `sequence_dividing_factor` must be consistent
- Value functions: `calculate_values()`, `calculate_values_bidir()`
- When `use_uncertainty_as_value=true`: uncertainty score (from `uncertainty_estimator.py`) replaces standard value; `uncertainty_mode` selects which estimator
- When `use_cluster_subplan_as_expansion=true`: per-cluster subplans from uncertainty sampling (`TreeNode.cluster_subplans`) are reused as child node expansions

### Sim State

Physical state dict with keys `qpos` (joint positions) and `qvel` (joint velocities). `qpos[:2]` is always `(x, y)` world coordinates. When `use_rollout=false`, synthetic sim_states are created by copying parent and updating `qpos[:2]` from plan position.

### Configuration System (Hydra)

Root config: `configurations/config.yaml`. The algorithm config is now a **three-layer hierarchy**:

1. **`ckpt_df_planning.yaml`** — ckpt-bound model definition (architecture, data identity, dimension indices). Saved into checkpoint `training_hparams`. Edit this to change what model you train.
2. **`train_df_planning.yaml`** — training-only params (optimizer, dataset stats, batch size). Composes `ckpt_df_planning` via Hydra defaults.
3. **`df_planning.yaml`** — eval-only (planning params, guidance scales, uncertainty, visualization). Fully self-contained; does **not** extend the train config.

At eval time, `_apply_ckpt_hparams_to_cfg()` overwrites null stubs in `df_planning.yaml` from the loaded checkpoint's `training_hparams`.

## Common Development Commands

**Training (Docker):**

```bash
bash train.sh   # menu: select state dim (2D/15D/29D), jump value, resume checkpoint
```

**Evaluation (Docker):**

```bash
bash eval.sh   # selects state dim → checkpoint → generates jobs → runs via Docker
```

**Resume / load checkpoint (inside Docker):**

```bash
python main.py resume=<wandb_run_id> experiment.tasks=[training]
python main.py load=<wandb_run_id> experiment.tasks=[validation]
```

**Debugging inside Docker (reduced scope):**

```bash
python main.py +name=test algorithm=df_planning mctd_max_search_num=10 parallel_search_num=1 wandb.mode=offline
```

**Test scripts (root directory):**

```bash
python test_padding_consistency.py   # validate padding mode changes
python test_gradient_flow.py
python test_trajectory_diagnostics.py
```

**Analysis:**

```bash
bash latency_analysis.sh                            # timing & job time breakdown
bash guidance_analysis.sh                            # guidance quality & MCTS debug
python scripts/generate_jobs_generalized.py         # generate eval job specs
python scripts/run_jobs.py                          # execute job specs
```

## Key Parameters (df_planning.yaml)

| Parameter                          | Purpose                                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| `sampling_timesteps`               | DDIM steps per generation (≤ `timesteps` from ckpt)                                         |
| `mctd_max_search_num`              | Max MCTS tree expansions                                                                     |
| `parallel_search_node`             | Parallel nodes; `parallel_search_num = parallel_search_node * len(mctd_guidance_scales)`    |
| `mctd_guidance_scales`             | List of HILP guidance scales for exploration                                                 |
| `sequence_dividing_factor`         | Segments per planning level (must match `is_unknown_final_token`)                            |
| `segment_episode_len`              | Raw episode length per segment; effective tokens = `(segment_episode_len * sdf // jump) // frame_stack + 1` |
| `open_loop_horizon`                | Steps executed per plan                                                                      |
| `val_max_loops`                    | Max rollout loops per validation episode                                                     |
| `meeting_delta`                    | Bidirectional tree convergence threshold (world units)                                       |
| `padding_mode`                     | `"same"` or `"zero"` — changing requires `test_padding_consistency.py`                      |
| `use_rollout`                      | If `false`, derive `obs` from plan_history (skip physical sim)                               |
| `leaf_parallelization`             | Parallelize leaf node evaluation                                                             |
| `uncertainty_mode`                 | `'entropy'` \| `'variance'` \| `'cluster'` — selects uncertainty estimator                  |
| `use_uncertainty_as_value`         | Replace MCTS value with uncertainty score                                                    |
| `use_cluster_subplan_as_expansion` | Reuse per-cluster subplans from uncertainty sampling as child expansions                     |
| `fast_sampling_multiple`           | Fast samples per candidate for uncertainty estimation                                        |
| `fast_sampling_steps`              | Denoising steps for fast uncertainty sampling (< `sampling_timesteps`)                       |
| `use_directly_inject_guidance_to_x0` | Apply guidance in clean x0 space instead of DPS-style xt space                            |
| `use_segment_wise_sliding_window`  | Use segment-wise DDIM sliding window                                                         |
| `use_hilp_memoization`             | Precompute HILP φ(s) on a 2D grid; use bilinear interpolation at query time                 |
| `hilp_checkpoint_path`             | Path to HILP value function checkpoint (`.pkl` for JAX/Flax, `.pt` for legacy)              |
| `TD_thres_for_far_target`          | HILP value threshold: V < threshold → switch to RMSE guidance                               |

## Code Conventions

### Algorithm Implementation

- All algorithms inherit from `BaseAlgo` or `BasePyTorchAlgo` (in `df_base.py`)
- Must implement `run()` and accept `DictConfig cfg` in `__init__`
- **Do not modify `df_base.py`** unless absolutely necessary — it's infrastructure
- Add new plan-processing methods to `PlanPostprocMixin` (`plan_postproc.py`); add visualization methods to `PlanVizMixin` (`plan_viz.py`)

### Environment Management

- Use `env_manager.EnvManager` for vectorized environments
- State management via `_get_sim_state()` / `_set_sim_state()` for MCTS rollouts
- AntMaze: state = `[qpos + qvel]` (29D), goal = `sub_goal_pos[:2]`
- PointMaze: PID controller from prev_sim_state to current_sim_state

### Configuration

- Never hardcode parameters — always use Hydra config via `self.cfg`
- Use `+` prefix for new config keys on CLI: `+new_key=value`
- Add new training/architecture params to `ckpt_df_planning.yaml`; add new eval/planning params to `df_planning.yaml`
- Set `WANDB_ENTITY` before running experiments with online logging

### What to Avoid

- Changing tensor shapes without validating throughout the pipeline
- Skipping padding validation when modifying sequence logic
- Overriding Hydra defaults in code (use config files)
- Assuming relative paths work with Hydra (use absolute paths or W&B run IDs)

## File Organization

| File                                                  | Purpose                                              |
| ----------------------------------------------------- | ---------------------------------------------------- |
| `main.py`                                             | Hydra entry point                                    |
| `algorithms/diffusion_forcing/df_planning.py`         | Core MCTS+diffusion algorithm (~5500 lines)          |
| `algorithms/diffusion_forcing/plan_postproc.py`       | `PlanPostprocMixin`: plan extraction & deduplication |
| `algorithms/diffusion_forcing/plan_viz.py`            | `PlanVizMixin`: HILP heatmap & denoising video       |
| `algorithms/diffusion_forcing/uncertainty_estimator.py` | Uncertainty estimators (entropy/variance/cluster)  |
| `algorithms/diffusion_forcing/tree_node.py`           | `TreeNode` for MCTS (holds `cluster_subplans`)       |
| `algorithms/diffusion_forcing/df_base.py`             | Base class infrastructure                            |
| `algorithms/diffusion_forcing/guidance.py`            | Plan guidance (goal, HILP, RDF, particle)            |
| `algorithms/diffusion_forcing/env_manager.py`         | Vectorized env management                            |
| `algorithms/diffusion_forcing/models/`                | Neural network architectures                         |
| `experiments/exp_planning.py`                         | Training/validation experiment tasks                 |
| `configurations/config.yaml`                          | Root Hydra config                                    |
| `configurations/algorithm/ckpt_df_planning.yaml`      | Ckpt-bound model definition (architecture, data)     |
| `configurations/algorithm/train_df_planning.yaml`     | Training-only params (optimizer, batch size)         |
| `configurations/algorithm/df_planning.yaml`           | Eval-only planning & guidance hyperparameters        |

- `debug/` for unit tests and debug scripts; `scripts/` for non-experiment scripts
- Do not run files in `algorithms/` directly — use `main.py`

## Logging & Monitoring

- **WandB**: configs auto-logged; offline mode via `wandb.mode=offline`
- **Checkpoints**: `outputs/[date]/[run_id]/checkpoints/`; `outputs/latest-run` symlink
- **Debug logging** via `Tracer` context manager:
  ```python
  from utils.debug_utils import Tracer
  tracer = Tracer(run_id=run_id)
  tracer.log("section.subsection", {"key": value})  # → logs/*.jsonl
  ```
- Use the `log-instrumentation` skill to add structured logging; `log-analysis` skill to generate HTML reports

## Known Issues

### Undenoised Tokens in Later Plan Segments (partially mitigated)

Only the first `used_len` tokens are fully denoised; remaining tokens carry noise artifacts. Current mitigation: use `sequence_dividing_factor=5` and `horizon_scale=0.5`. If noise artifacts persist, consider increasing `sequence_dividing_factor` further or implementing explicit denoising masking.
