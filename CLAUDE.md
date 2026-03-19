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

This is the primary file (~2600 lines). **Treat it as authoritative — it overrides `df_base.py` behavior.**

Key entry points and call chain:
```
interact() → p_mctd_plan() → _run_mcts_search() → _extract_output_plan()
                                                 → _execute_plan_in_env()
```

- `interact()`: Resets env, captures initial_sim_state, initializes bidirectional MCTS trees, runs search, executes plan
- `_run_mcts_search()`: Iterative tree expansion with diffusion guidance
- `_execute_plan_in_env()`: Loops `open_loop_horizon` steps; tracks sim_state, computes actions, steps env
- `_extract_output_plan()`: Extracts trajectory segment based on tree depth and `sequence_dividing_factor`

**Plan segment extraction:**
```
seg_size = plan_tokens // sequence_dividing_factor
used_len = depth * seg_size * frame_stack
plan_a_full[used_len:] = unused (contains undenoised noise)
```

### Key Tensor Dimensions

Always verify when modifying `_construct_sequence()` or value computation:
- Observations: `(batch, n_tokens, obs_dim)` after frame stacking
- Actions: `(batch, n_tokens-1, action_dim)` — **one fewer token than observations**
- Noise levels: `(batch, n_tokens)` aligned with diffusion timesteps

### Bidirectional MCTS

- `tree1` = forward planning, `tree2` = backward planning (uses separate envs)
- `is_unknown_final_token` and `sequence_dividing_factor` must be consistent
- Value functions: `calculate_values()`, `calculate_values_bidir()`

### Sim State

Physical state dict with keys `qpos` (joint positions) and `qvel` (joint velocities). `qpos[:2]` is always `(x, y)` world coordinates. When `use_rollout=false`, synthetic sim_states are created by copying parent and updating `qpos[:2]` from plan position.

### Configuration System (Hydra)

Root config: `configurations/config.yaml`. Defaults point to:
- `algorithm/df_planning.yaml` → extends `df_base.yaml`
- `dataset/og_antmaze_giant_stitch.yaml` (or similar)
- `experiment/exp_planning.yaml`

**Override syntax:** `python main.py +name=MyRun algorithm=df_planning dataset=og_antmaze_giant_stitch wandb.mode=offline`

## Common Development Commands

**Interactive training (Docker):**
```bash
bash train.sh   # menu: select state dim (2D/15D/29D), jump value, resume checkpoint
```

**Direct execution:**
```bash
python main.py experiment.tasks=[training] experiment=exp_planning algorithm=df_planning dataset=og_antmaze_giant_stitch +name=MyRun wandb.mode=offline
```

**Resume / load checkpoint:**
```bash
python main.py resume=<wandb_run_id> experiment.tasks=[training]
python main.py load=<wandb_run_id> experiment.tasks=[validation]
```

**Full eval pipeline:**
```bash
bash eval.sh   # selects state dim → checkpoint → generates jobs → runs via Docker
```

**Debugging (reduced scope):**
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
bash debug_log_report.sh <path/to/logfile.jsonl>   # JSONL → HTML report
bash guidance_analysis.sh
python scripts/generate_jobs_generalized.py         # generate eval job specs
python scripts/run_jobs.py                          # execute job specs
```

## Key Parameters (df_planning.yaml)

| Parameter | Purpose |
|-----------|---------|
| `mctd_num_denoising_steps` | Denoising iterations per generation (≤ `sampling_timesteps` in df_base.yaml) |
| `mctd_max_search_num` | Max MCTS tree expansions |
| `parallel_search_num` | Parallel search instances (each holds own MCTSTreeState — watch memory) |
| `mctd_guidance_scales` | List of guidance scales for exploration |
| `sequence_dividing_factor` | Segments per planning level (must match `is_unknown_final_token`) |
| `horizon_scale` | `horizon = episode_len * horizon_scale` |
| `open_loop_horizon` | Steps executed per plan |
| `val_max_loops` | Max rollout loops per validation episode |
| `meeting_delta` | Bidirectional tree convergence threshold |
| `padding_mode` | `"same"` or `"zero"` — changing this requires `test_padding_consistency.py` |
| `use_rollout` | If `false`, derive `obs_pos` from plan_history (skip physical sim) |
| `leaf_parallelization` | Parallelize leaf node evaluation |

## Code Conventions

### Algorithm Implementation
- All algorithms inherit from `BaseAlgo` or `BasePyTorchAlgo` (in `df_base.py`)
- Must implement `run()` and accept `DictConfig cfg` in `__init__`
- **Do not modify `df_base.py`** unless absolutely necessary — it's infrastructure

### Environment Management
- Use `env_manager.EnvManager` for vectorized environments
- State management via `_get_sim_state()` / `_set_sim_state()` for MCTS rollouts
- AntMaze: state = `[qpos + qvel]` (29D), goal = `sub_goal_pos[:2]`
- PointMaze: PID controller from prev_sim_state to current_sim_state

### Configuration
- Never hardcode parameters — always use Hydra config via `self.cfg`
- Use `+` prefix for new config keys on CLI: `+new_key=value`
- Set `WANDB_ENTITY` before running experiments with online logging

### What to Avoid
- Changing tensor shapes without validating throughout the pipeline
- Skipping padding validation when modifying sequence logic
- Overriding Hydra defaults in code (use config files)
- Assuming relative paths work with Hydra (use absolute paths or W&B run IDs)

## File Organization

| File | Purpose |
|------|---------|
| `main.py` | Hydra entry point |
| `algorithms/diffusion_forcing/df_planning.py` | Core MCTS+diffusion algorithm (~2600 lines) |
| `algorithms/diffusion_forcing/tree_node.py` | TreeNode for MCTS |
| `algorithms/diffusion_forcing/df_base.py` | Base class infrastructure |
| `algorithms/diffusion_forcing/guidance.py` | Plan guidance (goal, HILP, RDF) |
| `algorithms/diffusion_forcing/env_manager.py` | Vectorized env management |
| `algorithms/diffusion_forcing/models/` | Neural network architectures |
| `experiments/exp_planning.py` | Training/validation experiment tasks |
| `configurations/config.yaml` | Root Hydra config |
| `configurations/algorithm/df_planning.yaml` | Algorithm hyperparameters |

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
