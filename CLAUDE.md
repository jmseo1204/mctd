# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Monte Carlo Tree Diffusion (MCTD)** is a framework for improving inference-time performance of diffusion models by integrating denoising with Monte Carlo Tree Search (MCTS). This repository contains implementations of:

- **MCTD**: Original framework for planning in complex environments (point/ant mazes)
- **Fast-MCTD**: Enhanced version with parallel tree search and abstract-level planning

The codebase is forked from [Boyuan Chen's research template](https://github.com/buoyancy99/research-template) and builds on the Diffusion Forcing framework.

## Key Technologies

- **Framework**: PyTorch Lightning (for training)
- **Configuration**: Hydra (YAML-based config management)
- **Logging**: Weights & Biases (W&B)
- **Environment**: OGBench (customized with velocity observations and deterministic start/goal positions)
- **Physics Engine**: MuJoCo 2.1.0
- **Environments**: Point/Ant Mazes (medium, large, giant, teleport variants)

## Repository Structure

```
algorithms/
  ├── diffusion_forcing/
  │   ├── df_base.py              # Base class for Diffusion Forcing
  │   ├── df_planning.py          # MCTD planning algorithm (main implementation)
  │   ├── tree_node.py            # MCTS tree node implementation
  │   ├── models/                 # Neural network models
  │   └── df_video.py             # Video generation/visualization
  └── common/                     # General-purpose components

configurations/
  ├── config.yaml                 # Main Hydra config
  ├── algorithm/                  # Algorithm-specific configs
  │   ├── df_planning.yaml       # MCTD planning hyperparameters
  │   └── df_base.yaml           # Base diffusion config
  ├── dataset/                    # Dataset configs
  ├── experiment/                 # Experiment configs
  └── cluster/                    # Cluster/SLURM configs

experiments/
  ├── exp_base.py                 # Base experiment class
  ├── exp_planning.py             # Planning experiment (validation/evaluation)
  └── exp_video.py                # Video generation experiment

datasets/
  └── offline_rl/                 # Offline RL dataset loading

utils/
  ├── logging_utils.py            # Visualization and logging helpers
  ├── wandb_utils.py              # W&B integration
  ├── ckpt_utils.py               # Checkpoint management
  └── cluster_utils.py            # SLURM submission

dockerfile/                       # Docker setup with MuJoCo dependencies
dql/                             # DQL agent components
jobs/                            # Job queue (for experiment management)
exp_results/                     # Aggregated experiment results
```

## Core Algorithm Architecture

### df_planning.py (Main MCTD Implementation)

The `DiffusionForcingPlanning` class in `df_planning.py` (~2600 lines) is the **primary and only production algorithm implementation**. It extends PyTorch Lightning's `LightningModule` and contains all MCTD planning logic.

**Important**: `DiffusionForcingPlanning` overrides base diffusion training behavior from `df_base.py`. Use this file as the authoritative source for understanding the algorithm.

#### Key Data Structures

- **MCTSTreeState**: Dataclass holding complete state for a single MCTS tree instance (nodes, values, counts, etc.)
- **TreeNode** (in `tree_node.py`): MCTS tree nodes with visit counts, value estimates, and exploration tracking

#### Core Algorithm Methods

**Planning (Main Entry Points)**:
- **`p_mctd_plan()`** (~120 lines) - High-level MCTS planning algorithm coordinator. Orchestrates tree initialization, search phases, and plan extraction
- **`parallel_plan()`** (~350 lines) - Parallel execution wrapper for running multiple MCTS instances with load balancing
- **`interact()`** (~420 lines) - Full inference loop including environment interaction, plan execution, and replanning

**MCTS Search**:
- **`_run_mcts_search()`** (~550 lines) - Core tree search loop with bidirectional expansion, value computation, and leaf selection
- **`_init_mcts_tree()`** (~70 lines) - Initialize root and goal trees with proper state management
- **`_select_best_leaf()`** - Select highest-value leaf node for expansion
- **`_build_plan_from_leaf()`** (~75 lines) - Extract trajectory from leaf node back to root

**Sequence & Noise Management**:
- **`_construct_sequence()`** (~60 lines) - Build diffusion input sequences with proper padding and token arrangement
- **`_construct_noise_levels()`** (~40 lines) - Generate noise schedules for diffusion steps
- **`_generate_bidirectional_schedule()`** (~60 lines) - Create interleaved noise schedules for bidirectional tree search
- **`process_segment_noise_levels()`** (~50 lines) - Process hierarchical noise levels for pyramid scheduling

**Value & Guidance**:
- **`calculate_values()`** - Compute state values using diffusion model
- **`calculate_values_bidir()`** (~70 lines) - Bidirectional value computation for root and goal trees
- **`_get_hilp_value_fn()`** - Load hierarchical inverse RL value function for guidance
- **`_compute_hilp_values()`** (~60 lines) - Compute HILP guidance values

**Training & Model**:
- **`_build_model()`** - Override base class; constructs diffusion model
- **`_preprocess_batch()`** (~30 lines) - Override base class; handles data normalization
- **`training_step()`** - Override base class; training loop (inherits from base with modifications)
- **`validation_step()`** - Override base class; validation/planning step

#### Key Parameters (in `df_planning.yaml`)

Core MCTS parameters:
- `mctd_num_denoising_steps`: Number of diffusion denoising steps (e.g., 100)
- `mctd_max_search_num`: Maximum total nodes to expand in MCTS tree
- `mctd_guidance_scales`: List of guidance scales for tree expansion control

Parallel execution:
- `parallel_search_num`: Number of parallel MCTS tree instances to run
- `leaf_parallelization`: Enable parallel leaf expansion
- `parallel_multiple_visits`: Allow multiple visits per leaf during parallelization

Sequence & tree structure:
- `mctd_skip_level_steps`: Steps between pyramid hierarchy levels
- `padding_mode`: "same" (repeat last token) or "zero" for padding
- `sequence_dividing_factor`: Controls how sequences are divided for bidirectional search
- `is_unknown_final_token`: Whether final position is unknown (affects padding reserve)

Search behavior:
- `bidirectional_search`: Enable bidirectional expansion from start and goal
- `meeting_delta`: Threshold for detecting when trees meet
- `early_stopping_condition`: Condition for early search termination
- `virtual_visit_weight`: Weight for virtual visits in UCB exploration

Advanced features:
- `use_hilp_guidance`: Enable hierarchical inverse RL guidance
- `anchor_guidance_scale`: Scale for anchor point guidance
- `rdf_guidance_scale`: Scale for RDF (reverse diffusion flow) guidance
- `pyramid`: Enable hierarchical denoising (pyramid scheduling)
- `mcts_use_sim`: Use physics simulator for trajectory validation
- `sub_goal_interval`: Interval for intermediate subgoal extraction

### Configuration Hierarchy

Hydra composes configs from multiple files:
1. `configurations/config.yaml` - base/global settings
2. Algorithm config (e.g., `algorithm/df_planning.yaml`)
3. Dataset config (e.g., `dataset/pointmaze.yaml`)
4. Experiment config (e.g., `experiment/validation.yaml`)
5. Command-line overrides (e.g., `+name=test_run`)

## Common Development Commands

### Running Experiments

```bash
# Basic training/planning run (requires W&B login)
python main.py +name=run_name algorithm=df_planning dataset=pointmaze experiment=validation

# Resume a run from W&B checkpoint
python main.py +name=run_name +resume=WANDB_RUN_ID algorithm=df_planning

# Load a checkpoint locally
python main.py +name=run_name +load=/path/to/checkpoint.ckpt algorithm=df_planning

# Override specific parameters
python main.py +name=test algorithm=df_planning mctd_num_denoising_steps=50 parallel_search_num=4

# Disable W&B logging (offline mode)
python main.py +name=test algorithm=df_planning wandb.mode=offline
```

### Job Management Scripts

These scripts create and manage experiment jobs in a queue:

```bash
# Create evaluation jobs
python insert_pointmaze_validation_jobs.py  # Point Maze evaluation
python insert_antmaze_validation_jobs.py    # Ant Maze evaluation
python insert_giant_maze_validation_jobs.py # Giant Maze evaluation

# Run queued jobs with GPU assignment
python run_jobs.py     # For diffusion model training/evaluation
python run_dql_jobs.py # For DQL agent evaluation

# Generate jobs from template (generalized script)
python generate_jobs_generalized.py
```

**Before running job scripts**, edit them to set:
- `WANDB_ENTITY` - your W&B username/organization
- `WANDB_PROJECT_NAME` - project name for logging
- `available_gpus` - list of GPU IDs to use
- Other training hyperparameters as needed

### Test Scripts

Individual test files validate specific functionality:

```bash
# Run a test to validate functionality
python test_padding_consistency.py
python test_gradient_flow.py
python test_hilp_integration.py
python test_segment_rdf_guidance.py
python test_trajectory_diagnostics.py

# These are debugging/validation scripts, not unit tests
# Test files are added to .gitignore (test_*.py)
```

### Analysis and Visualization

```bash
# Summarize experiment results
python summarize_results.py

# Analyze guidance results
python analyze_guidance_results.py

# Extract checkpoint information
python peek_ckpt.py

# Generate analysis from logs
bash anal_logs.sh
```

## Important Development Notes

### Configuration Management

- **Always use Hydra for configs** - don't hardcode parameters in code
- **Config files in `configurations/`** - add new config groups for new domains (environments, algorithms)
- **Command-line overrides**: Use `+key=value` for new keys, `key=value` for existing
- **Defaults composition**: Use `defaults:` section in YAML to inherit from base configs

### Working with DiffusionForcingPlanning (df_planning.py)

The `DiffusionForcingPlanning` class is the complete algorithm implementation that handles both training and inference:

- **Model Building**: `_build_model()` creates the diffusion model architecture
- **Data Preprocessing**: `_preprocess_batch()` normalizes observations/actions using dataset statistics (`data_mean`, `data_std`)
- **Training**: `training_step()` runs diffusion loss computation (overrides base class)
- **Inference**: `p_mctd_plan()` or `interact()` runs the full MCTS planning loop
- **Guidance**: Uses diffusion predictions and optional HILP value functions to guide tree expansion via `mctd_guidance_scales`

**Note on df_base.py**: `DiffusionForcingBase` provides base training infrastructure (loss computation, model setup), but `DiffusionForcingPlanning` overrides key methods and adds all MCTS-specific logic. Treat `DiffusionForcingPlanning` as the authoritative source.

### Tree Search Implementation (df_planning.py)

The MCTS tree search in `DiffusionForcingPlanning` works as follows:

**Search Structure**:
- **Bidirectional trees**: `_run_mcts_search()` grows root and goal trees simultaneously toward each other
- **Tree state management**: `MCTSTreeState` maintains nodes, visit counts, values, and trajectory buffers
- **Leaf nodes**: Terminal nodes that can be expanded or rolled out to generate new trajectories

**Expansion & Rollout**:
- **Value computation**: `calculate_values()` and `calculate_values_bidir()` use diffusion model to estimate state values
- **Leaf selection**: `_select_best_leaf()` picks nodes with highest UCB scores using `virtual_visit_weight`
- **Plan extraction**: `_build_plan_from_leaf()` reconstructs trajectory from leaf back to root through parent pointers
- **Dynamic goals**: `_select_dynamic_goal()` can choose intermediate subgoals during rollout

**Parallel Execution**:
- **Parallel instances**: `parallel_plan()` runs multiple MCTS trees in parallel with different random seeds
- **Leaf parallelization**: When enabled, multiple leaves expand simultaneously at same tree level
- **Load balancing**: Handles trees of different depths/sizes efficiently

**Convergence & Stopping**:
- **Early stopping**: `early_stopping_condition` triggers when convergence is detected
- **Tree meeting**: `meeting_delta` threshold detects when root and goal trees meet
- **Max iterations**: Bounded by `mctd_max_search_num` to limit compute

**Key tuning parameters**:
- `mctd_max_search_num` - controls search depth; larger = more thorough but slower
- `parallel_search_num` - number of parallel tree instances; larger = faster but more memory
- `mctd_guidance_scales` - list of scales for guiding expansion; controls exploration vs exploitation
- `virtual_visit_weight` - affects UCB scores; larger = more exploration
- `leaf_parallelization` - enable to speed up search on multi-GPU setups

### Padding and Sequence Handling

- **Sequence padding** is critical for MCTS tree representation
- `padding_mode: "same"` repeats last token; `"zero"` pads with zeros
- Reserve space for final token in bidirectional search: `n_tokens - plan_tokens - 2`
- See `test_padding_consistency.py` for padding validation logic

### W&B Integration

- **Login required**: `wandb login` before first run
- **Entity and project**: Set in config or command line
- **Run resumption**: Use `+resume=RUN_ID` to continue from checkpoint
- **Offline mode**: `wandb.mode=offline` for local-only runs
- **Checkpoints**: Automatically saved/synced with W&B if configured

### Docker Setup

```bash
# Build Docker image
cd dockerfile
docker build -t fmctd:0.1 . -f Dockerfile

# Requires: MuJoCo 2.1.0 binaries in ./dockerfile/mujoco/mujoco210/
# (Download from linked Google Drive in README)
```

## Testing and Validation

### Running Test Files

Test files are Python scripts in the root directory with the prefix `test_`. They validate specific functionality:

- **test_padding_consistency.py** - Validates sequence padding logic
- **test_gradient_flow.py** - Checks gradient propagation through models
- **test_hilp_integration.py** - Tests hierarchical inverse RL planning integration
- **test_segment_rdf_guidance.py** - Validates RDF guidance mechanism
- **test_trajectory_diagnostics.py** - Analyzes trajectory properties

Run them directly: `python test_<name>.py`

### Validation Workflow

1. Create evaluation jobs with job insertion scripts
2. Run jobs with `run_jobs.py` (distribute across GPUs)
3. Aggregate results with `summarize_results.py`
4. Results saved to `exp_results/`

## Key Files to Know

**Primary Implementation** (all MCTS planning logic):
- **algorithms/diffusion_forcing/df_planning.py** - `DiffusionForcingPlanning` class (~2600 lines). This is the authoritative implementation for MCTD planning. Contains all MCTS, diffusion, and training logic.

**Supporting Files**:
- **main.py** - Entry point for all experiments (Hydra launcher)
- **algorithms/diffusion_forcing/tree_node.py** - `TreeNode` class for MCTS tree structure
- **algorithms/diffusion_forcing/df_base.py** - Base diffusion training infrastructure (loss computation, model setup). `DiffusionForcingPlanning` overrides this.
- **experiments/exp_planning.py** - Planning experiment wrapper for running inference/validation
- **utils/logging_utils.py** - Visualization and logging utilities
- **configurations/config.yaml** - Main Hydra configuration
- **configurations/algorithm/df_planning.yaml** - MCTD-specific hyperparameters

## Working with df_planning.py

### Code Organization

The `DiffusionForcingPlanning` class is large (~2600 lines) with clear method groupings:
1. **Initialization** (`__init__`, `_build_model`, `_preprocess_batch`) - lines ~75-280
2. **Training** (`training_step`, `validation_step`) - lines ~280-330
3. **Noise scheduling** (`process_segment_noise_levels`, `_construct_noise_levels`, `_construct_sequence`, `_generate_bidirectional_schedule`) - lines ~451-665
4. **Main planning entry points** (`parallel_plan`, `interact`) - lines ~670-1446
5. **Helper methods** (`pad_init`, `split_bundle`, `make_bundle`, `_generate_noise_levels`) - lines ~1448-1507
6. **Value computation** (`calculate_values`, `calculate_values_bidir`) - lines ~1534-1639
7. **MCTS core** (`_init_mcts_tree`, `_run_mcts_search`, `_select_best_leaf`) - lines ~1641-2471
8. **Plan extraction** (`_build_plan_from_leaf`, `_extract_output_plan`, `p_mctd_plan`) - lines ~2275-2647
9. **Simulation helpers** (`_get_sim_state`, `_set_sim_state`) - lines ~2649-2674

### Key Patterns

**Tensor Dimensions**:
- Observations: `(batch, n_tokens, obs_dim)` after frame stacking
- Actions: `(batch, n_tokens-1, action_dim)` (one less than observations)
- Noise levels: `(batch, n_tokens)` aligned with diffusion timesteps
- Always check padding: reserve space for final token in bidirectional search

**MCTS State Management**:
- `MCTSTreeState` holds complete tree data: `nodes` (list of TreeNodes), `values`, `visit_counts`, `trajectories`, etc.
- Root and goal trees maintained separately in bidirectional search
- Always initialize trees with proper state using `_init_mcts_tree()`

**Diffusion Inference**:
- Values come from denoising predictions at specific noise levels
- Guidance scales (`mctd_guidance_scales`) affect how strongly the model follows initial observations
- Multiple guidance scales can be used for exploration diversity in `parallel_search_num`

**Sequence Construction**:
- `_construct_sequence()` builds padded sequences for diffusion model input
- Padding mode affects how padding tokens are generated: "same" repeats last, "zero" uses zeros
- Token layout: `[obs_0, obs_1, ..., obs_n, padding]` for observations

### Common Modifications

**To change tree search behavior**:
1. Modify `_run_mcts_search()` for search strategy changes
2. Adjust `virtual_visit_weight` and `mctd_guidance_scales` parameters before implementing code changes

**To add new guidance signals**:
1. Compute guidance values (similar to `_compute_hilp_values()`)
2. Incorporate into value computation in `_run_mcts_search()` via weighted combination

**To modify plan extraction**:
1. Edit `_build_plan_from_leaf()` for trajectory reconstruction logic
2. Edit `_extract_output_plan()` for final output format

### Understanding the Inference Flow

When calling `interact()` (main inference entry point in df_planning.py):

1. **Reset environment** - Initialize observation from environment
2. **Planning loop** (each timestep):
   - Call `p_mctd_plan()` to get plan via MCTS
   - Execute first action from plan in environment
   - Get reward and next observation
   - Check for goal/timeout
3. **MCTS Planning** inside `p_mctd_plan()`:
   - Initialize root and goal trees with `_init_mcts_tree()`
   - Run `_run_mcts_search()` for fixed iterations or until convergence
   - Select best leaf with `_select_best_leaf()`
   - Extract trajectory with `_build_plan_from_leaf()`
   - Return best plan or replans if needed
4. **Search termination** - Via `early_stopping_condition`, `meeting_delta`, or `mctd_max_search_num`

Key parameters controlling this flow:
- `val_max_steps` - Max environment steps per episode
- `time_limit` - Max wall-clock time for planning
- `mctd_max_search_num` - Max MCTS expansions
- `num_tries_for_bad_plans` - Replanning attempts if confidence low
- `sub_goal_interval` - Extract intermediate subgoals

## Common Pitfalls

### df_planning.py Specific

1. **Tensor Shape Mismatches** - Observations and actions have different sequence lengths. Observations are `n_tokens`, actions are `n_tokens-1`. Always validate in `_construct_sequence()` if modifying tensor handling.

2. **Padding Consistency** - Padding must be consistent across diffusion steps and MCTS tree expansion. Use `padding_mode` ("same" or "zero") consistently. See `test_padding_consistency.py` for validation.

3. **Bidirectional Search Convergence** - Root and goal trees must meet properly. `is_unknown_final_token` and `sequence_dividing_factor` affect how sequences are split. If trees don't meet, check these parameters.

4. **Value Computation Timing** - `calculate_values()` is expensive (requires diffusion forward pass). It's called for every leaf evaluation. Adjust `mctd_num_denoising_steps` and `mctd_max_search_num` to balance quality vs speed.

5. **Memory with Large parallel_search_num** - Each parallel instance maintains its own `MCTSTreeState`. Memory scales with `parallel_search_num * n_tokens * obs_dim`. Reduce `mctd_max_search_num` or `parallel_search_num` if OOM.

6. **Guidance Scales Order** - `mctd_guidance_scales` is a list; each element is tried in parallel searches. Scales should typically increase (e.g., `[0.0, 1.0, 5.0]`). Order matters for reproducibility.

### General Pitfalls

7. **W&B Entity Not Set** - Will error; set in `config.yaml` or `wandb.entity=name` on command line
8. **Missing MuJoCo** - Docker setup required; system won't work without MuJoCo binaries
9. **GPU Memory** - `parallel_search_num` and `mctd_max_search_num` significantly affect memory
10. **Checkpoint Paths** - Use W&B run IDs or absolute paths; relative paths fail with Hydra's output directories

## Useful Resources

- **Paper**: https://arxiv.org/abs/2502.07202 (MCTD - ICML 2025)
- **Fast-MCTD Paper**: https://arxiv.org/abs/2506.09498
- **Project Page**: https://jaesikyoon.com/mctd-page/
- **W&B Logs**: https://wandb.ai/jaesikyoon/jaesik_mctd (public logs)
- **OGBench**: https://seohong.me/projects/ogbench/ (benchmark environments)
- **Hydra Docs**: https://hydra.cc/docs/intro/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
