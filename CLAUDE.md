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

The `MCTDPlanningAlgorithm` class (in `df_planning.py`) is the primary algorithm. Key components:

- **MCTSTreeState**: Dataclass holding state for a single MCTS tree
- **TreeNode**: MCTS nodes with counts, values, and exploration tracking
- **Bidirectional Search**: Trees grow from both start and goal simultaneously
- **Guidance**: Uses diffusion model predictions to guide tree expansion
- **Parallel Search**: Can run multiple tree instances in parallel
- **Pyramid Scheduling**: Optional hierarchical denoising structure

Key parameters (in `df_planning.yaml`):
- `mctd_num_denoising_steps`: Number of diffusion steps (e.g., 100)
- `mctd_max_search_num`: Maximum tree expansion nodes
- `parallel_search_num`: Number of parallel tree instances
- `mctd_guidance_scales`: List of guidance scales for tree expansion
- `mctd_skip_level_steps`: Steps to skip between tree levels
- `padding_mode`: "same" or "zero" for padding strategy

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

### Working with Diffusion Models

- **Base class**: `DiffusionForcingBase` (in `df_base.py`) handles core diffusion training logic
- **Planning class**: `MCTDPlanningAlgorithm` (in `df_planning.py`) implements MCTS + diffusion inference
- **Data normalization**: Always normalize observations/actions using dataset statistics (`data_mean`, `data_std`)
- **Guidance**: Diffusion guidance works via conditional predictions - see `guidance_scale` parameter

### Tree Search Implementation

The MCTS implementation has:
- **Bidirectional trees**: Root and goal trees that expand toward each other
- **Leaf parallelization**: Optional parallel expansion at tree leaves
- **Visit counting**: UCB-style exploration with virtual visit weights
- **Convergence detection**: Early stopping when trees meet

Key parameters for tuning:
- `mctd_max_search_num` - larger = more thorough search but slower
- `parallel_search_num` - larger = faster but requires more memory
- `mctd_guidance_scales` - affects exploration vs exploitation

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

- **main.py** - Entry point for all experiments (Hydra launcher)
- **algorithms/diffusion_forcing/df_planning.py** - Core MCTD algorithm (~3000+ lines)
- **algorithms/diffusion_forcing/tree_node.py** - MCTS tree node data structure
- **experiments/exp_planning.py** - Planning experiment wrapper
- **utils/logging_utils.py** - Visualization and logging utilities
- **configurations/config.yaml** - Main Hydra configuration

## Common Pitfalls

1. **W&B Entity Not Set** - Will error; set in `config.yaml` or `wandb.entity=name` on command line
2. **Missing MuJoCo** - Docker setup required; system won't work without MuJoCo binaries
3. **Padding Mismatch** - MCTS trees need consistent padding; see `test_padding_consistency.py`
4. **GPU Memory** - `parallel_search_num` and `mctd_max_search_num` significantly affect memory
5. **Checkpoint Paths** - Use W&B run IDs or absolute paths; relative paths fail with Hydra's output directories

## Useful Resources

- **Paper**: https://arxiv.org/abs/2502.07202 (MCTD - ICML 2025)
- **Fast-MCTD Paper**: https://arxiv.org/abs/2506.09498
- **Project Page**: https://jaesikyoon.com/mctd-page/
- **W&B Logs**: https://wandb.ai/jaesikyoon/jaesik_mctd (public logs)
- **OGBench**: https://seohong.me/projects/ogbench/ (benchmark environments)
- **Hydra Docs**: https://hydra.cc/docs/intro/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
