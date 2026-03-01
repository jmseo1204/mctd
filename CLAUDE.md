# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing **Diffusion Forcing for Trajectory Planning** (MCTD). The project uses a transformer-based diffusion model to generate trajectories in offline reinforcement learning environments (OGBench environments: AntMaze, PointMaze, etc.).

**Key Components:**
- **Diffusion Forcing Planning** (`algorithms/diffusion_forcing/df_planning.py`): Core MCTS-based planning algorithm using diffusion models
- **Environment Manager** (`algorithms/diffusion_forcing/env_manager.py`): Vectorized environment management
- **Guidance System** (`algorithms/diffusion_forcing/guidance.py`): Plan optimization via guidance
- **Experiments** (`experiments/`): Training and evaluation workflows using PyTorch Lightning
- **Configuration Management**: Hydra-based YAML configs in `configurations/`

## Architecture Overview

### High-Level Flow

```
main.py (Hydra entry point)
  ↓
experiments/build_experiment() → creates experiment instance (exp_planning, exp_video, etc.)
  ↓
algorithm instance (df_planning, df_video, etc.) with DictConfig
  ↓
dataset instance (OGBench environments) with DictConfig
  ↓
run training/evaluation tasks
```

### Key Algorithm Structure

**DiffusionForcingPlanning** (df_planning.py):
- `interact()`: Main interaction loop - resets env, runs MCTS search, executes plans
- `_run_mcts_search()`: MCTS tree expansion with diffusion model guidance
- `_execute_plan_in_env()`: Executes generated plans and collects rollouts
- `_extract_output_plan()`: Extracts trajectory segments from diffusion output

**MCTSTreeState** (df_planning.py):
- Maintains diffusion model noise levels across tree nodes
- Tracks plan quality and value estimates

### Configuration System (Hydra)

All configs are YAML files in `configurations/`:
- `config.yaml`: Root config with defaults and WandB settings
- `algorithm/*.yaml`: Algorithm-specific hyperparameters
- `dataset/*.yaml`: Dataset/environment configs
- `experiment/*.yaml`: Experiment task definitions
- `cluster/*.yaml`: Optional cluster/SLURM settings

**Override syntax:** `python main.py experiment.tasks=[training] dataset.episode_len=100 +name=MyRun`

## Common Development Commands

### Running Experiments

**Interactive training (Docker-based):**
```bash
bash train_interactive.sh
```
Launches an interactive menu to select dataset, configure episode length, and resume from checkpoints.

**Direct execution (local):**
```bash
python main.py experiment.tasks=[training] experiment=exp_planning algorithm=df_planning dataset=og_antmaze_giant_stitch +name=MyTrainingRun wandb.mode=offline
```

**Resume from checkpoint:**
```bash
python main.py resume=<wandb_run_id> experiment.tasks=[training]
```

**Load and evaluate a checkpoint:**
```bash
python main.py load=<wandb_run_id> experiment.tasks=[validation]
```

### Debugging & Analysis

**Generate job specifications:**
```bash
python generate_jobs_generalized.py
```

**Analyze logs:**
```bash
bash scripts/analyze_logs.sh
```

**Run with monitoring:**
```bash
bash scripts/run_with_monitoring.sh
```

### Key Python Scripts

- `main.py`: Hydra entry point - runs experiments with config overrides
- `run_experiment.py`: Wrapper for batch experiment execution
- `run_jobs.py`: Executes job specifications from job database
- `generate_*.py`: Generate experiment job specifications

## File Organization & Conventions

### `/algorithms/diffusion_forcing/`
Core algorithm implementation. **Do not run files directly** - use `main.py` instead.

- `df_planning.py`: Main planning algorithm with MCTS + diffusion
- `df_base.py`: Base classes for diffusion forcing algorithms
- `tree_node.py`: Tree node structure for MCTS
- `env_manager.py`: Vectorized environment management
- `guidance.py`: Plan guidance/optimization utilities
- `models/`: Neural network architectures

### `/experiments/`
Experiment definitions. Create new experiments by inheriting from `ExpBase` in `exp_base.py`:

```python
from experiments.exp_base import ExpBase

class MyExperiment(ExpBase):
    def train(self, *args, **kwargs):
        # Your training logic
        pass
```

Register in `experiments/__init__.py`.

### `/configurations/`
All runtime configuration via YAML. Structure:
- `experiment/`: Task definitions (training, validation, rollout)
- `algorithm/`: Algorithm hyperparameters
- `dataset/`: Environment/dataset specs
- `cluster/`: SLURM job settings (optional)

### `/utils/`
General utilities (logging, checkpointing, WandB integration, visualization).

### `/datasets/`
Dataset and environment code. Each dataset takes a DictConfig.

## Logging & Monitoring

### WandB Integration
- Configs auto-logged to WandB
- Run resumption via `resume=<run_id>` loads checkpoint and continues logging
- Offline mode supported for compute nodes: `wandb.mode=offline`

### Checkpoint Management
- Stored in `outputs/[date]/[run_id]/checkpoints/`
- Can download from WandB via `utils/ckpt_utils.py`
- Latest run symlinked to `outputs/latest-run`

### Debug Logging
Use the `Tracer` context manager (in code) for structured debug logging:
```python
from utils.debug_utils import Tracer
tracer = Tracer(run_id=run_id)
tracer.log("section.subsection", {"key": value, ...})
# Saved to logs/*.jsonl
```

## Important Conventions & Patterns

### 1. Algorithm Implementation
- All algorithms inherit from `BaseAlgo` or `BasePyTorchAlgo`
- Must implement `run()` method and accept DictConfig `cfg` in `__init__`
- Use `self.cfg` for accessing hyperparameters

### 2. Environment Management
- Use `env_manager.EnvManager` for vectorized environments
- Always call `envs.reset()` before initial step
- State management via `_get_sim_state()` / `_set_sim_state()` for MCTS rollouts

### 3. Configuration Access
```python
from omegaconf import DictConfig

def __init__(self, cfg: DictConfig):
    self.my_param = cfg.algorithm.my_param
    self.episode_len = cfg.dataset.episode_len
```

### 4. Hydra Integration
- Use command-line overrides for one-off config changes
- Store new defaults in YAML files for reproducibility
- Use `+` prefix for new config keys: `+new_key=value`

## Known Issues & Debugging

### Plan Generation
The main algorithm (`df_planning.py:interact()` and related methods) has two known subtle issues documented in memory:

1. **Position mismatch in MCTS execution** (line ~3119): Environment reset during plan execution can cause plans to execute from wrong starting position
2. **Noise artifacts in later plan segments** (line ~3448): Only first portion of diffusion output is denoised; remaining tokens contain unfiltered noise

Check `logs_memory_debug/` for diagnostic logs if experiencing trajectory quality issues.

### Logging & Instrumentation
Use the `log-instrumentation` skill to add coarse-to-fine structured logging to algorithm code. Generates JSONL logs compatible with the `log-analysis` skill for interactive diagnostics.

## Testing & Validation

**Create validation jobs:**
```bash
python insert_antmaze_validation_jobs.py  # or other environments
python run_jobs.py  # executes jobs
```

**Analyze results:**
- Checkpoints saved to WandB and `outputs/`
- Use visualization tools in `utils/` to inspect trajectories
- Check `visualizations/` and `rollout_visualizations/` directories

## Key Files to Know

| File | Purpose |
|------|---------|
| `main.py` | Hydra entry, experiment launcher |
| `df_planning.py` | Core MCTS+diffusion algorithm (2700+ lines) |
| `env_manager.py` | Vectorized env management |
| `tree_node.py` | MCTS node structure |
| `exp_planning.py` | Training/validation experiment tasks |
| `config.yaml` | Root Hydra config |
| `train_interactive.sh` | Interactive training launcher |

## Development Workflow

1. **Make code changes** in `algorithms/diffusion_forcing/` or algorithm of interest
2. **Test via config overrides**: `python main.py <config_overrides> wandb.mode=offline`
3. **For significant changes**, create new YAML config in `configurations/algorithm/`
4. **Monitor via WandB** or check `outputs/latest-run/` locally
5. **Use logging** (Tracer context manager) to debug algorithm behavior
6. **Commit changes** with clear messages describing algorithm modifications

## Useful Utilities

- `utils/ckpt_utils.py`: Download checkpoints from WandB
- `utils/wandb_utils.py`: WandB logger implementations (OfflineWandbLogger, SpaceEfficientWandbLogger)
- `utils/debug_utils.py`: Tracer for structured debug logging
- `utils/cluster_utils.py`: SLURM job submission
- `utils/distributed_utils.py`: Multi-GPU support checks
