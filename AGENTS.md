# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the Hydra entry point for experiments. Core planning code lives in `algorithms/diffusion_forcing/`, shared model utilities in `algorithms/common/`, experiment runners in `experiments/`, and dataset adapters in `datasets/`. Hydra configs are organized under `configurations/algorithm/`, `configurations/dataset/`, `configurations/experiment/`, and optional `configurations/cluster/`. Operational scripts such as job generation, checkpoint scanning, and analysis live in `scripts/`; Docker assets live in `dockerfile/`. Treat `logs/`, `wandb/`, `lightning_logs/`, `jobs/`, and generated images under `debug/` or the repo root as run artifacts, not source.

## Build, Test, and Development Commands
Use the Docker workflow first; the host machine is intentionally minimal.

```bash
bash dockerfile/docker_build_and_run.sh   # build/run the mctd:0.1 container
bash train.sh                             # interactive training launcher
bash eval.sh                              # checkpoint selection + eval job launch
python main.py +name=dev experiment=exp_planning algorithm=train_df_planning dataset=og_antmaze_giant_stitch wandb.mode=offline
python scripts/generate_jobs_generalized.py --dataset antmaze_giant_stitch --model_id <model_id>
```

## Coding Style & Naming Conventions
Python uses 4-space indentation, `snake_case` for functions/files/variables, and lower-snake-case YAML names such as `train_df_planning.yaml`. Follow the existing import grouping and keep type hints where the file already uses them. Prefer Hydra config values over hardcoded constants; new runtime options belong in the relevant config file, not inline in code. Shell scripts should stay Bash-compatible with `set -euo pipefail`, and shared environment values should come from `scripts/project_config.sh`.

## Testing Guidelines
There is no centralized `pytest` or lint configuration in this repository. Validation is script-driven: run focused checks such as `python test_sampling_way.py`, `python test_terminal_depth_fix.py`, or `python debug/test_kde_guidance_grad.py` for changes in sampling, terminal-depth logic, or guidance visualization. For algorithm changes, also run a reduced local Hydra command with `wandb.mode=offline` before launching full Docker jobs.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects, often with a status prefix in brackets, for example `[done] add feasibility check` or `[ongoing] training episode seperating with the one of eval`. Keep commits scoped to one change. Pull requests should state the research or engineering motivation, list the exact commands run, note any config or dataset assumptions, and attach plots/screenshots when behavior or visualization changes.

## Configuration & Environment Tips
Keep `WANDB_ENTITY`, `WANDB_PROJECT`, GPU selection, and Docker image tags aligned in `scripts/project_config.sh`. When adding configs, preserve the train/eval split: checkpoint-bound model settings belong in `ckpt_df_planning.yaml`, training loop settings in `train_df_planning.yaml`, and planning-time overrides in `df_planning.yaml`.
