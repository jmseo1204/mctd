#!/usr/bin/env python3
"""Migrate ckpt_scan_cache.json entries to the full ckpt_df_planning.yaml structure.

Old entries under algorithm: only had causal, scheduling_matrix, frame_stack,
obs_dim_indices, pos_dim_indices, diffusion.architecture.*.

New structure mirrors ckpt_df_planning.yaml: adds frame_skip, context_frames,
padding_mode, uncertainty_scale, external_cond_dim, jump, train_dataset_config,
and diffusion.* hyperparams.

For existing entries we cannot know the true values, so we fill with the
ckpt_df_planning.yaml defaults (which were fixed across all old checkpoints).

Usage (run inside Docker):
    python3 scripts/migrate_ckpt_scan_cache.py [path/to/ckpt_scan_cache.json]
    Default path: /home/jmseo1204/mctd_outputs/ckpt_scan_cache.json
"""
import json
import sys
from pathlib import Path

DEFAULT_CACHE_PATH = "/home/jmseo1204/mctd_outputs/ckpt_scan_cache.json"

# Stable defaults from ckpt_df_planning.yaml — valid for all old checkpoints.
ALGO_DEFAULTS = {
    "frame_skip": 1,
    "context_frames": 1,
    "padding_mode": "same",
    "uncertainty_scale": 1,
    "external_cond_dim": 0,
}
DIFFUSION_DEFAULTS = {
    "timesteps": 1000,
    "sampling_timesteps": 100,
    "beta_schedule": "linear",
    "objective": "pred_x0",
    "use_fused_snr": False,
    "snr_clip": 5.0,
    "cum_snr_decay": 0.98,
    "clip_noise": 20.0,
    "stabilization_level": 10,
    "schedule_fn_kwargs": {},
}


def migrate(cache_path: str) -> None:
    path = Path(cache_path)
    with open(path) as f:
        cache = json.load(f)

    updated = 0
    for key, entry in cache.items():
        algo = entry.get("algorithm") or {}
        ds   = entry.get("dataset") or {}
        changed = False

        # Fill missing algorithm-level fields
        for k, v in ALGO_DEFAULTS.items():
            if k not in algo:
                algo[k] = v
                changed = True

        # algorithm.jump should mirror dataset.jump (same value, different location)
        if "jump" not in algo and ds.get("jump") is not None:
            algo["jump"] = ds["jump"]
            changed = True

        # train_dataset_config mirrors dataset.config
        if "train_dataset_config" not in algo and ds.get("config"):
            algo["train_dataset_config"] = ds["config"]
            changed = True

        # Fill missing diffusion-level fields
        diff = algo.get("diffusion") or {}
        for k, v in DIFFUSION_DEFAULTS.items():
            if k not in diff:
                diff[k] = v
                changed = True
        algo["diffusion"] = diff

        if changed:
            entry["algorithm"] = algo
            updated += 1

    with open(path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"[migrate] {updated}/{len(cache)} entries updated → {path}")

    # Verify a sample entry
    sample = next(iter(cache.values()))
    print(f"[migrate] algorithm keys: {sorted(sample['algorithm'].keys())}")
    print(f"[migrate] diffusion keys: {sorted(sample['algorithm']['diffusion'].keys())}")


if __name__ == "__main__":
    cache_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CACHE_PATH
    migrate(cache_path)
