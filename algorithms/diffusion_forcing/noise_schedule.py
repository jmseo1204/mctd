"""noise_schedule.py
Noise schedule construction utilities for Diffusion Forcing planning.

Provides NoiseScheduleMixin — a mixin class whose methods handle:
  - Building per-step noise level tensors (_construct_noise_levels)
  - Generating tree-conditioned denoising schedules (_generate_tree_conditioned_schedule)
  - Overriding the base noise level sampler for training (_generate_noise_levels)

Intended to be inherited by DiffusionForcingPlanning alongside DiffusionForcingBase.
`super()` calls inside these methods correctly resolve to DiffusionForcingBase via MRO.
"""

from __future__ import annotations

from random import random
from typing import Optional

import numpy as np
import torch
from einops import rearrange


class NoiseScheduleMixin:
    """Mixin providing noise schedule construction methods."""

    def _construct_noise_levels(
        self,
        levels: np.ndarray,
        batch_size: int,
        stabilization: int = 0,
        pad_tokens: int = 0,
        include_final_token: bool = False,
        include_init_token: bool = False,
    ) -> torch.Tensor:
        """Build noise levels for diffusion inference. (batch, n_tokens) tensor

        This function builds the full noise schedule for a single diffusion step.

        Args:
            levels: Noise level schedule for plan tokens (b, plan_tokens) shape
            batch_size: Batch size
            stabilization: Noise level for parent obs token (typically 0-2)
            pad_tokens: Number of padding tokens
            include_final_token: Whether to include final_token
            include_init_token: Whether to prepend init_token slot (pre-built format always False).

        Returns:
            Noise levels array (t, b)
        """
        components = []
        components.append(
            np.full((batch_size, 1), stabilization, dtype=np.int64)
        )  # given parent_obs additional token
        components.append(levels)  # plan tokens

        components.append(
            np.full((batch_size, pad_tokens), self.sampling_timesteps, dtype=np.int64)
        )  # padding
        components = torch.from_numpy(np.concatenate(components, axis=1)).to(
            self.device
        )

        result = rearrange(components, "b t -> t b", b=batch_size)  # (n_tokens, b)

        # Validate result shape before returning
        assert result.ndim == 2, f"result.ndim={result.ndim}, expected 2"
        assert result.shape[1] == batch_size, (
            f"result.shape[1]={result.shape[1]}, expected batch_size={batch_size}"
        )

        return result

    def _generate_tree_conditioned_schedule(
        self,
        start_levels: np.ndarray,
        prefix_len_per_batch: Optional[np.ndarray] = None,
        is_replanning: bool = False,
        num_denoising_steps_override: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates the N-step denoising schedule for tree-conditioned search.
        Returns a tensor of shape (B, Steps, T) representing the sequence of noise levels.

        If complete_denoising=True, continues denoising all remaining segments until all
        tokens reach 0 (used for value estimation). Otherwise, denoises one segment only.

        Args:
            num_denoising_steps_override: If set, overrides self.sampling_timesteps
                for the reduction_amount computation (used for fast uncertainty sampling).
        """
        # start_levels shape: (B, plan_tokens)  # (b, t)

        batch_size = start_levels.shape[0]
        current_levels = start_levels.copy()
        schedule = [current_levels.copy()]

        def _one_segment_pass(levels):
            to_levels_list = []
            for b in range(batch_size):
                _prefix_len = int(prefix_len_per_batch[b]) if prefix_len_per_batch is not None else None
                to_levels_b = self.process_segment_noise_levels(
                    levels[b],
                    self.sequence_dividing_factor,
                    prefix_len=_prefix_len,
                    is_replanning=is_replanning,
                    num_denoising_steps_override=num_denoising_steps_override,
                )  # (m, t)
                to_levels_list.append(to_levels_b)

            max_m = max(len(steps) for steps in to_levels_list)
            for b in range(batch_size):
                if len(to_levels_list[b]) < max_m:
                    padding = np.tile(
                        to_levels_list[b][-1:], (max_m - len(to_levels_list[b]), 1)
                    )
                    to_levels_list[b] = np.concatenate(
                        [to_levels_list[b], padding], axis=0
                    )

            batch_steps = np.stack(to_levels_list, axis=1)  # (M, B, T)
            return batch_steps

        # First segment pass
        batch_steps = _one_segment_pass(current_levels)
        for m in range(1, batch_steps.shape[0]):
            schedule.append(batch_steps[m].copy())
        current_levels = batch_steps[-1]

        complete_denoising = False
        # [DEPRECATED] but leaving the code just in case
        # If complete_denoising, keep processing segments until all tokens reach 0
        if complete_denoising:
            max_segments = self.sequence_dividing_factor  # at most N more passes
            for _ in range(max_segments - 1):
                if not np.any(current_levels > 0):
                    break
                batch_steps = _one_segment_pass(current_levels)
                for m in range(1, batch_steps.shape[0]):
                    schedule.append(batch_steps[m].copy())
                current_levels = batch_steps[-1]

        return np.stack(schedule, axis=0).transpose(1, 0, 2)  # (B, TotalSteps, T)

    def _generate_noise_levels(
        self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise_levels = super()._generate_noise_levels(xs, masks)
        _, batch_size, *_ = xs.shape

        # first frame is almost always known, this reflect that
        if random() < 0.5:
            noise_levels[0] = torch.randint(
                0, self.timesteps // 4, (batch_size,), device=xs.device
            )

        return noise_levels
