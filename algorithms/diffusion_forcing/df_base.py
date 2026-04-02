"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
from tqdm import tqdm
from omegaconf import DictConfig
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any
from einops import rearrange

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.diffusion import Diffusion


class _DimLossJsonlLogger:
    """Logs per-dimension MSE loss to a local JSONL file, independent of wandb."""

    def __init__(self, log_path: str, log_every: int = 100):
        self.log_path = log_path
        self.log_every = log_every
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._buf = []

    def log(self, step: int, epoch: int, total_loss: float, per_dim_loss):
        if step % self.log_every != 0:
            return
        record = {
            "step": step,
            "epoch": epoch,
            "loss": round(float(total_loss), 6),
            "per_dim_loss": [round(float(v), 6) for v in per_dim_loss],
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


class DiffusionForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.get('guidance_scale', 2.0)
        self.context_frames = cfg.context_frames
        self.chunk_size = cfg.get('chunk_size', 50)
        self.external_cond_dim = cfg.external_cond_dim
        self.causal = cfg.causal

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise

        self._cum_snr_decay_raw = float(self.cfg.diffusion.cum_snr_decay)  # original value before frame_stack power
        self.cfg.diffusion.cum_snr_decay = self._cum_snr_decay_raw ** (self.frame_stack * cfg.frame_skip)

        self.validation_step_outputs = []
        self._dim_loss_logger = _DimLossJsonlLogger(
            log_path="logs/dim_loss.jsonl",
            log_every=100,
        )
        super().__init__(cfg)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Save training hyperparameters to checkpoint for eval-time recovery.

        These are the variables that MUST be identical at train and eval time.
        At eval time, exp_base reads these and applies them to the config before
        building the algorithm, so job configs no longer need to carry arch overrides.
        """
        from omegaconf import OmegaConf
        arch = OmegaConf.to_container(self.cfg.diffusion.architecture, resolve=True)
        hparams = {
            'frame_stack': int(self.cfg.frame_stack),
            'causal': bool(self.cfg.causal),
            'frame_skip': int(self.cfg.frame_skip),
            'uncertainty_scale': float(self.cfg.uncertainty_scale),
            'external_cond_dim': int(self.cfg.external_cond_dim),
            'diffusion': {
                'beta_schedule': str(self.cfg.diffusion.beta_schedule),
                'objective': str(self.cfg.diffusion.objective),
                'timesteps': int(self.cfg.diffusion.timesteps),
                'sampling_timesteps': int(self.cfg.diffusion.sampling_timesteps),
                'architecture': arch,
                # Diffusion behavior params — saved here so eval config can be null
                # (restored before model build via _apply_ckpt_hparams_to_cfg).
                # cum_snr_decay: save the PRE-modification value; __init__ applies
                # the frame_stack power, so restoring the raw value avoids double-application.
                'schedule_fn_kwargs': OmegaConf.to_container(self.cfg.diffusion.schedule_fn_kwargs, resolve=True),
                'use_fused_snr': bool(self.cfg.diffusion.use_fused_snr),
                'snr_clip': float(self.cfg.diffusion.snr_clip),
                'cum_snr_decay': self._cum_snr_decay_raw,
                'clip_noise': float(self.cfg.diffusion.clip_noise),
                'stabilization_level': int(self.cfg.diffusion.stabilization_level),
            },
        }
        # Save obs/pos dimension indices so eval-time algo reconstruction uses
        # the exact same observation selection as training.
        _obs_idx = self.cfg.get('obs_dim_indices', None)
        if _obs_idx is not None:
            hparams['obs_dim_indices'] = list(_obs_idx)
        _pos_idx = self.cfg.get('pos_dim_indices', None)
        if _pos_idx is not None:
            hparams['pos_dim_indices'] = list(_pos_idx)
        _dataset_cfg = self.cfg.get('train_dataset_config', None)
        if _dataset_cfg is not None:
            hparams['dataset_config'] = str(_dataset_cfg)
        _padding_mode = self.cfg.get('padding_mode', None)
        if _padding_mode is not None:
            hparams['padding_mode'] = str(_padding_mode)
        _context_frames = self.cfg.get('context_frames', None)
        if _context_frames is not None:
            hparams['context_frames'] = int(_context_frames)
        # Dataset normalization stats — required to build the model at eval time
        # (observation_dim = len(observation_mean), etc.) without a dataset config.
        for _key in ('observation_mean', 'observation_std', 'action_mean', 'action_std',
                     'reward_mean', 'reward_std', 'env_id', 'dataset'):
            _val = self.cfg.get(_key, None)
            if _val is not None:
                from omegaconf import OmegaConf as _OC
                hparams[_key] = _OC.to_container(_val, resolve=True) if hasattr(_val, '_metadata') else _val
        # Save effective (subsampled) episode_len = raw_episode_len // jump.
        # jump is declared in both train_df_planning.yaml and df_planning.yaml, so self.cfg.jump is always available.
        # This ensures eval-time job generation gets the model-aligned episode_len directly
        # without relying on stale WandB/hydra configs.
        jump = int(self.cfg.get('jump', 1))
        hparams['jump'] = jump
        raw_episode_len = (
            int(self.cfg.episode_len) if hasattr(self.cfg, 'episode_len')
            else int(self.cfg.dataset.episode_len) if hasattr(self.cfg, 'dataset') and hasattr(self.cfg.dataset, 'episode_len')
            else None
        )
        if raw_episode_len is not None:
            hparams['episode_len'] = raw_episode_len // jump
        checkpoint['training_hparams'] = hparams

    def _build_model(self):
        self.diffusion_model = Diffusion(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        params = tuple(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay, betas=self.cfg.optimizer_beta
        )
        return optimizer_dynamics

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # Warmup + cosine decay LR schedule (no external scheduler needed)
        step = self.trainer.global_step
        warmup = self.cfg.warmup_steps
        total_steps = max(self.trainer.max_steps, 1)
        lr_min_ratio = getattr(self.cfg, "lr_min_ratio", 0.1)

        if step < warmup:
            lr_scale = min(1.0, float(step + 1) / warmup)
        else:
            # cosine decay from 1.0 at step=warmup down to lr_min_ratio at step=total_steps
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            lr_scale = lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * self.cfg.lr

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        xs_pred, loss_raw = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs))

        # Log per-dimension loss to local JSONL (independent of wandb)
        # loss_raw: (T, B, fs*C) unreduced
        step = self.trainer.global_step if self.trainer else 0
        epoch = self.trainer.current_epoch if self.trainer else 0
        with torch.no_grad():
            per_dim = loss_raw.mean(dim=(0, 1)).detach().cpu().tolist()  # shape: (fs*C,)
        loss = self.reweight_loss(loss_raw, masks)
        self._dim_loss_logger.log(step, epoch, loss.item(), per_dim)

        # log the loss
        if batch_idx % 20 == 0:
            self.log("training/loss", loss)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch) 
        # xs: (T, B, fs * C)
        # conditions: (T, B, fs * cond_dim)
        # masks: (n_frames, B)
 

        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn((horizon, batch_size, *self.x_stacked_shape), device=self.device) # T, B, fs*C 
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens) # if n_tokens cannot cover the horizon, start frame moves to 1+

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    conditions[start_frame : curr_frame + horizon],
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )

            curr_frame += horizon
            pbar.update(horizon)

        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)
        self.validation_step_outputs.append((xs_pred.detach().cpu(), xs.detach().cpu()))

        return loss

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    def test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        num_frames, batch_size, *_ = xs.shape
        match self.cfg.noise_level:
            case "random_all":  # entirely random noise levels
                noise_levels = torch.randint(0, self.timesteps, (num_frames, batch_size), device=xs.device)

        if masks is not None:
            # for frames that are not available, treat as full noise
            discard = torch.all(~rearrange(masks.bool(), "(t fs) b -> t b fs", fs=self.frame_stack), -1)
            noise_levels = torch.where(discard, torch.full_like(noise_levels, self.timesteps - 1), noise_levels)

        return noise_levels
    def _generate_scheduling_matrix(self, horizon: int):
        match self.cfg.scheduling_matrix:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
            case "full_sequence":
                return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
            case "autoregressive":
                return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def reweight_loss(self, loss, weight=None):
        # Note there is another part of loss reweighting (fused_snr) inside the Diffusion class!
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape) - 1
            weight = rearrange(
                weight,
                "(t fs) b ... -> t b fs ..." + " 1" * expand_dim,
                fs=self.frame_stack,
            )
            loss = loss * weight

        return loss.mean()

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError("Number of context frames must be divisible by frame stack size")

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack

        if self.external_cond_dim:
            conditions = batch[1]
            conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(conditions, "b (t fs) d -> t b (fs d)", fs=self.frame_stack).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack).contiguous()

        return xs, conditions, masks

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape).to(xs.device)
        std = self.data_std.reshape(shape).to(xs.device)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)
