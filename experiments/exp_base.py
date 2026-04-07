"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Dict
import pathlib
import os
import sys
from tqdm import tqdm

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import DictConfig, OmegaConf, open_dict

from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero

torch.set_float32_matmul_precision("high")


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_training_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        train_dataset = self._build_dataset("training")
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        if train_dataset:
            use_gpu_cache = getattr(train_dataset, "use_gpu_cache", False)
            num_workers = 0 if use_gpu_cache else min(os.cpu_count(), self.cfg.training.data.num_workers)
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=(not use_gpu_cache),
                persistent_workers=(num_workers > 0),
            )
        else:
            return None

    def _build_validation_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation")
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        if validation_dataset:
            use_gpu_cache = getattr(validation_dataset, "use_gpu_cache", False)
            num_workers = 0 if use_gpu_cache else min(os.cpu_count(), self.cfg.validation.data.num_workers)
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.validation.batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=(not use_gpu_cache),
                persistent_workers=(num_workers > 0),
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        shuffle = False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.test.data.shuffle
        if test_dataset:
            use_gpu_cache = getattr(test_dataset, "use_gpu_cache", False)
            num_workers = 0 if use_gpu_cache else min(os.cpu_count(), self.cfg.test.data.num_workers)
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.test.batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=(not use_gpu_cache),
                persistent_workers=(num_workers > 0),
            )
        else:
            return None

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            # episode_len is not in YAML schema — inject from dataset config before building
            self._ensure_algo_cfg_fallbacks()
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        # Build dataloader to calculate total epochs from max_steps
        train_loader = self._build_training_loader()
        num_batches = len(train_loader)
        
        effective_max_steps = self.cfg.training.max_steps
        if effective_max_steps > 0:
            # Let max_steps be the sole stopping criterion to avoid epoch-count
            # conflicts when resuming from a checkpoint that already exceeded a
            # previously-computed max_epochs value.
            total_epochs = -1
        else:
            total_epochs = self.cfg.training.max_epochs

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    save_last=True,
                    **self.cfg.training.checkpointing,
                )
            )
            # Keep only the latest step-based checkpoint to save disk space
            _ckpt_dir = pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints"

            class _KeepLatestCheckpoint(pl.callbacks.Callback):
                """
                - Keeps the 2 most recent step/epoch checkpoints (current + previous).
                - After each checkpoint save, monitors training loss for WINDOW_BATCHES
                  micro-batches. If the post-save average loss exceeds the pre-save average
                  by more than SURGE_FACTOR, the latest checkpoint is discarded (the
                  previous one is kept as a safe restore point) and training is stopped.
                """
                SURGE_FACTOR   = 5.0   # post/pre loss ratio threshold
                WINDOW_BATCHES = 200   # micro-batches to average before/after each save

                def __init__(self):
                    self._recent_losses   = []   # rolling window of raw micro-batch losses
                    self._pre_ckpt_loss   = None # avg loss just before last checkpoint save
                    self._post_losses     = []   # losses collected after last save
                    self._collecting_post = False
                    self._seen_steps      = set()  # global_steps already processed

                def _step_ckpts(self):
                    if not _ckpt_dir.exists():
                        return []
                    return sorted(
                        [f for f in _ckpt_dir.glob("epoch=*.ckpt") if not f.is_symlink()],
                        key=lambda f: f.stat().st_mtime,
                    )

                def _cleanup(self):
                    for old in self._step_ckpts()[:-2]:  # keep 2 most recent
                        old.unlink(missing_ok=True)

                def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                    step = trainer.global_step

                    # --- accumulate per-micro-batch loss ---
                    loss = None
                    if isinstance(outputs, dict) and "loss" in outputs:
                        try:
                            loss = float(outputs["loss"])
                        except Exception:
                            pass
                    if loss is None:
                        v = trainer.callback_metrics.get("training/loss") or \
                            trainer.callback_metrics.get("loss")
                        if v is not None:
                            loss = float(v)

                    if loss is not None:
                        self._recent_losses.append(loss)
                        if len(self._recent_losses) > self.WINDOW_BATCHES * 2:
                            self._recent_losses = self._recent_losses[-self.WINDOW_BATCHES * 2:]
                        if self._collecting_post:
                            self._post_losses.append(loss)

                    # --- trigger once per global_step at every 2000 steps ---
                    if step > 0 and step % 2000 == 0 and step not in self._seen_steps:
                        self._seen_steps.add(step)
                        # Record pre-checkpoint average loss
                        recent_window = self._recent_losses[-self.WINDOW_BATCHES:]
                        if recent_window:
                            self._pre_ckpt_loss = sum(recent_window) / len(recent_window)
                        self._post_losses = []
                        self._collecting_post = True
                        self._cleanup()

                    # --- surge detection: after collecting WINDOW_BATCHES post-save losses ---
                    if self._collecting_post and len(self._post_losses) >= self.WINDOW_BATCHES:
                        self._collecting_post = False
                        if self._pre_ckpt_loss is not None and self._pre_ckpt_loss > 0:
                            post_avg = sum(self._post_losses) / len(self._post_losses)
                            ratio = post_avg / self._pre_ckpt_loss
                            if ratio > self.SURGE_FACTOR:
                                ckpts = self._step_ckpts()
                                if ckpts:
                                    victim = ckpts[-1]
                                    victim.unlink(missing_ok=True)
                                    kept = ckpts[-2].name if len(ckpts) >= 2 else "none"
                                    print(
                                        f"\n[SURGE DETECTED] step={step}: "
                                        f"post_loss={post_avg:.4f} vs pre_loss={self._pre_ckpt_loss:.4f} "
                                        f"(ratio={ratio:.1f}×). "
                                        f"Deleted '{victim.name}', kept '{kept}'. "
                                        f"Stopping training.",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"\n[SURGE DETECTED] step={step}: ratio={ratio:.1f}×. "
                                        f"No checkpoint to delete. Stopping training.",
                                        flush=True,
                                    )
                                trainer.should_stop = True

            callbacks.append(_KeepLatestCheckpoint())

            _model_id = self.root_cfg.get("name", None)

            class _RenameLastCheckpoint(pl.callbacks.Callback):
                """Rename last.ckpt → model.ckpt after each save, then update eval symlink."""
                def _rename(self, trainer):
                    if not trainer.is_global_zero:
                        return
                    last = _ckpt_dir / "last.ckpt"
                    model = _ckpt_dir / "model.ckpt"
                    if last.exists():
                        last.replace(model)
                    # Keep <MCTD_OUTPUT_DIR>/<model_id>/model.ckpt symlink up-to-date
                    # so it survives even if the host train.sh process dies before training ends.
                    output_base = os.environ.get("MCTD_OUTPUT_DIR")
                    if output_base and _model_id and model.exists():
                        eval_dir = pathlib.Path(output_base) / _model_id
                        eval_dir.mkdir(parents=True, exist_ok=True)
                        real_ckpt = model.resolve()
                        rel = os.path.relpath(real_ckpt, eval_dir)
                        tmp = eval_dir / ".model.ckpt.tmp"
                        tmp.symlink_to(rel)
                        tmp.replace(eval_dir / "model.ckpt")  # atomic replace

                def on_train_start(self, trainer, pl_module):
                    self._rename(trainer)  # handle any pre-existing last.ckpt on resume

                def on_train_epoch_end(self, trainer, pl_module):
                    self._rename(trainer)

                def on_train_end(self, trainer, pl_module):
                    self._rename(trainer)

            callbacks.append(_RenameLastCheckpoint())

        # Custom Callback for Overall Epoch Progress Bar
        class OverallEpochProgressBar(pl.callbacks.Callback):
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                self.pbar = None

            def on_train_start(self, trainer, pl_module):
                if trainer.is_global_zero:
                    # Define X-axis for WandB plots
                    if trainer.logger and hasattr(trainer.logger.experiment, "define_metric"):
                        trainer.logger.experiment.define_metric("training/loss", step_metric="trainer/global_step")
                        trainer.logger.experiment.define_metric("training/loss_epoch", step_metric="epoch")
                        trainer.logger.experiment.define_metric("trainer/global_step", step_metric="epoch")
                        for _dm in ["obs/pos_x", "obs/pos_y", "obs/quat_w", "obs/quat_x", "obs/quat_y", "obs/quat_z"]:
                            trainer.logger.experiment.define_metric(f"training/loss_dim/{_dm}", step_metric="trainer/global_step")

                    print(f"\n[Info] Starting Training. Total Epochs: {self.total_epochs}")
                    sys.stdout.flush()
                    self.pbar = tqdm(
                        desc="Total Training Progress (Epochs)",
                        total=self.total_epochs,
                        unit="ep",
                        initial=trainer.current_epoch,
                        dynamic_ncols=True,
                        leave=True,
                        file=sys.stdout
                    )
                    self.pbar.refresh()

            def on_train_epoch_end(self, trainer, pl_module):
                if trainer.is_global_zero and self.pbar:
                    self.pbar.update(1)
                    metrics = trainer.callback_metrics
                    loss = metrics.get("training/loss") or metrics.get("loss")
                    if loss is not None:
                        self.pbar.set_postfix({"loss": f"{loss:.4f}"}, refresh=True)

            def on_train_end(self, trainer, pl_module):
                if trainer.is_global_zero and self.pbar:
                    self.pbar.close()

        callbacks.append(OverallEpochProgressBar(total_epochs))

        # Epoch loss logger + live plot updater
        class EpochLossPlotter(pl.callbacks.Callback):
            def __init__(self, log_dir):
                import json as _json
                self._json = _json
                os.makedirs(log_dir, exist_ok=True)
                self.jsonl_path = os.path.join(log_dir, "epoch_loss.jsonl")
                self.plot_path = os.path.join(log_dir, "loss_plot.png")
                self.epochs = []
                self.losses = []
                # Resume existing data
                if os.path.exists(self.jsonl_path):
                    with open(self.jsonl_path) as f:
                        for line in f:
                            try:
                                d = _json.loads(line)
                                self.epochs.append(d["epoch"])
                                self.losses.append(d["loss"])
                            except Exception:
                                pass

            def on_train_epoch_end(self, trainer, pl_module):
                if not trainer.is_global_zero:
                    return
                metrics = trainer.callback_metrics
                loss = metrics.get("training/loss") or metrics.get("loss")
                if loss is None:
                    return
                loss_val = float(loss)
                epoch = trainer.current_epoch
                step = trainer.global_step
                self.epochs.append(epoch)
                self.losses.append(loss_val)
                with open(self.jsonl_path, "a") as f:
                    f.write(self._json.dumps({"epoch": epoch, "step": step, "loss": loss_val}) + "\n")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(self.epochs, self.losses, linewidth=1)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Training Loss  (step={step}, epoch={epoch})")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(self.plot_path, dpi=100)
                plt.close(fig)

        _hydra_output_dir = str(pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]))
        callbacks.append(EpochLossPlotter(os.path.join(_hydra_output_dir, "logs")))

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger if self.logger else False,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=total_epochs,
            max_steps=effective_max_steps,
            max_time=self.cfg.training.max_time,
            enable_progress_bar=True,
        )

        try:
            trainer.fit(
                self.algo,
                train_dataloaders=train_loader,
                val_dataloaders=self._build_validation_loader(),
                ckpt_path=self.ckpt_path,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] CUDA Out of Memory — step={trainer.global_step}, epoch={trainer.current_epoch}", flush=True)
                if torch.cuda.is_available():
                    for _i in range(torch.cuda.device_count()):
                        _alloc = torch.cuda.memory_allocated(_i) / 1024**3
                        _reserved = torch.cuda.memory_reserved(_i) / 1024**3
                        print(f"[OOM] GPU {_i}: allocated={_alloc:.2f} GB, reserved={_reserved:.2f} GB", flush=True)
            raise

    def validation(self) -> None:
        """
        All validation happens here
        """
        import time as _time, datetime as _dt
        _val_wall = _dt.datetime.now().strftime("%H:%M:%S")
        print(f"[LIFECYCLE {_val_wall}] validation() start  (checkpoint={self.ckpt_path})", flush=True)
        # Initialize Tracer for MCTS tree quality analysis
        from utils.tracer import Tracer, set_default_tracer

        # Build a timestamped run_id so each run gets its own validation_anal_*.jsonl
        # instead of overwriting the fixed "validation_run.jsonl" every time.
        _val_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        _val_model_id = "unknown"
        if self.ckpt_path:
            import re as _re
            # Extract model_id from path like ".../uzrq13fa/model.ckpt"
            _m = _re.search(r"/([a-z0-9]{8})/model\.ckpt", str(self.ckpt_path))
            if _m:
                _val_model_id = _m.group(1)
        _val_run_id = f"validation_anal_{_val_ts}_{_val_model_id}"

        try:
            tracer = Tracer(
                run_id=_val_run_id,
                purpose="bidirectional_mcts_tree_quality",
                log_dir="logs",
                extra_meta={"description": "MCTS tree quality analysis"},
            )
            set_default_tracer(tracer)
            # Wrap validation with tracer context
            self._run_validation_with_tracer(tracer)
        except Exception as e:
            # Fallback: run validation without tracer if initialization fails
            print(f"[WARNING] Tracer initialization failed: {e}. Running validation without logging.")
            self._validation_impl()

    def _run_validation_with_tracer(self, tracer) -> None:
        """Internal helper to run validation within tracer context."""
        with tracer:
            self._validation_impl()
        import time as _time, datetime as _dt
        print(f"[LIFECYCLE {_dt.datetime.now().strftime('%H:%M:%S')}] validation() complete  (wandb finalizing...)", flush=True)
        # Append validation_complete to the timing tracer file (interact() already closed it,
        # but append_record() reopens in "a" mode for this one record).
        try:
            from algorithms.diffusion_forcing.df_planning import _PROC_T0 as _df_t0
            _timing_tr = getattr(getattr(self, 'algo', None), 'tracer', None)
            if _timing_tr is not None:
                _timing_tr.append_record("lifecycle.validation_complete", {
                    "elapsed_s": round(_time.time() - _df_t0, 2),
                })
        except Exception:
            pass

    @staticmethod
    def _load_ckpt_training_hparams(ckpt_path) -> dict:
        """Load training_hparams dict from a checkpoint file, if present."""
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            return ckpt.get('training_hparams', {})
        except Exception as e:
            print(f"[WARNING] Could not read training_hparams from checkpoint: {e}")
            return {}

    @staticmethod
    def _load_cache_hparams(ckpt_path) -> dict:
        """Load training params from training_config.yaml stored alongside the checkpoint.

        training_config.yaml is written by train.sh and contains the full contents of
        ckpt_df_planning.yaml (all ckpt-bound params) plus dataset metadata.
        Returns a dict in the same flat format as training_hparams so _apply_ckpt_hparams_to_cfg
        can consume it unchanged.
        """
        try:
            import yaml as _yaml
            import pathlib as _pl
            config_path = _pl.Path(ckpt_path).parent / "training_config.yaml"
            if not config_path.exists():
                return {}
            with open(config_path) as f:
                data = _yaml.safe_load(f)
            algo = data.get('algorithm', {})
            ds   = data.get('dataset', {})
            # algo is the full ckpt_df_planning.yaml content — already in the right format
            hparams = dict(algo)
            # episode_len lives in dataset section (not in ckpt_df_planning.yaml itself)
            if ds.get('episode_len') is not None:
                hparams.setdefault('episode_len', ds['episode_len'])
            if ds.get('jump') is not None:
                hparams.setdefault('jump', ds['jump'])
            if ds.get('config') is not None:
                hparams.setdefault('dataset_config', ds['config'])
            print(f"[Info] Loaded hparams from training_config.yaml: {list(hparams.keys())}")
            return hparams
        except Exception as e:
            print(f"[WARNING] Could not read training_config.yaml: {e}")
            return {}

    def _apply_ckpt_hparams_to_cfg(self, hparams: dict) -> None:
        """Override algorithm config with training_hparams saved in the checkpoint.

        Handles two cases:
        - Architecture/behavioral params (frame_stack, causal, diffusion.architecture.*,
          etc.): exist in YAML schema → standard OmegaConf.update().
        - episode_len: not in YAML schema → set via open_dict on both algorithm and
          dataset configs so n_tokens and episode loading both use the training value.
        """
        if not hparams:
            return

        # episode_len: restore dataset.episode_len from ckpt for dataset loading.
        # algorithm.episode_len is NOT restored — train uses ${dataset.episode_len}
        # interpolation; eval uses eval_episode_len from df_planning.yaml (override).
        if 'episode_len' in hparams:
            ep_len = int(hparams['episode_len'])
            OmegaConf.update(self.root_cfg, "dataset.episode_len", ep_len, merge=True)

        # sampling_timesteps is an eval-time param (DDIM step count) — NOT a model
        # architecture param. Old checkpoints may have it saved; skip to preserve the
        # value set in df_planning.yaml.
        EVAL_ONLY_DIFFUSION_KEYS = {'sampling_timesteps'}

        # All other params: declared in df_planning.yaml schema → standard update
        updates = {}
        for k, v in hparams.items():
            if k == 'episode_len':
                continue  # handled above
            if k == 'diffusion' and isinstance(v, dict):
                for dk, dv in v.items():
                    if dk in EVAL_ONLY_DIFFUSION_KEYS:
                        continue  # eval-only: do not restore from ckpt
                    if dk == 'architecture' and isinstance(dv, dict):
                        for ak, av in dv.items():
                            updates[f"algorithm.diffusion.architecture.{ak}"] = av
                    else:
                        updates[f"algorithm.diffusion.{dk}"] = dv
            else:
                updates[f"algorithm.{k}"] = v

        for dotpath, val in updates.items():
            try:
                OmegaConf.update(self.root_cfg, dotpath, val, merge=True)
            except Exception as e:
                print(f"[WARNING] Could not apply ckpt hparam {dotpath}={val}: {e}")

    def _ensure_algo_cfg_fallbacks(self) -> None:
        """Fill null algorithm config values that were not restored from training_hparams.

        Triggered in two cases:
        1. Old checkpoint with no training_hparams at all (pre-dating on_save_checkpoint).
        2. New checkpoint missing a specific key (e.g. jump added after the ckpt was saved).
        In both cases df_planning.yaml schema stubs stay null and would crash __init__.
        """
        # episode_len: train_df_planning.yaml provides `episode_len: ${dataset.episode_len}`
        # (raw, pre-jump) via Hydra interpolation — no fallback needed here.
        # eval_episode_len (df_planning.yaml) is eval-only and must not be set here.

        # jump: fall back to root config value (set via CLI e.g. jump=5, default=1).
        if OmegaConf.select(self.root_cfg, "algorithm.jump") is None:
            root_jump = OmegaConf.select(self.root_cfg, "jump")
            fallback = int(root_jump) if root_jump is not None else 1
            OmegaConf.update(self.root_cfg, "algorithm.jump", fallback, merge=True)
            print(f"[Info] algorithm.jump was null — set to {fallback} from root config (old checkpoint fallback).")

        # normalization stats + env_id + dataset: injected from dataset config.
        for key in ('observation_mean', 'observation_std', 'action_mean', 'action_std',
                    'reward_mean', 'reward_std', 'env_id', 'dataset'):
            if OmegaConf.select(self.root_cfg, f"algorithm.{key}") is None:
                val = OmegaConf.select(self.root_cfg, f"dataset.{key}")
                if val is not None:
                    OmegaConf.update(self.root_cfg, f"algorithm.{key}", val, merge=True)
                    print(f"[Info] Synced algorithm.{key} from dataset config (old checkpoint fallback).")


    def _validation_impl(self) -> None:
        """Actual validation implementation (extracted from original validation())."""
        import time as _time, datetime as _dt
        _impl_t0 = _time.time()
        print(f"[LIFECYCLE {_dt.datetime.now().strftime('%H:%M:%S')}] _validation_impl start  (building algo/loading ckpt...)", flush=True)
        if not self.algo:
            # Load ckpt-bound params only when they haven't been pre-populated by CLI args.
            # generate_jobs_generalized.py always embeds ALL of the following in the job spec,
            # so all of them being non-null means we came from job generation.
            _prepopulated_keys = [
                "algorithm.frame_stack",
                "algorithm.causal",
                "algorithm.scheduling_matrix",
                "algorithm.jump",
                "algorithm.padding_mode",
                "algorithm.context_frames",
            ]
            _cfg_prepopulated = all(
                OmegaConf.select(self.root_cfg, k) is not None for k in _prepopulated_keys
            )
            if _cfg_prepopulated:
                print("[Info] Arch params already populated by CLI args — still loading ckpt metadata for remaining null params.")
            if self.ckpt_path:
                # 1. Try training_hparams embedded in the checkpoint
                hparams = self._load_ckpt_training_hparams(self.ckpt_path)
                # 2. Fallback: training_config.yaml saved by train.sh alongside the ckpt
                if not hparams:
                    hparams = self._load_cache_hparams(self.ckpt_path)
                if hparams:
                    self._apply_ckpt_hparams_to_cfg(hparams)
                    print(
                        f"[Info] Applied training hparams: "
                        f"frame_stack={hparams.get('frame_stack')}, "
                        f"causal={hparams.get('causal')}, "
                        f"scheduling_matrix={hparams.get('scheduling_matrix')}, "
                        f"episode_len={hparams.get('episode_len')}"
                    )
                else:
                    print("[WARNING] No training params found in ckpt or training_config.yaml — "
                          "using YAML defaults. Architecture mismatch may cause errors.")
            # Last resort: fill any remaining null schema stubs
            self._ensure_algo_cfg_fallbacks()
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        _algo_built_elapsed = _time.time() - _impl_t0
        print(f"[LIFECYCLE {_dt.datetime.now().strftime('%H:%M:%S')}] algo built  ({_algo_built_elapsed:.1f}s since impl start)  — starting pl.Trainer.validate()", flush=True)
        # Log to timing tracer (opened in algo.__init__)
        try:
            from algorithms.diffusion_forcing.df_planning import _PROC_T0 as _df_t0
            _timing_tr = getattr(self.algo, 'tracer', None)
            if _timing_tr is not None:
                _timing_tr.log("lifecycle.algo_built", {
                    "elapsed_s": round(_time.time() - _df_t0, 2),
                    "since_impl_start_s": round(_algo_built_elapsed, 2),
                }, step=0, depth=0)
        except Exception:
            pass
        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.validate(
            self.algo,
            dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
        )

    def _build_dataset(self, split: str) -> Optional[torch.utils.data.Dataset]:
        if split in ["training", "test", "validation"]:
            # Decouple the dataset config from the root config to avoid side effects via interpolation
            dataset_cfg_dict = OmegaConf.to_container(self.root_cfg.dataset, resolve=True)
            dataset_cfg = OmegaConf.create(dataset_cfg_dict)
            
            # dataset_cfg.episode_len is raw (pre-jump); the dataset's __getitem__ slices
            # every `jump` frames automatically, yielding:
            #   n_tokens = episode_len // (jump * frame_stack) + 1  (including init token)
            return self.compatible_datasets[dataset_cfg._name](dataset_cfg, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
