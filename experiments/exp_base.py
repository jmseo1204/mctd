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
            self._ensure_episode_len_in_algo_cfg()
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        # Build dataloader to calculate total epochs from max_steps
        train_loader = self._build_training_loader()
        num_batches = len(train_loader)
        
        effective_max_steps = self.cfg.training.max_steps
        if effective_max_steps > 0:
            # effective_max_steps counts optimizer steps; num_batches counts micro-batches.
            # Multiply by accumulate_grad_batches so total_epochs reflects the true budget.
            accum = self.cfg.training.optim.accumulate_grad_batches
            total_epochs = (effective_max_steps * accum + num_batches - 1) // num_batches
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

            class _RenameLastCheckpoint(pl.callbacks.Callback):
                """Rename last.ckpt → model.ckpt after each save (atomic, no copy)."""
                def _rename(self):
                    last = _ckpt_dir / "last.ckpt"
                    model = _ckpt_dir / "model.ckpt"
                    if last.exists():
                        last.replace(model)

                def on_train_start(self, trainer, pl_module):
                    self._rename()  # handle any pre-existing last.ckpt on resume

                def on_train_epoch_end(self, trainer, pl_module):
                    self._rename()

                def on_train_end(self, trainer, pl_module):
                    self._rename()

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
            enable_progress_bar=False, # Disable default batch-level bar
        )

        trainer.fit(
            self.algo,
            train_dataloaders=train_loader,
            val_dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        # Initialize Tracer for MCTS tree quality analysis
        from utils.tracer import Tracer, set_default_tracer

        try:
            tracer = Tracer(
                run_id="validation_run",
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

    @staticmethod
    def _load_ckpt_training_hparams(ckpt_path) -> dict:
        """Load training_hparams dict from a checkpoint file, if present."""
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            return ckpt.get('training_hparams', {})
        except Exception as e:
            print(f"[WARNING] Could not read training_hparams from checkpoint: {e}")
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

        # episode_len: not declared in any YAML → inject via open_dict
        if 'episode_len' in hparams:
            ep_len = int(hparams['episode_len'])
            OmegaConf.update(self.root_cfg, "dataset.episode_len", ep_len)
            with open_dict(self.root_cfg.algorithm):
                self.root_cfg.algorithm.episode_len = ep_len

        # All other params: declared in df_base.yaml schema → standard update
        updates = {}
        for k, v in hparams.items():
            if k == 'episode_len':
                continue  # handled above
            if k == 'diffusion' and isinstance(v, dict):
                for dk, dv in v.items():
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

    def _ensure_episode_len_in_algo_cfg(self) -> None:
        """Sync algorithm.episode_len from dataset config if not already set.

        Called at training time (episode_len is not in the YAML schema, so it must
        be injected programmatically from dataset.episode_len before building the algo).
        At eval time this is a fallback for old checkpoints that have no training_hparams.
        """
        if OmegaConf.select(self.root_cfg, "algorithm.episode_len") is None:
            with open_dict(self.root_cfg.algorithm):
                self.root_cfg.algorithm.episode_len = int(self.root_cfg.dataset.episode_len)

    def _validation_impl(self) -> None:
        """Actual validation implementation (extracted from original validation())."""
        if not self.algo:
            if self.ckpt_path:
                hparams = self._load_ckpt_training_hparams(self.ckpt_path)
                if hparams:
                    self._apply_ckpt_hparams_to_cfg(hparams)
                    print(
                        f"[Info] Applied training hparams from checkpoint: "
                        f"frame_stack={hparams.get('frame_stack')}, "
                        f"causal={hparams.get('causal')}, "
                        f"scheduling_matrix={hparams.get('scheduling_matrix')}, "
                        f"episode_len={hparams.get('episode_len')}"
                    )
                else:
                    print("[WARNING] No training_hparams in checkpoint — using YAML defaults. "
                          "Architecture mismatch may cause errors for old checkpoints.")
            # Fallback: if episode_len still not set (old checkpoint), sync from dataset
            self._ensure_episode_len_in_algo_cfg()
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

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
            # FIXME: changed the meaning of episode_len 
            
            # Decouple the dataset config from the root config to avoid side effects via interpolation
            dataset_cfg_dict = OmegaConf.to_container(self.root_cfg.dataset, resolve=True)
            dataset_cfg = OmegaConf.create(dataset_cfg_dict)
            
            # If the algorithm uses frame_stack, adjust the dataset's episode_len 
            # so that it provides exactly (orig_ep_len - fs + 1) frames.
            # This ensures that n_tokens = (frames - 1) // fs + 1 matches the original definition.
            if hasattr(self.root_cfg, "algorithm") and hasattr(self.root_cfg.algorithm, "frame_stack"):
                fs = self.root_cfg.algorithm.frame_stack
                # Use the original episode_len from the algorithm config
                orig_ep_len = self.root_cfg.algorithm.episode_len
                jump = getattr(dataset_cfg, 'jump', 1)
                if jump <= 1:
                    # jump=1: need (episode_len) divisible by frame_stack
                    dataset_cfg.episode_len = orig_ep_len - fs
                else:
                    # jump>1: need (episode_len // jump) divisible by frame_stack
                    # i.e., episode_len must be a multiple of (jump * frame_stack)
                    unit = jump * fs
                    dataset_cfg.episode_len = (orig_ep_len // unit) * unit
                
            return self.compatible_datasets[dataset_cfg._name](dataset_cfg, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
