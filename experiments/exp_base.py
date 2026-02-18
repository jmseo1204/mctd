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

from omegaconf import DictConfig, OmegaConf

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
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.training.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
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
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.cfg.validation.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_test_loader(self) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test")
        shuffle = False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.test.data.shuffle
        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.test.batch_size,
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        # Build dataloader to calculate total epochs from max_steps
        train_loader = self._build_training_loader()
        num_batches = len(train_loader)
        
        effective_max_steps = self.cfg.training.max_steps
        if effective_max_steps > 0:
            total_epochs = (effective_max_steps + num_batches - 1) // num_batches
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
        
        # Custom Callback for Overall Epoch Progress Bar
        class OverallEpochProgressBar(pl.callbacks.Callback):
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                self.pbar = None

            def on_train_start(self, trainer, pl_module):
                if trainer.is_global_zero:
                    # Define X-axis for WandB plots
                    if trainer.logger and hasattr(trainer.logger.experiment, "define_metric"):
                        trainer.logger.experiment.define_metric("training/loss", step_metric="epoch")
                        trainer.logger.experiment.define_metric("training/loss_epoch", step_metric="epoch")
                        trainer.logger.experiment.define_metric("trainer/global_step", step_metric="epoch")

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
        if not self.algo:
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
                dataset_cfg.episode_len = orig_ep_len - fs
                
            return self.compatible_datasets[dataset_cfg._name](dataset_cfg, split=split)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")
