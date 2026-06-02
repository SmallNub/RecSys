import os
import shutil
from tempfile import mkdtemp

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from ray import train
from ray.train import Checkpoint


def remove_then_create(path):
    """Remove the directory if it exists and create it."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class CustomReportCallback(pl.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        root = f"{os.getcwd()}/ray_report_checkpoints"
        remove_then_create(root)

        self.global_rank = train.get_context().get_world_rank()

        self.tmpdir_prefix = os.path.join(
            mkdtemp(dir=root),
            str(self.global_rank),
        )
        remove_then_create(self.tmpdir_prefix)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        # Creates a checkpoint dir with fixed name
        tmpdir = os.path.join(self.tmpdir_prefix, str(trainer.current_epoch))
        os.makedirs(tmpdir, exist_ok=True)

        # Save checkpoint to local
        ckpt_path = os.path.join(tmpdir, "checkpoint.ckpt")
        trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Save config to local
        cfg_path = os.path.join(tmpdir, "config.yaml")
        with open(cfg_path, "w") as f:
            f.write(OmegaConf.to_yaml(pl_module.full_hydra_config, resolve=True))

        # Report to train session
        checkpoint = Checkpoint.from_directory(tmpdir)

        # Fetch metrics
        metrics = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
        }

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        train.report(metrics=metrics, checkpoint=checkpoint)

        shutil.rmtree(tmpdir)
