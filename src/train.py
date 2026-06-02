from datetime import datetime
from pprint import pprint
from typing import Optional

import hydra
import pyarrow.fs
import pytorch_lightning as L
import ray
import ray.train
import ray.train.torch
import torch
from omegaconf import DictConfig, OmegaConf
from ray.train.lightning import prepare_trainer

from src.utils.ranked_logger import RankedLogger, log_hyperparameters
from src.utils.tools import selective_oc_resolver

OmegaConf.register_new_resolver("eval", eval)
log = RankedLogger(__name__, rank_zero_only=True)


def now_to_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def train(cfg):
    # Set precision to use tensor cores - highest | high | medium
    torch.set_float32_matmul_precision(cfg.get("fp32_matmul_precision", "high"))
    # Disable MHA Fast path (in this version it can NaN because of left-padding)
    torch.backends.mha.set_fastpath_enabled(False)

    log.info("Instantiating datamodule")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    print(model)
    # Storing it here so that we can access it in the callback
    model.full_hydra_config = cfg

    log.info("Instantiating trainer")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer = prepare_trainer(trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    log.info("Logging hyperparameters!")
    log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    return trainer.callback_metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Avoid "unknown resolver" error - resolve hydra-specific keys.
    # Plus we don't want each worker to have a different experiment name if they are spawned a second later.
    selective_oc_resolver(cfg, which=["now", "oc.env", "eval"])

    ray.init(runtime_env=OmegaConf.to_container(cfg.ray.runtime_env))
    print(ray.cluster_resources())
    pprint(OmegaConf.to_container(cfg))

    assert cfg.paths.protocol in [
        "viewfs",
        "file",
    ], "This code has only been tested for viewfs:// and file://."
    filesystem = (
        pyarrow.fs.FileSystem.from_uri("viewfs://root")[0]
        if cfg.paths.protocol == "viewfs"
        else None
    )

    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)
    name = f"{cfg.task_name}_{now_to_str()}"

    cfg.trainer.logger.name = name

    trainable = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train,
        train_loop_config=cfg,
        scaling_config=hydra.utils.instantiate(cfg.ray.scaling_config),
        run_config=hydra.utils.instantiate(
            cfg.ray.run_config,
            name=name,
            storage_path=cfg.ray.run_config.storage_path,
            storage_filesystem=filesystem,
            callbacks=[],
        ),
        datasets=ray_datasets,
    )

    return trainable.fit()


if __name__ == "__main__":
    main()
