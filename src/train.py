from datetime import datetime
from pprint import pprint
from typing import Optional

import hydra
import pyarrow.fs
import pytorch_lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from src.utils.ranked_logger import RankedLogger, log_hyperparameters
from src.utils.tools import selective_oc_resolver

OmegaConf.register_new_resolver("eval", eval)
log = RankedLogger(__name__, rank_zero_only=True)


def now_to_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    selective_oc_resolver(cfg, which=["now", "oc.env", "eval"])

    pprint(OmegaConf.to_container(cfg))

    assert cfg.paths.protocol in [
        "viewfs",
        "file",
    ], "This code has only been tested for viewfs:// and file://."

    name = f"{cfg.task_name}_{now_to_str()}"

    cfg.trainer.logger.name = name

    torch.set_float32_matmul_precision(cfg.get("fp32_matmul_precision", "high"))
    torch.backends.mha.set_fastpath_enabled(False)

    L.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating datasets")
    datasets = hydra.utils.instantiate(cfg.data.datasets, paths=cfg.paths)

    from src.models import SpecialTokens

    first_ds = next(iter(datasets.values())) if datasets else None
    if (
        first_ds is not None
        and hasattr(first_ds, "pp")
        and hasattr(first_ds.pp, "item_to_id")
    ):
        vocab_size = len(first_ds.pp.item_to_id) + len(SpecialTokens)
        log.info(f"Detected actual vocab size from dataset: {vocab_size}")
        if (
            "model" in cfg
            and "net" in cfg.model
            and cfg.model.net.get("_target_") == "src.models.sasrec.SASRec"
        ):
            cfg.model.net.vocab_size = vocab_size
            log.info(f"Overrode model.net.vocab_size to {vocab_size} for SASRec")

    log.info("Instantiating datamodule")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, datasets=datasets
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    print(model)
    model.full_hydra_config = cfg

    log.info("Instantiating trainer")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

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


if __name__ == "__main__":
    main()
