import os
import pytorch_lightning as pl
from omegaconf import OmegaConf


class SaveConfigCallback(pl.callbacks.Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            ckpt_cb = trainer.checkpoint_callback
            if ckpt_cb and ckpt_cb.dirpath:
                os.makedirs(ckpt_cb.dirpath, exist_ok=True)
                cfg_path = os.path.join(ckpt_cb.dirpath, "config.yaml")
                with open(cfg_path, "w") as f:
                    f.write(
                        OmegaConf.to_yaml(pl_module.full_hydra_config, resolve=True)
                    )
