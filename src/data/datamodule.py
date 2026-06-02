from typing import Any, Optional

from pytorch_lightning import LightningDataModule
from ray.train import get_dataset_shard
from torch.utils.data import DataLoader


class RayDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        ray_datasets=None,  # Used to load the datasets without a TorchTrainer
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.ray_datasets = ray_datasets

    def get_dataset(self, name: str):
        if self.ray_datasets is not None:
            return self.ray_datasets[name]
        else:
            return get_dataset_shard(name)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.data_train = self.get_dataset("train")
            self.data_valid = self.get_dataset("valid")

        if stage == "test" or stage is None:
            self.data_valid = self.get_dataset("valid")
            self.data_test = self.get_dataset("test")

    def train_dataloader(self) -> DataLoader[Any]:
        return self.data_train.iter_torch_batches(
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            prefetch_batches=3,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return self.data_valid.iter_torch_batches(
            batch_size=self.hparams.valid_batch_size,
            drop_last=False,
            prefetch_batches=3,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return self.data_test.iter_torch_batches(
            batch_size=self.hparams.valid_batch_size,
            drop_last=False,
            prefetch_batches=3,
        )
