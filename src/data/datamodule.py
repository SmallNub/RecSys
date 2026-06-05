from typing import Any, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class RecDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        datasets=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.datasets = datasets

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.data_train = self.datasets["train"]
            self.data_valid = self.datasets["valid"]

        if stage == "test" or stage is None:
            self.data_valid = self.datasets["valid"]
            self.data_test = self.datasets["test"]

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_valid,
            batch_size=self.hparams.valid_batch_size,
            drop_last=False,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.valid_batch_size,
            drop_last=False,
            num_workers=4,
        )
