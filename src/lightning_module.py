import torch
from pytorch_lightning import LightningModule


class LitModule(LightningModule):
    def __init__(self, net, optimizer, mode, scheduler=None):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.mode = mode
        self.Ks = [1, 5, 10, 20]

        assert self.mode in ["generative", "dense"]

        self.tplts = {
            "loss": "{split}/{category}/loss",
            "HR": "{split}/{category}/HR@{K}",
            "NDCG": "{split}/{category}/NDCG@{K}",
        }

        self.dcg_denom = torch.log2(torch.arange(1, max(self.Ks) + 1) + 1).view(1, -1)

    @property
    def category(self):
        return self.full_hydra_config.data.datasets.category

    def training_step(self, batch, batch_idx):
        # Check if the entire batch is padding before calling the model
        targets = batch["target"]
        if (targets[:, :, 0] == -100).all():
            return None  # PyTorch Lightning safely skips optimization for this step!

        loss, _ = self.net.get_loss(batch)

        self.log(
            self.tplts["loss"].format(
                split="train",
                category=self.category,
            ),
            loss,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def _shared_eval_step(self, batch, split):
        # Forward + Loss
        loss, _ = self.net.get_loss(batch)

        # Log the loss - Dataset-wise
        self.log(
            self.tplts["loss"].format(split=split, category=self.category),
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            add_dataloader_idx=False,
        )

        # Recall
        # Generated : Batch x Beams x L=4
        # Dense : Batch x Beams
        gen = self.net.search(batch, n_results=max(self.Ks))

        target = batch["target"]

        if self.dcg_denom.device != target.device:
            self.dcg_denom = self.dcg_denom.to(target.device)

        if self.mode == "generative":
            target = target[:, None, :] if target.dim() == 2 else target[:, -1:, :]

            B, _, L = target.shape

            assert gen.shape == (B, max(self.Ks), L)
            assert target.shape == (B, 1, L)

            all_hits = (gen == target).all(dim=-1)  # Broadcast along beam

            RatK = (all_hits.cumsum(dim=1) > 0).float().mean(dim=0)
            DCG = (
                ((2 ** all_hits.float() - 1) / self.dcg_denom).cumsum(dim=1).mean(dim=0)
            )
            # Note : we have a single positive, so IDCG is (2**1 - 1) / log2(2) = 1, we ignore it.

        elif self.mode == "dense":
            # Take a slice of the last items
            target = target[:, None] if target.dim() == 1 else target[:, -1:]

            # Gen : B x Beams, Target : B x 1 -> B x Beams
            all_hits = gen == target

            RatK = (all_hits.cumsum(dim=1) > 0).float().mean(dim=0)
            DCG = (
                ((2 ** all_hits.float() - 1) / self.dcg_denom).cumsum(dim=1).mean(dim=0)
            )

        for K in self.Ks:
            self.log(
                self.tplts["HR"].format(split=split, category=self.category, K=K),
                RatK[K - 1],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                reduce_fx="mean",
                add_dataloader_idx=False,
            )
            self.log(
                self.tplts["NDCG"].format(split=split, category=self.category, K=K),
                DCG[K - 1],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                reduce_fx="mean",
                add_dataloader_idx=False,
            )

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "test")

    def configure_optimizers(self):
        params = (
            self.net.get_param_groups()
            if hasattr(self.net, "get_param_groups")
            else self.net.parameters()
        )

        optimizer = self.hparams.optimizer(params)

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
