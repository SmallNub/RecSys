import numpy as np
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

        self.code_to_item = None
        self.id_to_item = None
        self.item_embeddings = None
        self.popularity_counts = None
        self.n_centroids = None

        self._test_rec_items = []
        self._test_ild_scores = []

    @property
    def category(self):
        return self.full_hydra_config.data.datasets.category

    def training_step(self, batch, batch_idx):
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
        loss, _ = self.net.get_loss(batch)
        self.log(
            self.tplts["loss"].format(split=split, category=self.category),
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="mean",
            add_dataloader_idx=False,
        )

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

        elif self.mode == "dense":
            target = target[:, None] if target.dim() == 1 else target[:, -1:]
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

        if split == "test" and (self.code_to_item is not None or self.id_to_item is not None):
            self._accumulate_diversity(gen)

    def _decode_token_to_centroid(self, code):
        """
        Decode model token IDs back to raw centroid indices.
        Encoding: token_id = centroid_idx + special_token_offset + level * n_centroids
        Decoding: centroid_idx = token_id - special_token_offset - level * n_centroids
        """
        from src.models import SpecialTokens
        offset = len(SpecialTokens)  # 2
        return tuple(
            c - offset - (level * self.n_centroids)
            for level, c in enumerate(code)
        )

    def _accumulate_diversity(self, gen):
        """
        Accumulate per-batch recommendations for end-of-epoch diversity computation.
        gen: B x K x L — token IDs from search() with filter_preds applied (generative)
             or B x K — item vocab indices (dense)
        """
        B = gen.shape[0]
        for b in range(B):
            rec_items = []
            if self.mode == "generative":
                rec_codes = gen[b].cpu().tolist()  # K x L
                for code in rec_codes:
                    centroid_code = self._decode_token_to_centroid(code)
                    if self.code_to_item is not None:
                        item_id = self.code_to_item.get(centroid_code)
                        if item_id is not None:
                            rec_items.append(item_id)
            elif self.mode == "dense":
                rec_ids = gen[b].cpu().tolist()  # K
                for idx in rec_ids:
                    if self.id_to_item is not None:
                        item_id = self.id_to_item.get(idx)
                        if item_id is not None:
                            rec_items.append(item_id)
                    else:
                        rec_items.append(idx)

            self._test_rec_items.extend(rec_items)

            if self.item_embeddings is not None:
                embs = np.array([
                    self.item_embeddings[item]
                    for item in rec_items
                    if item in self.item_embeddings
                ])
                if len(embs) >= 2:
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    embs = embs / (norms + 1e-8)
                    sim_matrix = embs @ embs.T
                    dist_matrix = 1 - sim_matrix
                    n = len(embs)
                    ild = dist_matrix[np.triu_indices(n, k=1)].mean()
                    self._test_ild_scores.append(float(ild))

    def on_test_epoch_start(self):
        self._test_rec_items = []
        self._test_ild_scores = []

    def on_test_epoch_end(self):
        if self.code_to_item is None and self.id_to_item is None:
            return

        mean_ild = float(np.mean(self._test_ild_scores)) if self._test_ild_scores else 0.0
        rec_counts = {}
        for item in self._test_rec_items:
            rec_counts[item] = rec_counts.get(item, 0) + 1

        counts = np.array(list(rec_counts.values()), dtype=float)
        counts_sorted = np.sort(counts)
        n = len(counts_sorted)

        if n > 1:
            gini = (
                2 * np.sum(np.arange(1, n + 1) * counts_sorted)
                / (n * counts_sorted.sum())
            ) - (n + 1) / n
        else:
            gini = 0.0

        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

        self.log(f"test/{self.category}/diversity/ILD", mean_ild)
        self.log(f"test/{self.category}/popularity_bias/Gini", float(gini))
        self.log(f"test/{self.category}/popularity_bias/Entropy", entropy)

        print(f"[Diversity] ILD={mean_ild:.4f} | Gini={gini:.4f} | Entropy={entropy:.4f}")

        self._test_rec_items = []
        self._test_ild_scores = []

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