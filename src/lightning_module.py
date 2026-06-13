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

        # Diversity/popularity context — injected from test.py before trainer.test
        self.code_to_item = None       # dict: tuple(code) -> item_id
        self.item_embeddings = None    # dict: item_id -> np.array
        self.popularity_counts = None  # dict: item_id -> int

        # Accumulators reset each test epoch
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

        # Accumulate recommendations for diversity metrics (test only)
        # Pass batch so we can re-run search with filter_preds=False
        if split == "test" and self.code_to_item is not None:
            self._accumulate_diversity(batch)

    def _accumulate_diversity(self, batch):
        """
        Re-run search with filter_preds=False to get clean codebook codes
        that reliably map back to real items via code_to_item.
        Using the filtered gen causes misses because beam search explores
        invalid codebook combinations when filtering out seen items.
        """
        original_filter = self.net.filter_preds
        self.net.filter_preds = False

        with torch.no_grad():
            gen_clean = self.net.search(batch, n_results=10)

        self.net.filter_preds = original_filter

        # Temporary debug — remove after confirming
        if not hasattr(self, '_div_debug_done'):
            self._div_debug_done = True
            sample_code = tuple(gen_clean[0][0].cpu().tolist())
            print(f"[DIV DEBUG] gen_clean shape: {gen_clean.shape}")
            print(f"[DIV DEBUG] gen_clean min/max: {gen_clean.min().item()}/{gen_clean.max().item()}")
            print(f"[DIV DEBUG] Sample code: {sample_code}")
            print(f"[DIV DEBUG] Match in code_to_item: {sample_code in self.code_to_item}")
            hits = sum(1 for code in gen_clean[0].cpu().tolist() if tuple(code) in self.code_to_item)
            print(f"[DIV DEBUG] Codes mapping to real items (user 0): {hits}/10")

        B = gen_clean.shape[0]
        for b in range(B):
            rec_codes = gen_clean[b].cpu().tolist()  # K x L
            rec_items = []
            for code in rec_codes:
                item_id = self.code_to_item.get(tuple(code))
                if item_id is not None:
                    rec_items.append(item_id)

            self._test_rec_items.extend(rec_items)

            # ILD: mean pairwise cosine distance for this user
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
        if self.code_to_item is None:
            return

        # ILD
        mean_ild = float(np.mean(self._test_ild_scores)) if self._test_ild_scores else 0.0

        # Popularity bias — count item exposure across all recommendations
        rec_counts = {}
        for item in self._test_rec_items:
            rec_counts[item] = rec_counts.get(item, 0) + 1

        counts = np.array(list(rec_counts.values()), dtype=float)
        counts_sorted = np.sort(counts)
        n = len(counts_sorted)

        # Gini
        if n > 1:
            gini = (
                2 * np.sum(np.arange(1, n + 1) * counts_sorted)
                / (n * counts_sorted.sum())
            ) - (n + 1) / n
        else:
            gini = 0.0

        # Entropy
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

        self.log(f"test/{self.category}/diversity/ILD", mean_ild)
        self.log(f"test/{self.category}/popularity_bias/Gini", float(gini))
        self.log(f"test/{self.category}/popularity_bias/Entropy", entropy)

        print(f"[Diversity] ILD={mean_ild:.4f} | Gini={gini:.4f} | Entropy={entropy:.4f}")

        # Reset accumulators
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