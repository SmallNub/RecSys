import logging
import os
import random
from collections import defaultdict
from math import ceil
from uuid import uuid4
from datetime import datetime

import fsspec
import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from torch import optim
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from src.utils.tools import patch_fsspec


class LinearDecayScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, cooldown_steps, factor=0.1):
        self.optimizer = optimizer
        self.factor = factor
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_step_sizes = [
            (base_lr - self.factor * base_lr) / warmup_steps
            for base_lr in self.base_lrs
        ]
        self.cooldown_step_sizes = [
            (self.factor * base_lr - base_lr) / cooldown_steps
            for base_lr in self.base_lrs
        ]
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr * self.factor for base_lr in self.base_lrs]
        elif self.last_epoch < self.warmup_steps:
            return [
                base_lr * self.factor + step_size * self.last_epoch
                for base_lr, step_size in zip(self.base_lrs, self.warmup_step_sizes)
            ]
        elif (self.last_epoch - self.warmup_steps) < self.cooldown_steps:
            return [
                base_lr + step_size * (self.last_epoch - self.warmup_steps)
                for base_lr, step_size in zip(self.base_lrs, self.cooldown_step_sizes)
            ]
        else:
            return [base_lr * self.factor for base_lr in self.base_lrs]


class Trainer(object):
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.logger = logging.getLogger()
        self.is_hyper = self.config.model.get("type", "euclidean") == "hyperbolic"

        self.epochs = config.optim.epochs
        self.eval_step = min(config.optim.eval_step, self.epochs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ckpt_dir = config.ckpt_dir  
        self.local_dir = self.ckpt_dir
        os.makedirs(self.local_dir, exist_ok=True)

        self.last_ckpt = "last_model.pth"

        self.optimizer = self._build_optimizer()
        if config.optim.use_scheduler:
            self.scheduler = LinearDecayScheduler(
                self.optimizer,
                warmup_steps=config.optim.warmup,
                cooldown_steps=self.epochs - config.optim.warmup,
            )
        else:
            self.scheduler = None
        self.model = self.model.to(self.device)

    def _build_optimizer(self):
        def _select(n):
            return "embedding" in n or "siglip" in n

        no_wd_group = {
            "params": [
                p
                for n, p in self.model.named_parameters()
                if _select(n) and p.requires_grad
            ],
            "weight_decay": 0.0,
        }
        other_params = {
            "params": [
                p
                for n, p in self.model.named_parameters()
                if not _select(n) and p.requires_grad
            ],
            "weight_decay": self.config.optim.weight_decay,
        }

        optimizer = optim.AdamW(
            [no_wd_group, other_params],
            betas=(0.9, self.config.optim.get("beta2", 0.999)),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )
        return optimizer

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()

        # Dynamic Curvature Annealing Calculations
        anneal_period = max(1, int(self.epochs * 0.10))
        if epoch_idx < anneal_period:
            c = 0.01 + (1.0 - 0.01) * (epoch_idx / anneal_period)
        else:
            c = 1.0

        total_loss = 0
        
        model_name = "HYPERBOLIC" if self.is_hyper else "EUCLIDEAN"
        desc_str = f"[{model_name}] Train {epoch_idx}"
        if self.is_hyper:
            desc_str += f" (c={c:.4f})"

        iter_data = (
            tqdm(train_data, total=len(train_data), ncols=100, desc=desc_str)
            if self.config.loss.contrastive_weight > 0
            else tqdm(range(len(train_data)), desc=desc_str)
        )

        metrics = defaultdict(float)
        for _, data in enumerate(iter_data):
            if self.config.loss.contrastive_weight == 0:
                data = {"items": None, "timelines": None}
            else:
                data = {k: v.to(self.device, non_blocking=True) for k, v in data.items()}

            self.optimizer.zero_grad()

            # --- HYBRID TOGGLE: Safe parameter routing ---
            if self.is_hyper:
                loss, b_metrics = self.model.training_loss(**data, c=c)
            else:
                loss, b_metrics = self.model.training_loss(**data)

            self._check_nan(loss)
            loss.backward()

            # =================================================================
            # FIX ADDED HERE: Hyperbolic Gradient Explosion Protection
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # =================================================================

            self.optimizer.step()

            total_loss += loss.item() / len(iter_data)
            for k, v in b_metrics.items():
                if hasattr(v, "item"):
                    v = v.item()
                metrics[k] += v / len(iter_data)

        return total_loss, metrics

    @torch.no_grad()
    def _valid_epoch(self):
        self.model.eval()

        indices_list = []
        
        # --- HYBRID TOGGLE ---
        if self.is_hyper:
            indices, _ = self.model.get_indices(c=1.0) 
        else:
            indices, _ = self.model.get_indices()

        for index in indices:
            code = tuple(index.cpu().numpy().tolist())
            indices_list.append(code)

        levels = list(zip(*indices_list))
        token_usage_per_level = {
            f"L{i}_token_usage": len(set(level)) for i, level in enumerate(levels)
        }

        freq_count = defaultdict(int)
        for c_code in indices_list:
            freq_count[c_code] += 1
        max_value = max(list(freq_count.values()))
        min_value = min(list(freq_count.values()))
        mean_value = np.mean(list(freq_count.values()))

        indices_set = set(indices_list)
        collision_rate = (len(indices_list) - len(indices_set)) / len(indices_list)

        return {
            "max_collisions": max_value,
            "min_collisions": min_value,
            "mean_collisions": mean_value,
            "collision_rate": collision_rate,
            **token_usage_per_level,
        }

    def _save_checkpoint(self, epoch, ckpt_file):
        ckpt_path = os.path.join(self.local_dir, ckpt_file)
        state = {
            "config": self.config,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(f"Saving current: {ckpt_path}")

    def fit(self, train_data):
        model_name = "HYPERBOLIC" if self.is_hyper else "EUCLIDEAN"
        print(f"\n{'='*60}")
        print(f">>>> STARTING {model_name} TRAINING LOOP <<<<")
        print(f"{'='*60}\n")

        for epoch_idx in range(self.epochs):
            if self.scheduler is not None:
                self.scheduler.step()

            train_loss, metrics = self._train_epoch(train_data, epoch_idx)

            wandb.log(
                {
                    "epoch": epoch_idx,
                    "train_loss": train_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    **metrics,
                }
            )

            if (epoch_idx + 1) % self.eval_step == 0:
                print("=" * 100)
                print(">>>> EVALUATING <<<<")
                metrics = self._valid_epoch()
                wandb.log({"epoch": epoch_idx, **metrics})

        self._save_checkpoint(epoch_idx, ckpt_file=self.last_ckpt)
        return os.path.join(self.local_dir, self.last_ckpt)


class _Dataset(torch.utils.data.IterableDataset):
    def __init__(self, timelines, items_to_row, bs, items_cut):
        self.items_to_row = items_to_row
        self.timelines = np.array(
            [
                np.array(
                    [self.items_to_row[item] for item in row["timeline"]]
                    + [-1] * (items_cut - len(row["timeline"]))
                )
                for i, row in timelines.iterrows()
            ],
            dtype=object,
        )
        self.bs = bs
        self.items_cut = items_cut

    def __len__(self):
        return len(self.timelines) // self.bs

    def _to_size(self, tl):
        if len(tl) > self.items_cut:
            start = random.randint(0, len(tl) - self.items_cut)
            tl = tl[start : start + self.items_cut]
        return tl

    def __iter__(self):
        w_info = torch.utils.data.get_worker_info()
        world_size = w_info.num_workers if w_info else 1
        worker_id = w_info.id if w_info else 0
        T = len(self)
        L = ceil(T / world_size) if T % world_size > worker_id else T // world_size
        self._indices = np.random.permutation(len(self.timelines))
        for i in range(L):  
            yield self.__make_batch(i)

    def __make_batch(self, i):
        sel_indices = self._indices[i * self.bs : (i + 1) * self.bs]
        sel_timelines = self.timelines[sel_indices]
        sel_timelines = np.array([self._to_size(t) for t in sel_timelines])
        all_items = np.unique(np.concatenate(sel_timelines))  
        if all_items[0] == -1:  
            all_items = all_items[1:]
        return {
            "items": torch.from_numpy(all_items),
            "timelines": torch.from_numpy(sel_timelines),
        }


class DataLoader:
    def __init__(self, items, timelines, bs, cut):
        self.timelines = timelines
        self.items_to_row = {item: i for i, item in enumerate(items)}
        self.ds = _Dataset(self.timelines, self.items_to_row, bs, cut)
        self.dl = torch.utils.data.DataLoader(
            self.ds,
            collate_fn=self._collate_fn,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

    def _collate_fn(self, batch):  
        assert len(batch) == 1, "Batch size should be 1"
        return batch[0]

    def __iter__(self):
        yield from self.dl

    def __len__(self):
        return len(self.dl)


def make_quantized_df(quant_method, config, product_id, model, filesystem):
    print(f"Generating output for {quant_method}...")
    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)

    quantized_path = config.paths.semantic_ids_tplt.format(
        emb_method=config.data.emb_method,
        quant_method=quant_method,
        category=config.data.category,
    )
    model_path = config.paths.semantic_model_tplt.format(
        emb_method=config.data.emb_method,
        quant_method=quant_method,
        category=config.data.category,
    )

    print("Forward.")
    # --- HYBRID TOGGLE ---
    is_hyper = config.model.get("type", "euclidean") == "hyperbolic"
    if is_hyper:
        indices, _ = model.get_indices(use_sk=False, c=1.0)
    else:
        indices, _ = model.get_indices(use_sk=False)
        
    indices = indices.cpu().numpy().astype(np.int32)

    print("Preparing dataframe.")
    series = []
    for _ in range(indices.shape[1]):
        series.append(pd.Series(indices[:, _], name=f"L{_}"))
    quant_df = pd.concat(series, axis=1)
    quant_df["product_id"] = product_id

    print("Saving quantized dataframe.")
    quant_df.to_parquet(quantized_path, filesystem=filesystem)

    sd = model.state_dict()
    del sd["embeddings"]  
    with fs.open(model_path, "wb") as f:
        torch.save(
            {
                "config": config,
                "state_dict": sd,
                "epoch": config.optim.epochs,  
            },
            f,
        )


def make_name(config, timestamp):
    model_type = config.model.get("type", "euclidean")
    prefix = "COSETTE_HYPER" if model_type == "hyperbolic" else "COSETTE"
    name = f"{prefix}_{config.data.category}_{timestamp}"
    if config.marker is not None:
        name += f"_{config.marker}"
    return name


def make_cosette_embs(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.ckpt_dir = os.path.join(config.ckpt_dir, timestamp)

    # --- HYBRID TOGGLE: Dynamic Model Import ---
    model_type = config.model.get("type", "euclidean")
    if model_type == "hyperbolic":
        print(">>> INITIALIZING HYPERBOLIC COSETTE <<<")
        from src.models.cosette_hyper import COSETTE
    else:
        print(">>> INITIALIZING EUCLIDEAN COSETTE <<<")
        version_type = config.model.get("version", "default")
        if version_type == "default":
            from src.models.cosette import COSETTE
        elif version_type == "alt":
            from src.models.cosette_alt import COSETTE

    embeddings_path = config.paths.embeddings_tplt.format(
        emb_method=config.data.emb_method, category=config.data.category
    )
    train_timelines_path = config.paths.timelines_tplt.format(
        category=config.data.category, split="train"
    )
    items_path = config.paths.unique_items_tplt.format(category=config.data.category)

    patch_fsspec()
    fs = fsspec.filesystem(config.paths.protocol)

    with fs.open(items_path, "rb") as f:
        items = pd.read_pickle(f)
    embeddings_df = pd.read_parquet(embeddings_path, filesystem=fs)
    train_timelines = pd.read_parquet(train_timelines_path, filesystem=fs)

    embeddings_df.set_index("product_id", inplace=True)
    embeddings_df = embeddings_df.loc[items]
    embs_block = torch.from_numpy(
        np.stack(embeddings_df["embedding"].values, dtype=np.float32)
    )
    embs_block = embs_block.to("cuda")
    embeddings_df.reset_index(inplace=True)

    train_loader = DataLoader(
        items=items,
        timelines=train_timelines,
        bs=config.optim.batch_size,
        cut=config.loss.sequence_length,
    )

    model = COSETTE(
        embs_block=embs_block,
        in_dim=embs_block.shape[-1],
        layers=config.model.layers,
        dropout_prob=config.optim.dropout_prob,
        loss_weights={
            "quantization": 1,
            "reconstruction": config.loss.reconstruction_weight,
            "contrastive": config.loss.contrastive_weight,
            "latent_consistency": config.loss.get("latent_consistency_weight", 0.0),
            "latent_consistency_l1_loss": config.loss.get("latent_consistency_l1_weight", 0.0),
            "reconstruction_l1_loss": config.loss.get("reconstruction_l1_weight", 0.0),
        },
        tau=config.loss.tau,
        bias=config.loss.bias,
        freeze_tau=config.loss.freeze_tau,
        freeze_bias=config.loss.freeze_bias,
        n_centroids_list=config.centroids.n_centroids_list,
        kmeans_init=config.centroids.kmeans_init,
        kmeans_iters=config.centroids.kmeans_iters,
        sk_epsilons=config.centroids.sk_epsilons,
        sk_iters=config.centroids.sk_iters,
    )

    print("Model : ", model)
    quant_method = make_name(config, timestamp)

    wandb.login()
    wandb.init(
        project=config.paths.wandb_project_name,
        entity=config.paths.wandb_entity,
        mode=config.paths.wandb_mode,
        config=OmegaConf.to_container(config, resolve=True),
        name=quant_method,
    )

    trainer = Trainer(config, model, {"product_id": embeddings_df["product_id"].values})
    last_ckpt = trainer.fit(train_loader)

    sd = torch.load(last_ckpt, weights_only=False, map_location="cpu")
    model.load_state_dict(sd["state_dict"])
    model.eval()
    model.cuda()

    make_quantized_df(
        quant_method=quant_method,
        config=config,
        product_id=embeddings_df["product_id"].values,
        model=model,
        filesystem=fs,
    )


@hydra.main(config_path="../configs", config_name="2_train_cosette", version_base="1.2")
def main(config):
    make_cosette_embs(config)


if __name__ == "__main__":
    main()