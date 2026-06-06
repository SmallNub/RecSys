from dataclasses import dataclass
from datetime import datetime
import glob
import math
import os
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange

from src.models import SpecialTokens
from src.models.cosette import COSETTE


@dataclass
class TransformerConfig:
    n_layers: int
    d_head: int
    d_model: int
    vocab_size: int
    dropout: float
    emb_dropout: float
    seq_len: int
    sigma: float = 0.0
    ode_steps: int = 10
    fourier_dim: int = 32


def get_latest_timestamp_dir(base_dir_path: str, timestamp_format: str = "%Y%m%d_%H%M%S") -> Path:
    base_dir = Path(base_dir_path)
    latest_dir = None
    latest_time = None

    if not base_dir.exists():
        print(f"[DEBUG] Base directory does not exist: {base_dir_path}")
        return None

    for entry in base_dir.iterdir():
        if entry.is_dir():
            try:
                folder_time = datetime.strptime(entry.name, timestamp_format)
                if latest_time is None or folder_time > latest_time:
                    latest_time = folder_time
                    latest_dir = entry
            except ValueError:
                continue
    return latest_dir


def load_latest_catalog_tokens(device="cuda"):
    cosette_base_dir = os.path.join(os.getcwd(), "outputs", "checkpoints", "cosette")
    print(f"[DEBUG] Searching for checkpoints in: {cosette_base_dir}")
    
    latest_timestamp_folder = get_latest_timestamp_dir(cosette_base_dir)
    if latest_timestamp_folder is None:
        print("[DEBUG] No valid timestamp folders found.")
        return None

    latest_checkpoint_path = latest_timestamp_folder / "last_model.pth"
    print(f"[DEBUG] Identified latest checkpoint path target: {latest_checkpoint_path}")
    
    if not latest_checkpoint_path.exists():
        print(f"[DEBUG] Target file missing: {latest_checkpoint_path}")
        return None

    try:
        checkpoint = torch.load(str(latest_checkpoint_path), map_location=torch.device("cpu"), weights_only=False)
        print("[DEBUG] Checkpoint successfully loaded into memory via torch.load.")
    except Exception as e:
        print(f"[DEBUG] Crash during torch.load execution: {e}")
        return None
    
    model_cfg = checkpoint.pop("config", None)
    state_dict = checkpoint.pop("state_dict", None)

    if model_cfg is None:
        print("[DEBUG] Unpacking failed: 'config' block key is missing.")
        return None
    if state_dict is None:
        print("[DEBUG] Unpacking failed: 'state_dict' block key is missing.")
        return None

    emb_method = getattr(model_cfg.data, "emb_method", "sentence-t5-xl")
    embeddings_path = os.path.join(os.getcwd(), "datasets", "data", "embeddings", emb_method, "items.parquet")
    print(f"[DEBUG] Attempting to load external parquet file from target: {embeddings_path}")

    if not os.path.exists(embeddings_path):
        print(f"[DEBUG] Target path directory structure does not exist. Attempting lookups using wildcard fallback strategy...")
        wildcard_path = os.path.join(os.getcwd(), "datasets", "data", "embeddings", "**", "*.parquet")
        fallback_paths = glob.glob(wildcard_path, recursive=True)
        if fallback_paths:
            embeddings_path = fallback_paths[0]
            print(f"[DEBUG] Re-mapped tracking path target dynamically to file: {embeddings_path}")
        else:
            print(f"[DEBUG] Error: Could not locate fallback items parquet file using query sequence {wildcard_path}")
            return None

    try:
        embs_df = pd.read_parquet(embeddings_path)
        embs_block = np.stack(embs_df["embedding"].values)
        print(f"[DEBUG] Successfully loaded parquet. Shape: {embs_block.shape}")
    except Exception as e:
        print(f"[DEBUG] Failed reading external data frame properties matrix: {e}")
        return None

    in_dim = embs_block.shape[-1]

    try:
        cosette_model = COSETTE(
            embs_block=None,
            in_dim=in_dim,
            layers=model_cfg.model.layers,
            n_centroids_list=model_cfg.centroids.n_centroids_list,
            dropout_prob=model_cfg.optim.dropout_prob,
            loss_weights={
                "quantization": 1,
                "reconstruction": model_cfg.loss.reconstruction_weight,
                "contrastive": model_cfg.loss.contrastive_weight,
            },
            tau=model_cfg.loss.tau,
            bias=model_cfg.loss.bias,
            freeze_tau=model_cfg.loss.freeze_tau,
            freeze_bias=model_cfg.loss.freeze_bias,
            kmeans_init=model_cfg.centroids.kmeans_init,
            kmeans_iters=model_cfg.centroids.kmeans_iters,
            sk_epsilons=model_cfg.centroids.sk_epsilons,
            sk_iters=model_cfg.centroids.sk_iters,
        )
        print("[DEBUG] COSETTE module instantiation complete.")
    except Exception as e:
        print(f"[DEBUG] Crash during COSETTE constructor logic: {e}")
        return None

    try:
        cosette_model.load_state_dict(state_dict)
        print("[DEBUG] Load state dict applied successfully to COSETTE.")
    except Exception as e:
        print(f"[DEBUG] Crash within load_state_dict assignment transfer: {e}")
        return None

    cosette_model = cosette_model.to(device)
    cosette_model.eval()

    try:
        print("[DEBUG] Executing get_indices projection sweep across external item embeddings matrix...")
        with torch.no_grad():
            embeddings_tensor = torch.from_numpy(embs_block).to(device).float()
            catalog_tokens, _ = cosette_model.get_indices(embeddings=embeddings_tensor, use_sk=False)
            catalog_tokens = catalog_tokens.detach().cpu()
        print(f"[DEBUG] Indexing complete. Tokens matrix shape: {catalog_tokens.shape}")
    except Exception as e:
        print(f"[DEBUG] Crash inside get_indices operation: {e}")
        return None

    return catalog_tokens


class MARIUS(torch.nn.Module):
    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        tie_embeddings=False,
        filter_preds=False,
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds

        self.sigma = getattr(self.depth_cfg, "sigma", 0.0)
        self.ode_steps = getattr(self.depth_cfg, "ode_steps", 10)
        self.fourier_dim = getattr(self.depth_cfg, "fourier_dim", 32)

        assert (
            self.depth_cfg.vocab_size == self.temporal_cfg.vocab_size
        ), "Vocab size mismatch"

        if self.temporal_cfg.emb_dropout is None:
            self.temporal_cfg.emb_dropout = self.temporal_cfg.dropout
        if self.depth_cfg.emb_dropout is None:
            self.depth_cfg.emb_dropout = self.depth_cfg.dropout

        self.temp_emb = torch.nn.Embedding(
            self.temporal_cfg.vocab_size,
            self.temporal_cfg.d_model,
            padding_idx=SpecialTokens.PAD.value,
        )

        self.tie_embeddings = tie_embeddings
        if tie_embeddings:
            self.depth_emb = self.temp_emb
        else:
            self.depth_emb = torch.nn.Embedding(
                self.depth_cfg.vocab_size,
                self.depth_cfg.d_model,
                padding_idx=SpecialTokens.PAD.value,
            )

        self.temp_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.temporal_cfg.seq_len, self.temporal_cfg.d_model))
        )
        
        self.register_buffer(
            "fourier_frequencies", 
            torch.randn(self.fourier_dim // 2) * 2 * math.pi
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.fourier_dim, self.depth_cfg.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.depth_cfg.d_model, self.depth_cfg.d_model),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokens_map = load_latest_catalog_tokens(device=device)
        if tokens_map is not None:
            self.register_buffer("catalog_tokens", tokens_map)
        else:
            self.catalog_tokens = None

        self.temp_dropout = torch.nn.Dropout(self.temporal_cfg.emb_dropout)
        self.depth_dropout = torch.nn.Dropout(self.depth_cfg.emb_dropout)

        self.temp_tf = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.temporal_cfg.d_model,
                nhead=self.temporal_cfg.d_model // self.temporal_cfg.d_head,
                dim_feedforward=self.temporal_cfg.d_model * 4,
                dropout=self.temporal_cfg.dropout,
                batch_first=True,
                norm_first=True,
            ),
            enable_nested_tensor=False,
            num_layers=self.temporal_cfg.n_layers,
            norm=torch.nn.LayerNorm(self.temporal_cfg.d_model),
        )

        self.depth_tf = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.depth_cfg.d_model,
                nhead=self.depth_cfg.d_model // self.depth_cfg.d_head,
                dim_feedforward=self.depth_cfg.d_model * 4,
                dropout=self.depth_cfg.dropout,
                batch_first=True,
                norm_first=True,
            ),
            enable_nested_tensor=False,
            num_layers=self.depth_cfg.n_layers,
            norm=torch.nn.LayerNorm(self.depth_cfg.d_model),
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    (self.temporal_cfg.seq_len, self.temporal_cfg.seq_len),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

        self.mid_proj = torch.nn.Linear(
            self.temporal_cfg.d_model, self.depth_cfg.d_model
        )
        self.criterion = torch.nn.MSELoss()

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n
        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def get_fourier_time_embedding(self, t):
        phases = t @ self.fourier_frequencies.unsqueeze(0)
        fourier_features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.time_mlp(fourier_features)

    def temporal_forward(self, input):
        B, L, K = input.shape
        input_embs = self.temp_emb(input).sum(dim=-2)
        input_embs += self.temp_pos_emb[:, : input.shape[1], :]
        input_embs = self.temp_dropout(input_embs)

        out = self.temp_tf(
            input_embs,
            mask=self.causal_mask[:L, :L],
            src_key_padding_mask=input[:, :, 0] == SpecialTokens.PAD.value,
        )
        return out

    def depth_forward(self, x_t, t, mid_tokens):
        t_emb = self.get_fourier_time_embedding(t).unsqueeze(1)
        x_t_emb = x_t.unsqueeze(1)
        x_t_emb = self.depth_dropout(x_t_emb)

        dec_inputs = torch.cat([mid_tokens, t_emb, x_t_emb], dim=1)
        depth_preds = self.depth_tf(dec_inputs)
        return depth_preds[:, -1, :]

    def train_forward(self, input, target):
        B, L, _ = input.shape
        temporal_tokens = self.temporal_forward(input)
        mid_tokens = self.mid_proj(temporal_tokens)
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")

        target_flat = rearrange(target, "b l k -> (b l) k")
        keep = target_flat[:, 0] != -100

        mid_tokens = mid_tokens[keep]
        target_flat = target_flat[keep]

        x_1 = self.depth_emb(target_flat).sum(dim=1)

        input_flat = rearrange(input, "b l k -> (b l) k")[keep]
        x_0 = self.depth_emb(input_flat).sum(dim=1)

        time_steps = torch.linspace(0.0, 0.9, steps=L, device=input.device)
        t_matrix = time_steps.unsqueeze(0).repeat(B, 1)
        t_flat = rearrange(t_matrix, "b l -> (b l) 1")[keep]

        x_t = t_flat * x_1 + (1.0 - (1.0 - self.sigma) * t_flat) * x_0
        target_velocity = x_1 - (1.0 - self.sigma) * x_0

        v_pred = self.depth_forward(x_t, t_flat, mid_tokens)
        return v_pred, target_velocity

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]
        v_pred, target_velocity = self.train_forward(input, target)
        loss = self.criterion(v_pred, target_velocity)
        return loss, v_pred

    def search(self, batch, n_results):
        assert self.training is False

        input = batch["input"]
        L = batch["target"].shape[-1] 

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal_tokens = self.temporal_forward(input)
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :].unsqueeze(1)

        B = input.shape[0]
        device = input.device

        last_item_ids = input[:, -1, :]
        x_0 = self.depth_emb(last_item_ids).sum(dim=1)

        t_start, t_end = 0.9, 1.0
        x_t = x_0.clone()
        dt = (t_end - t_start) / self.ode_steps

        for step in range(self.ode_steps):
            t_val = t_start + (step * dt)
            t = torch.full((B, 1), t_val, device=device, dtype=x_t.dtype)
            v_pred = self.depth_forward(x_t, t, mid_tokens)
            x_t = x_t + v_pred * dt

        target_item_embedding = x_t

        if self.catalog_tokens is not None:
            catalog_embeddings = self.depth_emb(self.catalog_tokens.to(device)).sum(dim=1)
            scores = torch.einsum("bd, id -> bi", target_item_embedding, catalog_embeddings)
            _, topk_item_indices = torch.topk(scores, n_results, dim=-1)
            indices = self.catalog_tokens[topk_item_indices]
        else:
            logits = torch.einsum("bd, vd -> bv", target_item_embedding, self.depth_emb.weight)
            _, topk_indices = torch.topk(logits, n_results, dim=-1)
            indices = topk_indices.unsqueeze(-1).repeat(1, 1, L)

        if self.filter_preds:
            arranged = torch.arange(B, device=indices.device).view(-1, 1)
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)

            filter_scores = torch.zeros((B, indices.size(1)), device=device)
            filter_scores[is_in_query] = -torch.inf

            _, topk_indices = torch.topk(filter_scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices
