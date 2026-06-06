from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

from src.models import SpecialTokens


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

        # No external item configurations or parquets are required
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

    def train_forward(self, input, target, timestamps=None):
        """
        timestamps: Tensor of shape (B, L) containing the randomly assigned 
                    but sequentially ordered timestamps (e.g., 0.12, 0.25, 0.43...)
        """
        B, L, _ = input.shape
        device = input.device

        if timestamps is None:
            # Fallback: create random sorted timestamps per batch if not provided by dataloader
            timestamps = torch.sort(torch.rand((B, L), device=device), dim=-1).values

        # 1. Generate a Time-Based Causal Mask for the temporal encoder
        # Position i can only attend to position j if timestamp[j] <= timestamp[i]
        time_mask = timestamps.unsqueeze(1) < timestamps.unsqueeze(2) # (B, L, L)
        
        # Combined mask with your padding mask and structural causal mask
        # Note: PyTorch Transformer Encoder expects True where attention is forbidden
        pad_mask = (input[:, :, 0] == SpecialTokens.PAD.value).unsqueeze(1).repeat(1, L, 1)
        full_temporal_mask = time_mask | pad_mask

        # 2. Pass through temporal encoder with our strict timestamp mask
        # We must loop or use a custom attention mechanism if masks vary per batch element,
        # or process them using PyTorch's native tuple/3D mask support.
        temporal_tokens = []
        for b in range(B):
            # Process per batch element because each has a unique random timeline
            input_embs = self.temp_emb(input[b]).sum(dim=-1) + self.temp_pos_emb[0, :L, :]
            out_b = self.temp_tf(
                input_embs.unsqueeze(0),
                mask=full_temporal_mask[b]
            )
            temporal_tokens.append(out_b)
        temporal_tokens = torch.cat(temporal_tokens, dim=0)

        mid_tokens = self.mid_proj(temporal_tokens)
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")

        # 3. Flow Matching Setup
        target_flat = rearrange(target, "b l k -> (b l) k")
        keep = target_flat[:, 0] != -100

        mid_tokens = mid_tokens[keep]
        target_flat = target_flat[keep]
        
        x_1 = self.depth_emb(target_flat).sum(dim=1)
        
        input_flat = rearrange(input, "b l k -> (b l) k")[keep]
        x_0 = self.depth_emb(input_flat).sum(dim=1)

        # Crucial Fix: Flow time (t) must be completely decoupled from history timeline
        # to avoid algebraic reverse-engineering tricks!
        t_flat = torch.rand((mid_tokens.shape[0], 1), device=device)

        # Standard Rectified Flow Equation
        x_t = t_flat * x_1 + (1.0 - t_flat) * x_0
        target_velocity = x_1 - x_0

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
            # Reverts directly back to embedding lookup structures if no token maps are explicitly passed
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
