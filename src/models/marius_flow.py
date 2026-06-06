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

        # The velocity network processes vectors directly; we only need a 1-layer projection or MLP blocks
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
        # x_t: (B, d_model), t: (B, 1), mid_tokens: (B, d_model)
        t_emb = self.get_fourier_time_embedding(t)
        
        # Inject context directly via vector addition to break bidirectional shortcuts
        combined_features = x_t + t_emb + mid_tokens
        
        dec_inputs = combined_features.unsqueeze(1) # (B, 1, d_model)
        dec_inputs = self.depth_dropout(dec_inputs)
        
        depth_preds = self.depth_tf(dec_inputs)
        return depth_preds.squeeze(1) # (B, d_model)

    def train_forward(self, input, target):
        B, L, _ = input.shape
        device = input.device
        
        # 1. Map history sequentially under a strict temporal causal mask
        temporal_tokens = self.temporal_forward(input) # (B, L, d_model)
        mid_tokens = self.mid_proj(temporal_tokens)    # (B, L, d_model)

        # 2. Randomly select an interaction index per batch element to form pairs
        # Needs at least 1 valid history token to flow from, and 1 next item target
        valid_indices = []
        chosen_positions = []
        
        for b in range(B):
            # Locate active unpadded entries
            valid_len = (input[b, :, 0] != SpecialTokens.PAD.value).sum().item()
            # Also ensure target isn't an ignored flag
            valid_targets = torch.where(target[b, :, 0] != -100)[0]
            
            # Intersection of existing steps and real targets
            possible_idx = [idx for idx in range(valid_len) if idx in valid_targets]
            
            if len(possible_idx) > 0:
                pos = float(torch.randint(0, len(possible_idx), (1,)).item())
                chosen_positions.append(possible_idx[int(pos)])
                valid_indices.append(b)
                
        if len(valid_indices) == 0:
            # Fallback if batch contains only padding junk
            return torch.zeros((1, self.depth_cfg.d_model), device=device), torch.zeros((1, self.depth_cfg.d_model), device=device)

        # Squeeze batch data down to only valid parsed entries
        valid_b = torch.tensor(valid_indices, device=device)
        valid_l = torch.tensor(chosen_positions, device=device)

        # Extract context representing history up to step l
        mid_tokens_sampled = mid_tokens[valid_b, valid_l] # (N, d_model)

        # Pull item tokens
        input_sampled = input[valid_b, valid_l]   # (N, K) -> Item at t=0
        target_sampled = target[valid_b, valid_l] # (N, K) -> Next item at t=1

        x_0 = self.depth_emb(input_sampled).sum(dim=1)
        x_1 = self.depth_emb(target_sampled).sum(dim=1)

        # 3. Continuous Integration Mechanics
        t_flat = torch.rand((x_0.shape[0], 1), device=device)

        # Flow straight path setups
        x_t = t_flat * x_1 + (1.0 - t_flat) * x_0
        target_velocity = x_1 - x_0

        v_pred = self.depth_forward(x_t, t_flat, mid_tokens_sampled)
        return v_pred, target_velocity

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]
        v_pred, target_velocity = self.train_forward(input, target)
        loss = self.criterion(v_pred, target_velocity)
        return loss, v_pred

    def search(self, batch, n_results):
        assert self.training is False

        input = batch["input"]
        L_target = batch["target"].shape[-1] 

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        # Extract sequence context based completely on past item paths
        temporal_tokens = self.temporal_forward(input)
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :] # Vector context (B, d_model)

        B = input.shape[0]
        device = input.device

        # Inference starts precisely at the last interacted item (t=0.0)
        last_item_ids = input[:, -1, :]
        x_0 = self.depth_emb(last_item_ids).sum(dim=1)

        t_start, t_end = 0.0, 1.0
        x_t = x_0.clone()
        dt = (t_end - t_start) / self.ode_steps

        # Integrator loop traveling down the velocity fields
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
            indices = topk_indices.unsqueeze(-1).repeat(1, 1, L_target)

        if self.filter_preds:
            arranged = torch.arange(B, device=indices.device).view(-1, 1)
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)

            filter_scores = torch.zeros((B, indices.size(1)), device=device)
            filter_scores[is_in_query] = -torch.inf

            _, topk_indices = torch.topk(filter_scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices