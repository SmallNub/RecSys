from dataclasses import dataclass
import math
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

        # Flow Matching variables
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

        # Embeddings
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

        # Timeline Positional Encoding
        self.temp_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.temporal_cfg.seq_len, self.temporal_cfg.d_model))
        )
        
        # Fourier Time Embedding Network
        self.register_buffer(
            "fourier_frequencies", 
            torch.randn(self.fourier_dim // 2) * 2 * math.pi
        )
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.fourier_dim, self.depth_cfg.d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(self.depth_cfg.d_model, self.depth_cfg.d_model),
        )

        # Embedding dropout
        self.temp_dropout = torch.nn.Dropout(self.temporal_cfg.emb_dropout)
        self.depth_dropout = torch.nn.Dropout(self.depth_cfg.emb_dropout)

        # Transformer Blocks
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

        # Projection
        self.mid_proj = torch.nn.Linear(
            self.temporal_cfg.d_model, self.depth_cfg.d_model
        )

        self.criterion = torch.nn.MSELoss()

    def get_fourier_time_embedding(self, t):
        """Maps timesteps t [N, 1] to [N, d_model] using Fourier Features."""
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
        """Estimates velocity fields v_t(x_t, t) given timeline context."""
        t_emb = self.get_fourier_time_embedding(t).unsqueeze(1) # N x 1 x D
        x_t_emb = x_t.unsqueeze(1)                              # N x 1 x D
        x_t_emb = self.depth_dropout(x_t_emb)

        dec_inputs = torch.cat([mid_tokens, t_emb, x_t_emb], dim=1) # N x 3 x D
        depth_preds = self.depth_tf(dec_inputs)
        
        return depth_preds[:, -1, :]

    def train_forward(self, input, target):
        B, L, _ = input.shape
        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)     # B x L x d
        
        # Flatten sequence length context
        mid_tokens_flat = rearrange(mid_tokens, "b l d -> (b l) 1 d") 

        # Map target items to continuous embedding targets (x_1)
        target_flat = rearrange(target, "b l k -> (b l) k")  # BL x K
        keep = target_flat[:, 0] != -100

        mid_tokens_flat = mid_tokens_flat[keep]
        target_flat = target_flat[keep]
        x_1 = self.depth_emb(target_flat).sum(dim=1)  # N x D (Target items to hit)

        # --- Dynamic Timeline Mapping Mechanics ---
        # Map input history tokens to continuous starting point representations (x_0)
        input_flat = rearrange(input, "b l k -> (b l) k")[keep]
        x_0 = self.depth_emb(input_flat).sum(dim=1)   # N x D (Historical base items)

        # Calculate time positions based on historical slot indices
        # Creating relative scale where target prediction always hits 1.0
        time_steps = torch.linspace(0.0, 0.9, steps=L, device=input.device)
        t_matrix = time_steps.unsqueeze(0).repeat(B, 1) # B x L
        t_flat = rearrange(t_matrix, "b l -> (b l) 1")[keep] # N x 1

        # Construct Flow Trajectories moving from historical states (x_0) to next-items (x_1)
        x_t = t_flat * x_1 + (1.0 - (1.0 - self.sigma) * t_flat) * x_0
        target_velocity = x_1 - (1.0 - self.sigma) * x_0

        v_pred = self.depth_forward(x_t, t_flat, mid_tokens_flat)

        return v_pred, target_velocity

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]
        v_pred, target_velocity = self.train_forward(input, target)
        loss = self.criterion(v_pred, target_velocity)
        return loss, v_pred

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode (dropout)."

        input = batch["input"]
        L = batch["target"].shape[-1] 
        seq_len = input.shape[1]

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        # Extract timeline context for the last sequence interaction item
        temporal_tokens = self.temporal_forward(input)                  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :].unsqueeze(1)  # B x 1 x D

        B = input.shape[0]
        device = input.device

        # 1. Initialize trajectory using the last known physical item embedding as x_0
        last_item_ids = input[:, -1, :] # B x K
        x_0 = self.depth_emb(last_item_ids).sum(dim=1) # B x D

        # 2. Set starting time value based on training configuration sequence thresholds
        # The history leaves off at t = 0.9, and we want to flow forward to t = 1.0
        t_start = 0.9
        t_end = 1.0
        
        x_t = x_0.clone()
        dt = (t_end - t_start) / self.ode_steps

        # 3. Integrate vector fields along the remaining path interval [0.9 -> 1.0]
        for step in range(self.ode_steps):
            t_val = t_start + (step * dt)
            t = torch.full((B, 1), t_val, device=device, dtype=x_t.dtype)
            
            v_pred = self.depth_forward(x_t, t, mid_tokens)
            x_t = x_t + v_pred * dt

        # 4. Map resulting position at t=1.0 back into item code indices
        target_item_embedding = x_t 
        logits = torch.einsum("bd, vd -> bv", target_item_embedding, self.depth_emb.weight)
        
        _, topk_indices = torch.topk(logits, n_results, dim=-1)
        indices = topk_indices.unsqueeze(-1).repeat(1, 1, L)

        if self.filter_preds:
            arranged = torch.arange(B, device=indices.device).view(-1, 1)
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)

            scores = torch.zeros((B, indices.size(1)), device=device)
            scores[is_in_query] = -torch.inf

            _, topk_indices = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices
