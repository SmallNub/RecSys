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
    fourier_dim: int = 32  # Added for Fourier Time Embedding resolution


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

        # Timeline Positional Encoding (Still required for history context order)
        self.temp_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.temporal_cfg.seq_len, self.temporal_cfg.d_model))
        )
        
        # Fourier Time Embedding Network (Replaces old positional codebooks for flow)
        # We project 1 scalar into a static frequency band, then project to d_model
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

        # Loss
        self.criterion = torch.nn.MSELoss()

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n

        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]

        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def get_fourier_time_embedding(self, t):
        """
        Maps timesteps t [N, 1] using random Fourier projecting components
        Output Shape: [N, d_model]
        """
        # Linear projection scaling multiplication
        # [N, 1] * [F/2] -> [N, F/2]
        phases = t @ self.fourier_frequencies.unsqueeze(0)
        
        # Calculate sinusoidal pairings
        fourier_features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1) # [N, F]
        
        # Pass through the projection network MLP
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
        """
        Estimates velocity fields v_t(x_t, t)
        """
        # Generate Fourier time projection tokens
        t_emb = self.get_fourier_time_embedding(t).unsqueeze(1) # N x 1 x D
        x_t_emb = x_t.unsqueeze(1)                              # N x 1 x D

        x_t_emb = self.depth_dropout(x_t_emb)

        # Structured sequence input token block
        dec_inputs = torch.cat([mid_tokens, t_emb, x_t_emb], dim=1) # N x 3 x D

        depth_preds = self.depth_tf(dec_inputs)
        
        # Vector Field Velocity Output Slice position
        v_pred = depth_preds[:, -1, :]
        return v_pred

    def train_forward(self, input, target):
        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)     # B x L x d
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")  # BL x 1 x D

        target = rearrange(target, "b l k -> (b l) k")  # BL x K
        keep = target[:, 0] != -100

        # Do not forward padding tokens.
        mid_tokens = mid_tokens[keep]
        target = target[keep]

        # Extract unified dense continuous representation
        x_1 = self.depth_emb(target).sum(dim=1)  # N x D

        # Sample trajectories
        N = x_1.shape[0]
        t = torch.rand((N, 1), device=x_1.device, dtype=x_1.dtype)
        x_0 = torch.randn_like(x_1)

        # Construct Optimal Transport mappings
        x_t = t * x_1 + (1.0 - (1.0 - self.sigma) * t) * x_0
        target_velocity = x_1 - (1.0 - self.sigma) * x_0

        v_pred = self.depth_forward(x_t, t, mid_tokens)

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

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :]  # B, D
        mid_tokens = mid_tokens.unsqueeze(1) # B x 1 x D

        B = input.shape[0]
        device = input.device

        # 1. Prior distribution coordinates sampling
        x_t = torch.randn((B, self.depth_cfg.d_model), device=device)

        # 2. ODE Solver Integration loop using Fourier time steps embeddings
        dt = 1.0 / self.ode_steps
        for step in range(self.ode_steps):
            t_val = step * dt
            t = torch.full((B, 1), t_val, device=device, dtype=x_t.dtype)
            
            v_pred = self.depth_forward(x_t, t, mid_tokens)
            x_t = x_t + v_pred * dt

        # 3. Discrete Similarity Vocabulary Lookup
        target_item_embedding = x_t # B x D
        logits = torch.einsum("bd, vd -> bv", target_item_embedding, self.depth_emb.weight)
        
        _, topk_indices = torch.topk(logits, n_results, dim=-1)
        indices = topk_indices.unsqueeze(-1).repeat(1, 1, L)

        if self.filter_preds:
            arranged = torch.arange(B, device=indices.device).view(-1, 1)
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)  # Shape B x b

            scores = torch.zeros((B, indices.size(1)), device=device)
            scores[is_in_query] = -torch.inf

            _, topk_indices = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices
