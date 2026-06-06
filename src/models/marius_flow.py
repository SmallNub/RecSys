from dataclasses import dataclass
import glob
import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange

# Importing your existing Cosette model architecture directly
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


# ==========================================
# AUTOMATED LATEST CHECKPOINT LOADER FUNCTION
# ==========================================

def load_latest_catalog_tokens(device="cuda"):
    """
    Finds the newest timestamp checkpoint inside outputs/checkpoints/cosette/{timestamp}/last_model.pth,
    loads the pre-trained weights, and extracts the full token catalog map.
    """
    # Strictly lock the path pattern to the cosette timestamp directory structure
    base_path = os.path.join(os.getcwd(), "outputs", "checkpoints", "cosette", "*", "last_model.pth")
    checkpoint_files = glob.glob(base_path)

    if not checkpoint_files:
        print("⚠️ Warning: No Cosette checkpoints found in outputs/checkpoints/cosette/! Reverting to vocabulary flat lookups.")
        return None

    # Sorting strings naturally sorts 'YYYYMMDD_HHMMSS' chronologically
    checkpoint_files.sort()
    latest_checkpoint_path = checkpoint_files[-1]
    print(f"🚀 Found latest Cosette checkpoint path target: {latest_checkpoint_path}")

    # Safely load weights map
    checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if checkpoint is not None else None
    
    # CRITICAL GUARD: Stop immediately if the state_dict couldn't be loaded
    if state_dict is None:
        print("⚠️ Warning: Checkpoint state dict is empty. Reverting to vocabulary flat lookups.")
        return None

    # Recover structural dimension features from saved embedding parameters
    embeddings_weight = state_dict.get("embeddings", state_dict.get("module.embeddings", None))
    if embeddings_weight is None:
        print("⚠️ Warning: Checkpoint missing base embedding layer mappings. Reverting.")
        return None

    total_items, in_dim = embeddings_weight.shape

    # Recover structural dimension features from state layers directly
    encoder_weights = [v for k, v in state_dict.items() if "encoder.mlp_layers" in k and "weight" in k]
    layers_dims = [w.shape[0] for w in encoder_weights]
    
    # Read centroid counts dynamically from parameters
    num_quantizers = len([k for k in state_dict.keys() if "rq.vq_layers" in k and "embedding.weight" in k])
    n_centroids_list = [256] * num_quantizers

    # Initialize imported model structure instance
    cosette_model = COSETTE(
        embs_block=embeddings_weight,
        in_dim=in_dim,
        layers=layers_dims,
        n_centroids_list=n_centroids_list,
        dropout_prob=0.0,
        tau=1.0, bias=0.0, freeze_tau=True, freeze_bias=True,
        loss_weights={"reconstruction": 0.0, "quantization": 0.0, "contrastive": 0.0}
    ).to(device)

    # Clean key prefixes if wrapped inside extra distributed DataParallel modules
    sanitized_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        sanitized_state_dict[name] = v

    cosette_model.load_state_dict(sanitized_state_dict, strict=False)
    cosette_model.eval()

    # Process all available catalog vectors directly into discrete multi-scale tokens
    with torch.no_grad():
        catalog_tokens, _ = cosette_model.get_indices(use_sk=False)

    print(f"✅ Successfully loaded and processed {catalog_tokens.shape[0]} unique items with codebook depth {catalog_tokens.shape[1]}.")
    return catalog_tokens


# ==========================================
# MARIUS FLOW SEQUENCE MODEL IMPLEMENTATION
# ==========================================

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

        # Runtime Dynamic Catalog Registration Interface
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokens_map = load_latest_catalog_tokens(device=device)
        if tokens_map is not None:
            self.register_buffer("catalog_tokens", tokens_map)
        else:
            self.catalog_tokens = None

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

        # Baseline trajectory modification: map continuous start point to past history step
        input_flat = rearrange(input, "b l k -> (b l) k")[keep]
        x_0 = self.depth_emb(input_flat).sum(dim=1)

        # Dynamic timescale creation matching total timeline boundaries
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
        assert self.training is False, "Not in evaluation mode (dropout)."

        input = batch["input"]
        L = batch["target"].shape[-1] 

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal_tokens = self.temporal_forward(input)
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :].unsqueeze(1)

        B = input.shape[0]
        device = input.device

        # Initialize continuous trajectory position using the user's last historic interaction sequence item
        last_item_ids = input[:, -1, :]
        x_0 = self.depth_emb(last_item_ids).sum(dim=1)

        # Execute ODE tracking iterations inside the prediction spectrum window [0.9 -> 1.0]
        t_start, t_end = 0.9, 1.0
        x_t = x_0.clone()
        dt = (t_end - t_start) / self.ode_steps

        for step in range(self.ode_steps):
            t_val = t_start + (step * dt)
            t = torch.full((B, 1), t_val, device=device, dtype=x_t.dtype)
            v_pred = self.depth_forward(x_t, t, mid_tokens)
            x_t = x_t + v_pred * dt

        target_item_embedding = x_t

        # Fully integrated real item candidate matching lookup step
        if self.catalog_tokens is not None:
            # Reconstruct dense coordinate space map of full hierarchical catalog items
            catalog_embeddings = self.depth_emb(self.catalog_tokens.to(device)).sum(dim=1)
            scores = torch.einsum("bd, id -> bi", target_item_embedding, catalog_embeddings)
            _, topk_item_indices = torch.topk(scores, n_results, dim=-1)
            indices = self.catalog_tokens[topk_item_indices]
        else:
            # Fallback uniform index projection if directory lookup was skipped
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