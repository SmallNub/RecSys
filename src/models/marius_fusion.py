from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange

from src.models import SpecialTokens
from src.models.cosette_wrapper import CosetteWrapper


@dataclass
class TransformerConfig:
    n_layers: int
    d_head: int
    d_model: int
    vocab_size: int
    dropout: float
    emb_dropout: float
    seq_len: int


class MARIUS(torch.nn.Module):
    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        cosette: CosetteWrapper,
        filter_preds=False,
        warmup_steps=15000,   # 15k steps: Pure discrete embedding refinement
        fade_steps=30000,    # 30k steps: Ultra-smooth continuous integration
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds
        self.cosette = cosette
        
        self.warmup_steps = warmup_steps
        self.fade_steps = fade_steps

        assert (
            self.depth_cfg.vocab_size == self.temporal_cfg.vocab_size
        ), "Vocab size mismatch"

        if self.temporal_cfg.emb_dropout is None:
            self.temporal_cfg.emb_dropout = self.temporal_cfg.dropout
        if self.depth_cfg.emb_dropout is None:
            self.depth_cfg.emb_dropout = self.depth_cfg.dropout

        # Persistent training step buffer
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        # Discrete Backbone Embeddings
        self.temp_emb = torch.nn.Embedding(
            self.temporal_cfg.vocab_size,
            self.temporal_cfg.d_model,
            padding_idx=SpecialTokens.PAD.value,
        )
        self.depth_emb = torch.nn.Embedding(
            self.depth_cfg.vocab_size,
            self.depth_cfg.d_model,
            padding_idx=SpecialTokens.PAD.value,
        )

        # Continuous Projections
        self.temp_proj = torch.nn.Linear(
            self.cosette.model.in_dim, self.temporal_cfg.d_model
        )
        self.depth_proj = torch.nn.Linear(
            self.cosette.model.centroids_dim, self.depth_cfg.d_model
        )

        # Dual-Stream Layer Normalization
        self.temp_ln_discrete = torch.nn.LayerNorm(self.temporal_cfg.d_model)
        self.temp_ln_continuous = torch.nn.LayerNorm(self.temporal_cfg.d_model)
        
        self.depth_ln_discrete = torch.nn.LayerNorm(self.depth_cfg.d_model)
        self.depth_ln_continuous = torch.nn.LayerNorm(self.depth_cfg.d_model)

        # FiLM Generators
        self.temp_film = torch.nn.Linear(self.temporal_cfg.d_model, self.temporal_cfg.d_model * 2)
        self.depth_film = torch.nn.Linear(self.depth_cfg.d_model, self.depth_cfg.d_model * 2)

        # Positional Encoding
        self.temp_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.temporal_cfg.seq_len, self.temporal_cfg.d_model))
        )
        self.depth_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.depth_cfg.seq_len, self.depth_cfg.d_model))
        )

        # Embedding dropout
        self.temp_dropout = torch.nn.Dropout(self.temporal_cfg.emb_dropout)
        self.depth_dropout = torch.nn.Dropout(self.depth_cfg.emb_dropout)

        # Output Classification Head
        self.output_head = torch.nn.Linear(
            self.depth_cfg.d_model, self.depth_cfg.vocab_size
        )

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
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def get_param_groups(self):
        def _select_no_decay(n):
            return "bias" in n or "norm" in n

        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input, alpha=1.0):
        B, L, K = input.shape

        discrete_embs = self.temp_emb(input[:, :, 0])
        decoded_features = self.cosette.decode(input)
        continuous_embs = self.temp_proj(decoded_features)
        
        discrete_embs = self.temp_ln_discrete(discrete_embs)
        continuous_embs = self.temp_ln_continuous(continuous_embs)
        
        film_params = self.temp_film(continuous_embs)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Smooth scaling via curated internal alpha
        gamma = gamma * alpha
        beta = beta * alpha
            
        input_embs = discrete_embs * (1.0 + gamma) + beta

        input_embs += self.temp_pos_emb[:, :L, :]
        input_embs = self.temp_dropout(input_embs)

        out = self.temp_tf(
            input_embs,
            mask=self.causal_mask[:L, :L],
            src_key_padding_mask=input[:, :, 0] == SpecialTokens.PAD.value,
        )
        return out

    def depth_forward(self, in_embs):
        K = in_embs.shape[1]
        in_embs = in_embs + self.depth_pos_emb[:, :K, :]
        in_embs = self.depth_dropout(in_embs)
        depth_preds = self.depth_tf(in_embs, mask=self.causal_mask[:K, :K])
        logits = self.output_head(depth_preds)
        return logits

    def train_forward(self, input, target, alpha=1.0):
        temporal_tokens = self.temporal_forward(input, alpha=alpha)
        mid_tokens = self.mid_proj(temporal_tokens)
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")

        target = rearrange(target, "b l k -> (b l) k")
        keep = target[:, 0] != -100

        mid_tokens = mid_tokens[keep]
        target = target[keep]

        N, K = target.shape
        dec_embs_list = []

        for i in range(K - 1):
            quantizer = self.cosette.model.rq.vq_layers[i]
            layer_indices = target[:, i]

            num_embeddings = getattr(quantizer, "n_centroids", 32000)
            valid_mask = (layer_indices >= 0) & (layer_indices < num_embeddings)
            safe_indices = torch.where(valid_mask, layer_indices, torch.zeros_like(layer_indices))

            discrete_res = self.depth_emb(safe_indices)
            x_res = quantizer.get_codebook_entry(safe_indices, shape=None)
            continuous_res = self.depth_proj(x_res)

            discrete_res = self.depth_ln_discrete(discrete_res)
            continuous_res = self.depth_ln_continuous(continuous_res)

            film_params = self.depth_film(continuous_res)
            gamma, beta = film_params.chunk(2, dim=-1)
            
            gamma = gamma * alpha
            beta = beta * alpha

            proj_res = discrete_res * (1.0 + gamma) + beta

            v_mask = valid_mask.unsqueeze(-1).to(discrete_res.dtype)
            proj_res = proj_res * v_mask
            dec_embs_list.append(proj_res)

        if len(dec_embs_list) > 0:
            dec_embs = torch.stack(dec_embs_list, dim=1)
            dec_embs = torch.cat([mid_tokens, dec_embs], dim=1)
        else:
            dec_embs = mid_tokens

        depth_logits = self.depth_forward(dec_embs)
        return depth_logits, target

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]
        
        if self.training:
            current_step = self.global_step.item()
            if current_step < self.warmup_steps:
                alpha = 0.0
            else:
                alpha = min(1.0, (current_step - self.warmup_steps) / self.fade_steps)
            
            self.global_step += 1
        else:
            alpha = 1.0

        logits, m_target = self.train_forward(input, target, alpha=alpha)
        logits = rearrange(logits, "B k v -> B v k")
        loss = self.criterion(logits, m_target)
        return loss, logits

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode."

        input = batch["input"]
        L = batch["target"].shape[-1]

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal_tokens = self.temporal_forward(input, alpha=1.0)
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :]

        B, b, D = input.shape[0], n_results, self.depth_cfg.d_model

        sequences = mid_tokens[:, None, :]
        depth_logits = self.depth_forward(sequences)
        log_probs = F.log_softmax(depth_logits[:, -1, :], dim=-1)

        topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

        indices = topk_indices.unsqueeze(2)
        scores = topk_log_probs
        sequences = sequences.unsqueeze(1).repeat(1, b, 1, 1)

        quantizer = self.cosette.model.rq.vq_layers[0]
        num_embeddings = getattr(quantizer, "n_centroids", 32000)
        valid_mask = (topk_indices >= 0) & (topk_indices < num_embeddings)
        safe_indices = torch.where(valid_mask, topk_indices, torch.zeros_like(topk_indices))

        discrete_new = self.depth_emb(safe_indices)
        raw_new_tokens = quantizer.get_codebook_entry(safe_indices, shape=None)
        continuous_new = self.depth_proj(raw_new_tokens)

        discrete_new = self.depth_ln_discrete(discrete_new)
        continuous_new = self.depth_ln_continuous(continuous_new)

        film_new = self.depth_film(continuous_new)
        gamma_new, beta_new = film_new.chunk(2, dim=-1)
        fused_new = discrete_new * (1.0 + gamma_new) + beta_new

        v_mask = valid_mask.unsqueeze(-1).to(discrete_new.dtype)
        new_tokens = (fused_new * v_mask).unsqueeze(2)
        sequences = torch.concat([sequences, new_tokens], dim=2)

        arranged = torch.arange(B, device=sequences.device).view(-1, 1)

        for i in range(2, L + 1):
            last_logits = self.depth_forward(sequences.view(B * b, i, D))[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

            topk_log_probs = topk_log_probs.view(B, b, b)
            topk_indices = topk_indices.view(B, b, b)

            expanded_indices = indices.unsqueeze(2).repeat(1, 1, b, 1)
            expanded_indices = torch.cat(
                [expanded_indices, topk_indices.unsqueeze(-1)], dim=3
            )

            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)

            quantizer = self.cosette.model.rq.vq_layers[i - 1]
            num_embeddings = getattr(quantizer, "n_centroids", 32000)
            valid_mask = (topk_indices >= 0) & (topk_indices < num_embeddings)
            safe_indices = torch.where(valid_mask, topk_indices, torch.zeros_like(topk_indices))

            flat_topk = safe_indices.view(-1)
            flat_valid = valid_mask.view(-1, 1)

            flat_discrete = self.depth_emb(flat_topk)
            flat_decoded = quantizer.get_codebook_entry(flat_topk, shape=None)
            flat_projected = self.depth_proj(flat_decoded)

            flat_discrete = self.depth_ln_discrete(flat_discrete)
            flat_projected = self.depth_ln_continuous(flat_projected)

            flat_film = self.depth_film(flat_projected)
            flat_gamma, flat_beta = flat_film.chunk(2, dim=-1)
            flat_fused = flat_discrete * (1.0 + flat_gamma) + flat_beta

            next_tokens = (flat_fused * flat_valid.to(flat_discrete.dtype)).view(B, b, b, 1, D)
            expanded_sequences = torch.cat([expanded_sequences, next_tokens], dim=3)

            expanded_scores = scores.unsqueeze(2) + topk_log_probs

            expanded_scores = expanded_scores.view(B, -1)
            expanded_sequences = expanded_sequences.view(
                B, -1, expanded_sequences.size(-2), D
            )
            expanded_indices = expanded_indices.view(B, -1, expanded_indices.size(-1))

            topk_scores, topk_indices_choice = torch.topk(expanded_scores, b, dim=-1)

            sequences = expanded_sequences[arranged, topk_indices_choice]
            indices = expanded_indices[arranged, topk_indices_choice]
            scores = topk_scores

        if self.filter_preds:
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)
            scores[is_in_query] = -torch.inf
            topk_scores, topk_indices_choice = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices_choice]

        return indices
