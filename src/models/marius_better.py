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


class MARIUS(torch.nn.Module):
    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        tie_embeddings=False,
        filter_preds=False,
        top_k=10,
        truncation_weight=0.25,
        listwise_weight=0.40,
        repr_weight=0.35,
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds
        
        # Loss Optimization Weights
        self.top_k = top_k
        self.truncation_weight = truncation_weight
        self.listwise_weight = listwise_weight
        self.repr_weight = repr_weight

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

        # Stable Native Transformers (Fast Path Preserved)
        self.temp_tf = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.temporal_cfg.d_model,
                nhead=self.temporal_cfg.d_model // self.temporal_cfg.d_head,
                dim_feedforward=self.temporal_cfg.d_model * 4,
                dropout=self.temporal_cfg.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
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
                activation="gelu",
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

        # Baseline Target Loss
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # Learned Scaling Parameters for Loss Calibration
        self.listwise_temp = torch.nn.Parameter(torch.tensor(0.0))
        self.repr_temp = torch.nn.Parameter(torch.tensor(0.0))

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n or "temp" in n

        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]

        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        B, L, K = input.shape

        input_embs = self.temp_emb(input).sum(dim=-2)
        input_embs += self.temp_pos_emb[:, : input.shape[1], :]

        input_embs = self.temp_dropout(input_embs)

        out = self.temp_tf(
            input_embs,
            mask=self.causal_mask[:L, :L].to(input.device),
            src_key_padding_mask=input[:, :, 0] == SpecialTokens.PAD.value,
        )

        return out

    def depth_forward(self, in_embs):
        K = in_embs.shape[1]

        in_embs = in_embs + self.depth_pos_emb[:, :K, :]
        in_embs = self.depth_dropout(in_embs)

        depth_preds = self.depth_tf(in_embs, mask=self.causal_mask[:K, :K].to(in_embs.device))

        logits = torch.einsum("bkd, vd -> bkv", depth_preds, self.depth_emb.weight)

        return logits, depth_preds

    def train_forward(self, input, target):
        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)  # B x L x d
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")  # BL x 1 x D

        target = rearrange(target, "b l k -> (b l) k")  # BL x K
        keep = target[:, 0] != -100

        # Do not forward padding tokens.
        mid_tokens = mid_tokens[keep]
        target = target[keep]

        # Drop L4, we never predict L5 - BL x K x D
        dec_embs = self.depth_emb(target[:, :-1])

        # mid_tokens : BL x 1 x D - dec_embs : BL x K x D
        dec_embs = torch.cat([mid_tokens, dec_embs], dim=1)

        depth_logits, hidden_states = self.depth_forward(dec_embs)  # BL x K x V & BL x K x D
        return depth_logits, hidden_states, target

    # =========================================================================
    # ADVANCED STRUCTURAL LOSS CORE FUNCTIONS
    # =========================================================================

    def compute_softmax_loss_at_k(self, flat_logits, flat_targets, k=10):
        pos_scores = flat_logits.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
        topk_scores, _ = torch.topk(flat_logits, k=k, dim=-1)
        
        max_scores = torch.max(topk_scores, dim=-1)[0]
        shifted_topk = topk_scores - max_scores.unsqueeze(-1)
        shifted_pos = pos_scores - max_scores

        denom = torch.log(torch.exp(shifted_pos) + torch.exp(shifted_topk).sum(dim=-1) + 1e-8) + max_scores
        return torch.mean(denom - pos_scores)

    def compute_plackett_luce_loss(self, flat_logits, flat_targets):
        pos_scores = flat_logits.gather(dim=-1, index=flat_targets.unsqueeze(-1))
        topk_neg, _ = torch.topk(flat_logits, k=32, dim=-1)
        
        slate = torch.cat([pos_scores, topk_neg], dim=-1)
        t = torch.exp(self.listwise_temp).clamp(min=1e-3, max=5.0)
        scaled_slate = slate / t

        log_prob = 0.0
        for i in range(min(5, scaled_slate.shape[-1])):
            remainder = scaled_slate[:, i:]
            max_val = torch.max(remainder, dim=-1, keepdim=True)[0]
            norm_rem = remainder - max_val
            denom = torch.log(torch.exp(norm_rem).sum(dim=-1) + 1e-8) + max_val.squeeze(-1)
            log_prob += scaled_slate[:, i] - denom

        return torch.mean(-log_prob)

    def compute_infonce_loss(self, flat_hidden, flat_targets):
        pos_emb = self.depth_emb(flat_targets)
        
        flat_h_norm = F.normalize(flat_hidden, p=2, dim=-1)
        pos_emb_norm = F.normalize(pos_emb, p=2, dim=-1)

        similarity_matrix = torch.matmul(flat_h_norm, pos_emb_norm.T)
        
        t = torch.exp(self.repr_temp).clamp(min=1e-2, max=1.0)
        logits = similarity_matrix / t
        
        labels = torch.arange(logits.shape[0], device=flat_hidden.device)
        return F.cross_entropy(logits, labels)

    # =========================================================================

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]

        logits, hidden_states, m_target = self.train_forward(input, target)

        # 1. Native Cross Entropy Formulation (Channel Transposed)
        logits_ce = rearrange(logits, "B k v -> B v k")
        base_ce_loss = self.criterion(logits_ce, m_target)

        # 2. Reshape and Mask Elements for Listwise Structural Calculation
        flat_logits = logits.view(-1, logits.size(-1))
        flat_hidden = hidden_states.view(-1, hidden_states.size(-1))
        flat_targets = m_target.view(-1)
        
        mask = (flat_targets != -100) & (flat_targets != SpecialTokens.PAD.value)
        
        if mask.any():
            masked_logits = flat_logits[mask]
            masked_hidden = flat_hidden[mask]
            masked_targets = flat_targets[mask]

            # Compute Structural Alternatives
            trunc_loss = self.compute_softmax_loss_at_k(masked_logits, masked_targets, k=self.top_k)
            listwise_loss = self.compute_plackett_luce_loss(masked_logits, masked_targets)
            repr_loss = self.compute_infonce_loss(masked_hidden, masked_targets)

            # Unified Loss Composition Array
            total_loss = base_ce_loss + (
                self.truncation_weight * trunc_loss +
                self.listwise_weight * listwise_loss +
                self.repr_weight * repr_loss
            )
        else:
            total_loss = base_ce_loss

        return total_loss, logits

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode (dropout)."

        input = batch["input"]
        L = batch["target"].shape[-1]

        if self.filter_preds:  # Generate more, so that we can filter out seen items.
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        # Validation only on the last prediction of the sequence
        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :]  # B, D

        B, b, D = input.shape[0], n_results, self.depth_cfg.d_model

        # Initialize the beam search ==============================================
        ## Get logits
        sequences = mid_tokens[:, None, :]  # B x 1 x D
        depth_logits, _ = self.depth_forward(sequences)  # B x 1 x D
        log_probs = F.log_softmax(depth_logits[:, -1, :], dim=-1)  # B x V

        ## Get topk indices
        topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)  # B x b

        indices = topk_indices.unsqueeze(2)  # Shape: (B, b, 1)
        scores = topk_log_probs  # Shape: B x b

        ## Expand sequences
        sequences = sequences.unsqueeze(1).repeat(1, b, 1, 1)  # Shape: (B, b, 1, D)

        new_tokens = self.depth_emb(topk_indices).unsqueeze(2)  # Shape : (B, b, 1, D)
        sequences = torch.concat([sequences, new_tokens], dim=2)

        arranged = torch.arange(B, device=sequences.device).view(-1, 1)

        # Start the beam search ===================================================
        for i in range(2, L + 1):
            # Forward the current sequence to get logits of the next token
            last_logits, _ = self.depth_forward(sequences.view(B * b, i, D))
            last_logits = last_logits[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)  # B * b x V

            topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

            # Shape: (B * b, b) -> (B, b, b)
            topk_log_probs = topk_log_probs.view(B, b, b)
            topk_indices = topk_indices.view(B, b, b)

            # Expand indices - Update with top values
            expanded_indices = indices.unsqueeze(2).repeat(1, 1, b, 1)
            expanded_indices = torch.cat(
                [expanded_indices, topk_indices.unsqueeze(-1)], dim=3
            )

            # Expand sequences
            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)
            # Choose the new tokens - Shape: (B, b, b, 1, D)
            next_tokens = self.depth_emb(topk_indices).unsqueeze(3)
            expanded_sequences = torch.cat([expanded_sequences, next_tokens], dim=3)

            # Expand scores
            expanded_scores = scores.unsqueeze(2) + topk_log_probs

            # Start flattening before selection
            expanded_scores = expanded_scores.view(B, -1)
            expanded_sequences = expanded_sequences.view(
                B, -1, expanded_sequences.size(-2), D
            )
            expanded_indices = expanded_indices.view(B, -1, expanded_indices.size(-1))

            # Select topk from flattened scores
            topk_scores, topk_indices = torch.topk(expanded_scores, b, dim=-1)

            # Shape: (B, b, L+1, D)
            sequences = expanded_sequences[arranged, topk_indices]
            indices = expanded_indices[arranged, topk_indices]

            scores = topk_scores  # Shape: (B, b)

        if self.filter_preds:
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)  # Shape B x b

            scores[is_in_query] = -torch.inf

            topk_scores, topk_indices = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices
