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
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds
        self.cosette = cosette
        self.num_quantizers = self.cosette.model.rq.num_quantizers

        assert (
            self.depth_cfg.vocab_size == self.temporal_cfg.vocab_size
        ), "Vocab size mismatch"

        if self.temporal_cfg.emb_dropout is None:
            self.temporal_cfg.emb_dropout = self.temporal_cfg.dropout
        if self.depth_cfg.emb_dropout is None:
            self.depth_cfg.emb_dropout = self.depth_cfg.dropout

        # Linear projection from continuous COSETTE full item manifold to temporal transformer space
        self.temp_proj = torch.nn.Linear(
            self.cosette.model.in_dim, self.temporal_cfg.d_model
        )

        # Linear projection from COSETTE codebook space (centroids_dim) to depth transformer space
        self.depth_proj = torch.nn.Linear(
            self.cosette.model.centroids_dim, self.depth_cfg.d_model
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

        # Output Classification Heads mapping back to discrete IDs layer-by-layer
        self.output_heads = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.depth_cfg.d_model, self.depth_cfg.vocab_size)
                for _ in range(self.num_quantizers)
            ]
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

        # Projection
        self.mid_proj = torch.nn.Linear(
            self.temporal_cfg.d_model, self.depth_cfg.d_model
        )

        # Loss
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def get_param_groups(self):
        def _select_no_decay(n):
            return "bias" in n or "norm" in n

        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]

        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        # input shape: B x L x K
        B, L, K = input.shape

        # Decode item sequence representation using COSETTE full item decoder
        decoded_features = self.cosette.decode(input)  # B x L x Cosette_In_Dim
        input_embs = self.temp_proj(decoded_features)  # B x L x d_model_temporal

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

        # Route through layer-specific linear classification heads
        logits_list = []
        for k in range(K):
            logits_k = self.output_heads[k](depth_preds[:, k, :])
            logits_list.append(logits_k)

        logits = torch.stack(logits_list, dim=1)  # X x K x V
        return logits

    def train_forward(self, input, target):
        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)  # B x L x d
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")  # BL x 1 x d

        target = rearrange(target, "b l k -> (b l) k")  # BL x K
        keep = target[:, 0] != -100

        mid_tokens = mid_tokens[keep]
        target = target[keep]

        # Extract codebook vectors layer-by-layer matching depth positions
        dec_embs_list = []
        for i in range(target.shape[-1] - 1):
            quantizer = self.cosette.model.rq.vq_layers[i]
            layer_indices = target[:, i]

            # Defensive boundary check
            num_embeddings = getattr(quantizer, "n_centroids", 32000)
            layer_indices = torch.clamp(layer_indices, 0, num_embeddings - 1)

            x_res = quantizer.get_codebook_entry(layer_indices, shape=None)  # BL x centroids_dim
            dec_embs_list.append(x_res)

        dec_embs = torch.stack(dec_embs_list, dim=1)  # BL x (K-1) x centroids_dim
        dec_embs = self.depth_proj(dec_embs)  # BL x (K-1) x d_model_depth

        # Concatenate summary start token with layer token embeddings
        dec_embs = torch.cat([mid_tokens, dec_embs], dim=1)  # BL x K x d_model_depth

        depth_logits = self.depth_forward(dec_embs)  # BL x K x V
        return depth_logits, target

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]

        logits, m_target = self.train_forward(input, target)
        logits = rearrange(logits, "B k v -> B v k")

        loss = self.criterion(logits, m_target)
        return loss, logits

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode (dropout)."

        input = batch["input"]
        L = batch["target"].shape[-1]

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :]  # B, D

        B, b, D = input.shape[0], n_results, self.depth_cfg.d_model

        # =========================================================================
        # BEAM SEARCH INITIALIZATION
        # =========================================================================
        sequences = mid_tokens[:, None, :]  # B x 1 x D
        depth_logits = self.depth_forward(sequences)  # B x 1 x V
        log_probs = F.log_softmax(depth_logits[:, -1, :], dim=-1)  # B x V

        topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)  # B x b

        indices = topk_indices.unsqueeze(2)  # Shape: (B, b, 1)
        scores = topk_log_probs  # Shape: B x b

        sequences = sequences.unsqueeze(1).repeat(1, b, 1, 1)  # Shape: (B, b, 1, D)

        # Extract continuous codebook representation from Layer 0
        quantizer_0 = self.cosette.model.rq.vq_layers[0]
        num_embeddings_0 = getattr(quantizer_0, "n_centroids", 32000)
        safe_topk_indices = torch.clamp(topk_indices, 0, num_embeddings_0 - 1)

        raw_new_tokens = quantizer_0.get_codebook_entry(safe_topk_indices, shape=None)  # B x b x centroids_dim
        new_tokens = self.depth_proj(raw_new_tokens).unsqueeze(2)  # Shape: (B, b, 1, D)
        sequences = torch.concat([sequences, new_tokens], dim=2)

        arranged = torch.arange(B, device=sequences.device).view(-1, 1)

        # =========================================================================
        # AUTOREGRESSIVE GENERATION LOOP
        # =========================================================================
        for i in range(2, L + 1):
            last_logits = self.depth_forward(sequences.view(B * b, i, D))[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)  # B * b x V

            topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

            topk_log_probs = topk_log_probs.view(B, b, b)
            topk_indices = topk_indices.view(B, b, b)

            expanded_indices = indices.unsqueeze(2).repeat(1, 1, b, 1)
            expanded_indices = torch.cat(
                [expanded_indices, topk_indices.unsqueeze(-1)], dim=3
            )

            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)

            # Query the precise hierarchical layer (i - 1) for newly added beams
            quantizer_i = self.cosette.model.rq.vq_layers[i - 1]
            num_embeddings_i = getattr(quantizer_i, "n_centroids", 32000)
            safe_topk_indices_i = torch.clamp(topk_indices, 0, num_embeddings_i - 1)
            
            raw_new_tokens_i = quantizer_i.get_codebook_entry(safe_topk_indices_i, shape=None)  # B x b x b x centroids_dim
            next_tokens = self.depth_proj(raw_new_tokens_i).unsqueeze(3)  # B x b x b x 1 x D
            expanded_sequences = torch.concat([expanded_sequences, next_tokens], dim=3)

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
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)  # Shape B x b

            scores[is_in_query] = -torch.inf

            topk_scores, topk_indices_choice = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices_choice]

        return indices
