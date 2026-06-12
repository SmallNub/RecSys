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
        tie_embeddings=False,
        filter_preds=False,
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds
        self.cosette = cosette

        assert (
            self.depth_cfg.vocab_size == self.temporal_cfg.vocab_size
        ), "Vocab size mismatch"

        if self.temporal_cfg.emb_dropout is None:
            self.temporal_cfg.emb_dropout = self.temporal_cfg.dropout
        if self.depth_cfg.emb_dropout is None:
            self.depth_cfg.emb_dropout = self.depth_cfg.dropout

        # Original Discrete Embeddings
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

        # =========================================================================
        # FIX: ZERO-INITIALIZED RESIDUAL PROJECTIONS
        # =========================================================================
        self.temp_proj = torch.nn.Linear(self.cosette.model.in_dim, self.temporal_cfg.d_model)
        self.depth_proj = torch.nn.Linear(self.cosette.model.centroids_dim, self.depth_cfg.d_model)

        # Initialize to exact zeroes. This guarantees identical step 0 performance
        # to the baseline, allowing organic gradient-driven fade-in.
        torch.nn.init.zeros_(self.temp_proj.weight)
        torch.nn.init.zeros_(self.temp_proj.bias)
        torch.nn.init.zeros_(self.depth_proj.weight)
        torch.nn.init.zeros_(self.depth_proj.bias)

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

        # Transformer
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
            return "temp_emb" in n or "depth_emb" in n

        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        print("No decay :", len(no_decay))
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]

        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        B, L, K = input.shape

        # Original discrete extraction
        discrete_embs = self.temp_emb(input).sum(dim=-2)
        
        # Extract Cosette full-state continuous representation
        decoded_features = self.cosette.decode(input)
        continuous_embs = self.temp_proj(decoded_features)
        print("TEMP")
        print(discrete_embs.norm(dim=-1).mean())
        print(continuous_embs.norm(dim=-1).mean())

        # Zero-Init addition guarantees safety
        input_embs = discrete_embs #+ continuous_embs
        
        input_embs += self.temp_pos_emb[:, : input.shape[1], :]
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

        logits = torch.einsum("bkd, vd -> bkv", depth_preds, self.depth_emb.weight)

        return logits

    def train_forward(self, input, target):
        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)  # B x L x d
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")  # BL x 1 x D

        target = rearrange(target, "b l k -> (b l) k")  # BL x K
        keep = target[:, 0] != -100

        mid_tokens = mid_tokens[keep]
        target = target[keep]

        # Extract native discrete embeddings directly to preserve exact logic
        dec_embs_orig = self.depth_emb(target[:, :-1])

        # Safely extract and inject residual continuous embeddings
        K_minus_1 = dec_embs_orig.shape[1]
        if K_minus_1 > 0:
            cont_res_list = []
            for i in range(K_minus_1):
                quantizer = self.cosette.model.rq.vq_layers[i]
                layer_indices = target[:, i]

                num_embeddings = getattr(quantizer, "n_centroids", 32000)
                valid_mask = (layer_indices >= 0) & (layer_indices < num_embeddings)
                safe_indices = torch.where(valid_mask, layer_indices, torch.zeros_like(layer_indices))

                x_res = quantizer.get_codebook_entry(safe_indices, shape=None)
                continuous_res = self.depth_proj(x_res)
                
                v_mask = valid_mask.unsqueeze(-1).to(continuous_res.dtype)
                cont_res_list.append(continuous_res * v_mask)

            continuous_tensor = torch.stack(cont_res_list, dim=1)
            dec_embs = dec_embs_orig #+ continuous_tensor
        else:
            dec_embs = dec_embs_orig

        dec_embs = torch.cat([mid_tokens, dec_embs], dim=1)

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
        print("WEIGHTS")
        print(self.temp_proj.weight.norm())
        print(self.depth_proj.weight.norm())

        input = batch["input"]
        L = batch["target"].shape[-1]

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :]  # B, D

        B, b, D = input.shape[0], n_results, self.depth_cfg.d_model

        # Initialize the beam search ==============================================
        sequences = mid_tokens[:, None, :]  # B x 1 x D
        depth_logits = self.depth_forward(sequences)  # B x 1 x D
        log_probs = F.log_softmax(depth_logits[:, -1, :], dim=-1)  # B x V

        topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)  # B x b

        indices = topk_indices.unsqueeze(2)  # Shape: (B, b, 1)
        scores = topk_log_probs  # Shape: B x b
        sequences = sequences.unsqueeze(1).repeat(1, b, 1, 1)  # Shape: (B, b, 1, D)

        # FIX: Discrete embedding must look up unmodified topk_indices to preserve EOS/Padding
        discrete_new = self.depth_emb(topk_indices)

        quantizer = self.cosette.model.rq.vq_layers[0]
        num_embeddings = getattr(quantizer, "n_centroids", 32000)
        valid_mask = (topk_indices >= 0) & (topk_indices < num_embeddings)
        safe_indices = torch.where(valid_mask, topk_indices, torch.zeros_like(topk_indices))

        raw_new_tokens = quantizer.get_codebook_entry(safe_indices, shape=None)
        continuous_new = self.depth_proj(raw_new_tokens)
        print("NORMS")
        print(discrete_new.norm(dim=-1).mean())
        print(continuous_new.norm(dim=-1).mean())
        
        v_mask = valid_mask.unsqueeze(-1).to(continuous_new.dtype)
        continuous_new = continuous_new * v_mask

        # Residual Addition
        fused_new = discrete_new #+ continuous_new
        new_tokens = fused_new.unsqueeze(2)  # Shape : (B, b, 1, D)
        sequences = torch.concat([sequences, new_tokens], dim=2)

        arranged = torch.arange(B, device=sequences.device).view(-1, 1)

        # Start the beam search ===================================================
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
            
            # --- Continuous + Discrete Fusion Loop ---
            flat_topk = topk_indices.view(-1)
            flat_discrete = self.depth_emb(flat_topk)

            quantizer = self.cosette.model.rq.vq_layers[i - 1]
            num_embeddings = getattr(quantizer, "n_centroids", 32000)
            valid_mask = (flat_topk >= 0) & (flat_topk < num_embeddings)
            safe_indices = torch.where(valid_mask, flat_topk, torch.zeros_like(flat_topk))

            flat_decoded = quantizer.get_codebook_entry(safe_indices, shape=None)
            flat_projected = self.depth_proj(flat_decoded)
            
            flat_valid = valid_mask.unsqueeze(-1).to(flat_projected.dtype)
            flat_projected = flat_projected * flat_valid

            flat_fused = flat_discrete #+ flat_projected
            next_tokens = flat_fused.view(B, b, b, 1, D)
            # ----------------------------------------
            
            expanded_sequences = torch.cat([expanded_sequences, next_tokens], dim=3)

            expanded_scores = scores.unsqueeze(2) + topk_log_probs
            expanded_scores = expanded_scores.view(B, -1)
            expanded_sequences = expanded_sequences.view(
                B, -1, expanded_sequences.size(-2), D
            )
            expanded_indices = expanded_indices.view(B, -1, expanded_indices.size(-1))

            topk_scores, topk_indices = torch.topk(expanded_scores, b, dim=-1)

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
