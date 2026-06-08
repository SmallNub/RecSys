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

        # Use COSETTE as fixed embeddings
        self.temp_proj = torch.nn.Linear(
            self.cosette.model.in_dim, self.temporal_cfg.d_model
        )
        self.depth_proj = torch.nn.Linear(
            self.cosette.model.in_dim, self.depth_cfg.d_model
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
        self.output_head = torch.nn.Linear(
            self.depth_cfg.d_model, self.depth_cfg.vocab_size
        )

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
        # input shape: B x L x K (Batch, Seq_Len, Num_Quantizers)
        B, L, K = input.shape

        # 1. Decode IDs directly to continuous pre-trained latent features
        # Output shape from wrapper: B x L x Cosette_In_Dim
        decoded_features = self.cosette.decode(input)

        # 2. Project to transformer hidden dimension and combine the depth dimension
        # (Replacing the old explicit sum over raw token embeddings)
        input_embs = self.temp_proj(decoded_features)

        # 3. Inject positions and apply standard dropout
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
        logits = self.output_head(depth_preds)  # Shape: B x K x Vocab_Size
        return logits

    def train_forward(self, input, target):
        temporal_tokens = self.temporal_forward(input)  # B x L x D
        mid_tokens = self.mid_proj(temporal_tokens)  # B x L x d
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")  # BL x 1 x d

        target = rearrange(target, "b l k -> (b l) k")  # BL x K
        keep = target[:, 0] != -100

        mid_tokens = mid_tokens[keep]
        target = target[keep]

        # --- Replace standard lookup with COSETTE decode ---
        # Shift targets for teacher forcing and decode through the wrapper
        raw_dec_features = self.cosette.decode(
            target[:, :-1]
        )  # BL x (K-1) x Cosette_In_Dim
        dec_embs = self.depth_proj(raw_dec_features)  # BL x (K-1) x d_model_depth

        # Concatenate temporal context prefix with your decoded targets
        dec_embs = torch.cat([mid_tokens, dec_embs], dim=1)

        depth_logits = self.depth_forward(dec_embs)  # BL x K x V
        return depth_logits, target

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]

        logits, m_target = self.train_forward(input, target)

        # Torch CELoss expects (N, C, d_i...) and (N, d_i...) as input
        logits = rearrange(logits, "B k v -> B v k")

        loss = self.criterion(logits, m_target)

        return loss, logits

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
        depth_logits = self.depth_forward(sequences)  # B x 1 x D
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
        # Starting with 1 token, generated 1... need L-1 more.
        for i in range(2, L + 1):
            # Forward the current sequence to get logits of the next token
            last_logits = self.depth_forward(sequences.view(B * b, i, D))[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)  # B * b x V

            topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

            # Shape: (B * b, b) -> (B, b, b)
            topk_log_probs = topk_log_probs.view(B, b, b)
            topk_indices = topk_indices.view(B, b, b)

            # Expand indices - Update with top values
            # (B x b x L) -> B x b x b x L+1
            expanded_indices = indices.unsqueeze(2).repeat(1, 1, b, 1)
            expanded_indices = torch.cat(
                [expanded_indices, topk_indices.unsqueeze(-1)], dim=3
            )

            # Expand sequences
            # Shape: (B, b, L, D) -> (B, b, b, L, D)
            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)
            # Choose the new tokens - Shape: (B, b, b, 1, D)
            next_tokens = self.depth_emb(topk_indices).unsqueeze(3)
            expanded_sequences = torch.cat([expanded_sequences, next_tokens], dim=3)

            # Expand scores
            # Update scores - Shape: (B, b, b)
            expanded_scores = scores.unsqueeze(2) + topk_log_probs

            # Start flattening before selection
            # Shape: (B, b * b)
            expanded_scores = expanded_scores.view(B, -1)
            # Shape: (B, b * b, L+1, D)
            expanded_sequences = expanded_sequences.view(
                B, -1, expanded_sequences.size(-2), D
            )
            # Shape : (B, b * b, L)
            expanded_indices = expanded_indices.view(B, -1, expanded_indices.size(-1))

            # Select topk from flattened scores
            # Shape: (B, b)
            topk_scores, topk_indices = torch.topk(expanded_scores, b, dim=-1)

            # Shape: (B, b, L+1, D)
            sequences = expanded_sequences[arranged, topk_indices]
            indices = expanded_indices[arranged, topk_indices]

            scores = topk_scores  # Shape: (B, b)

        if self.filter_preds:
            # indices : B x b x L
            # input : B x T x L
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)  # Shape B x b

            scores[is_in_query] = -torch.inf

            topk_scores, topk_indices = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices
