from dataclasses import dataclass
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
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds

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
            torch.empty((1, self.temporal_cfg.seq_len, self.temporal_cfg.d_model))
        )
        self.depth_pos_emb = torch.nn.Parameter(
            torch.empty((1, self.depth_cfg.seq_len, self.depth_cfg.d_model))
        )

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

        self.mid_proj = torch.nn.Linear(
            self.temporal_cfg.d_model, self.depth_cfg.d_model
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.reset_marius_parameters()

    def reset_marius_parameters(self):
        """Applies stable standard-deviation initialization across discrete spaces."""
        torch.nn.init.trunc_normal_(self.temp_pos_emb, std=0.02)
        torch.nn.init.trunc_normal_(self.depth_pos_emb, std=0.02)

        torch.nn.init.trunc_normal_(self.temp_emb.weight, std=0.02)
        if not self.tie_embeddings:
            torch.nn.init.trunc_normal_(self.depth_emb.weight, std=0.02)

        with torch.no_grad():
            self.temp_emb.weight[SpecialTokens.PAD.value].zero_()
            if not self.tie_embeddings:
                self.depth_emb.weight[SpecialTokens.PAD.value].zero_()

        torch.nn.init.trunc_normal_(self.mid_proj.weight, std=0.02)
        if self.mid_proj.bias is not None:
            torch.nn.init.constant_(self.mid_proj.bias, 0.0)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                if m != self.mid_proj:
                    torch.nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0.0)
                torch.nn.init.constant_(m.weight, 1.0)

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n

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

        depth_preds = self.depth_tf(
            in_embs, mask=self.causal_mask[:K, :K].to(in_embs.device)
        )
        logits = torch.einsum("bkd, vd -> bkv", depth_preds, self.depth_emb.weight)

        return logits

    def train_forward(self, input, target):
        temporal_tokens = self.temporal_forward(input)
        mid_tokens = self.mid_proj(temporal_tokens)
        mid_tokens = rearrange(mid_tokens, "b l d -> (b l) 1 d")

        target = rearrange(target, "b l k -> (b l) k")
        keep = target[:, 0] != -100

        mid_tokens = mid_tokens[keep]
        target = target[keep]

        dec_embs = self.depth_emb(target[:, :-1])
        dec_embs = torch.cat([mid_tokens, dec_embs], dim=1)

        depth_logits = self.depth_forward(dec_embs)
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

        temporal_tokens = self.temporal_forward(input)
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :]

        B, b, D = input.shape[0], n_results, self.depth_cfg.d_model

        sequences = mid_tokens[:, None, :]
        depth_logits = self.depth_forward(sequences)
        log_probs = F.log_softmax(depth_logits[:, -1, :], dim=-1)

        topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

        indices = topk_indices.unsqueeze(2)
        scores = topk_log_probs

        sequences = sequences.unsqueeze(1).repeat(1, b, 1, 1)

        new_tokens = self.depth_emb(topk_indices).unsqueeze(2)
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
            next_tokens = self.depth_emb(topk_indices).unsqueeze(3)
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

            scores = topk_scores

        if self.filter_preds:
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)

            scores[is_in_query] = -torch.inf

            topk_scores, topk_indices = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices
