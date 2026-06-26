"""MARIUS Student model architecture for knowledge distillation."""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models import SpecialTokens
from src.models.utils import load_model


@dataclass
class TransformerConfig:
    n_layers: int
    d_head: int
    d_model: int
    vocab_size: int
    dropout: float
    emb_dropout: float
    seq_len: int


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) module."""
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, d_model, d_head, dropout_p=0.1, use_rope=True):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.use_rope = use_rope
        self.dropout_p = dropout_p

        self.n_heads = d_model // d_head
        self.n_kv_heads = max(1, self.n_heads // 4)

        self.q_dim = self.n_heads * d_head
        self.kv_dim = self.n_kv_heads * d_head

        self.qkv_proj = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = nn.RMSNorm(d_head)
        self.k_norm = nn.RMSNorm(d_head)

        if self.use_rope:
            self.rope = RoPE(d_head)

    def forward(self, x, attn_mask=None, is_causal=False):
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", d=self.d_head)
        k = rearrange(k, "b l (h d) -> b h l d", d=self.d_head)
        v = rearrange(v, "b l (h d) -> b h l d", d=self.d_head)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            cos, sin = self.rope.get(L, x.device, x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )

        y = rearrange(y, "b h l d -> b l (h d)")
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = 4 * d_model * 2 // 3
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.w12 = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        w1_out, w2_out = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(w1_out) * w2_out)


class TemporalBlock(nn.Module):
    def __init__(self, d_model, d_head, dropout_p=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, d_head, dropout_p=dropout_p, use_rope=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, attn_mask=None, is_causal=False):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x


class DepthBlock(nn.Module):
    def __init__(self, d_model, d_head, dropout_p=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, d_head, dropout_p=dropout_p, use_rope=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, attn_mask=None, is_causal=False):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x


class MARIUS(nn.Module):
    """MARIUS Student model, designed for knowledge distillation from a pre-trained teacher model."""
    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        tie_embeddings=False,
        filter_preds=False,
        distill_temp=4.0,
        ce_weight=0.1,
        distill_weight=1.0,
        student_mask_prob=0.25,
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds
        self.distill_temp = distill_temp
        self.ce_weight = ce_weight
        self.distill_weight = distill_weight
        self.student_mask_prob = student_mask_prob

        assert depth_cfg.vocab_size == temporal_cfg.vocab_size, "Vocab size mismatch"

        t_emb_drop = (
            temporal_cfg.emb_dropout
            if temporal_cfg.emb_dropout is not None
            else temporal_cfg.dropout
        )
        d_emb_drop = (
            depth_cfg.emb_dropout
            if depth_cfg.emb_dropout is not None
            else depth_cfg.dropout
        )

        # Load Pre-trained Fixed Expert Teacher
        self.teacher = load_model(
            "outputs/checkpoints/marius/MARIUS_teacher_260619_133025"
        )
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        self.temp_emb = nn.Embedding(
            temporal_cfg.vocab_size,
            temporal_cfg.d_model,
            padding_idx=SpecialTokens.PAD.value,
        )

        self.depth_emb = (
            self.temp_emb
            if tie_embeddings
            else nn.Embedding(
                depth_cfg.vocab_size,
                depth_cfg.d_model,
                padding_idx=SpecialTokens.PAD.value,
            )
        )

        self.temp_emb_ln = nn.LayerNorm(temporal_cfg.d_model)
        self.depth_emb_ln = nn.LayerNorm(depth_cfg.d_model)
        self.depth_pos_emb = nn.Embedding(128, depth_cfg.d_model)

        self.dropout_t = nn.Dropout(t_emb_drop)
        self.dropout_d = nn.Dropout(d_emb_drop)

        self.temp_tf = nn.ModuleList(
            [
                TemporalBlock(
                    temporal_cfg.d_model,
                    temporal_cfg.d_head,
                    dropout_p=temporal_cfg.dropout,
                )
                for _ in range(temporal_cfg.n_layers)
            ]
        )

        self.depth_tf = nn.ModuleList(
            [
                DepthBlock(
                    depth_cfg.d_model, depth_cfg.d_head, dropout_p=depth_cfg.dropout
                )
                for _ in range(depth_cfg.n_layers)
            ]
        )

        self.temp_final_norm = nn.RMSNorm(temporal_cfg.d_model)
        self.depth_final_norm = nn.LayerNorm(depth_cfg.d_model)
        self.mid_proj = nn.Linear(temporal_cfg.d_model, depth_cfg.d_model, bias=False)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)

        for block in self.temp_tf:
            if hasattr(block.attn, "o_proj"):
                torch.nn.init.normal_(
                    block.attn.o_proj.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.temporal_cfg.n_layers),
                )
            if hasattr(block.ffn, "w3"):
                torch.nn.init.normal_(
                    block.ffn.w3.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.temporal_cfg.n_layers),
                )

        for block in self.depth_tf:
            if hasattr(block.attn, "o_proj"):
                torch.nn.init.normal_(
                    block.attn.o_proj.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.depth_cfg.n_layers),
                )
            if hasattr(block.ffn, "w3"):
                torch.nn.init.normal_(
                    block.ffn.w3.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.depth_cfg.n_layers),
                )

    def _corrupt_student_input(self, input_tensor):
        """Forces the student to infer missing context by randomly masking token streams."""
        corrupted = input_tensor.clone()
        # Create a mask so we only drop valid tokens, never padding tokens
        valid_mask = corrupted[:, :, 0] != SpecialTokens.PAD.value

        rand_grid = torch.rand(corrupted.shape[:2], device=input_tensor.device)
        mask_condition = (rand_grid < self.student_mask_prob) & valid_mask

        corrupted[mask_condition] = 0
        return corrupted

    def get_param_groups(self):
        def _select_no_decay(n):
            return any(k in n for k in ["_emb", "norm"])

        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        B, L, K = input.shape

        discrete = self.temp_emb(input).sum(dim=-2)
        discrete = self.temp_emb_ln(discrete)
        x = self.dropout_t(discrete)

        valid_mask = input[:, :, 0] != SpecialTokens.PAD.value
        causal_mask = torch.tril(
            torch.ones(L, L, device=input.device, dtype=torch.bool)
        )
        combined_mask = valid_mask.unsqueeze(1).unsqueeze(2) & causal_mask.unsqueeze(
            0
        ).unsqueeze(1)

        for blk in self.temp_tf:
            x = blk(x, attn_mask=combined_mask, is_causal=False)

        return self.temp_final_norm(x)

    def depth_forward(self, tgt):
        seq_len = tgt.shape[1]
        positions = torch.arange(seq_len, device=tgt.device)
        pos_embeddings = self.depth_pos_emb(positions).unsqueeze(0)

        x = self.dropout_d(tgt + pos_embeddings)

        for blk in self.depth_tf:
            x = blk(x, attn_mask=None, is_causal=True)

        hidden_states = self.depth_final_norm(x)
        logits = torch.matmul(hidden_states, self.depth_emb.weight.T)
        return logits, hidden_states

    def train_forward(self, input, target):
        temporal = self.temporal_forward(input)
        mid = self.mid_proj(temporal)
        mid = rearrange(mid, "b l d -> (b l) 1 d")

        target = rearrange(target, "b l k -> (b l) k")
        keep = target[:, 0] != -100

        mid = mid[keep]
        target = target[keep]

        safe_target = target[:, :-1].clone()
        safe_target[safe_target == -100] = SpecialTokens.PAD.value

        dec = self.depth_emb(safe_target)
        dec = self.depth_emb_ln(dec)
        K = dec.shape[1]

        if K > 0:
            tgt = torch.cat([mid, dec], dim=1)
        else:
            tgt = mid

        logits, hidden_states = self.depth_forward(tgt)
        return logits, hidden_states, mid, target

    def get_loss(self, batch):
        inp, tgt = batch["input"], batch["target"]

        # 1. Teacher Forward Pass (Pristine Input Track)
        if self.teacher is not None:
            with torch.no_grad():
                logits_teacher, *_ = self.teacher.train_forward(inp, tgt)

        # 2. Student Forward Pass (Asymmetric Masked Track)
        student_input = self._corrupt_student_input(inp) if self.training else inp
        logits_student, _, _, target_rearranged = self.train_forward(student_input, tgt)

        # 3. Suppressed Hard CE Loss Component
        logits_rearranged = rearrange(logits_student, "b l v -> b v l")
        # Standard Cross-Entropy against ground truth labels
        ce_loss = self.criterion(logits_rearranged, target_rearranged)
        final_loss = self.ce_weight * ce_loss

        # 4. Soft Target Ranking Distribution Alignment
        if self.teacher is not None:
            mask = target_rearranged != -100
            if mask.any():
                flat_student = logits_student[mask]
                flat_teacher = logits_teacher[mask]

                soft_student = F.log_softmax(flat_student / self.distill_temp, dim=-1)
                soft_teacher = F.softmax(flat_teacher / self.distill_temp, dim=-1)
                # KL Divergence between teacher soft targets and student predictions, scaled by temperature squared
                distill_loss = F.kl_div(
                    soft_student, soft_teacher, reduction="batchmean"
                ) * (self.distill_temp**2)

                final_loss += self.distill_weight * distill_loss

        return final_loss, logits_student

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode."
        input = batch["input"]
        L = batch["target"].shape[-1]

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal = self.temporal_forward(input)
        mid = self.mid_proj(temporal)[:, -1, :]

        B, b, D = input.shape[0], n_results, self.depth_cfg.d_model
        arranged_batch = torch.arange(B, device=input.device).view(-1, 1)

        sequences = mid[:, None, :]
        logits, _ = self.depth_forward(sequences)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

        indices = topk_indices.unsqueeze(2)
        scores = topk_log_probs
        sequences = sequences.unsqueeze(1).repeat(1, b, 1, 1)

        discrete_new = self.depth_emb(topk_indices)
        discrete_new = self.depth_emb_ln(discrete_new)
        sequences = torch.cat([sequences, discrete_new.unsqueeze(2)], dim=2)

        for i in range(2, L + 1):
            logits, _ = self.depth_forward(sequences.view(B * b, i, D))
            last_logits = logits[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)

            topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

            topk_log_probs = topk_log_probs.view(B, b, b)
            topk_indices = topk_indices.view(B, b, b)

            expanded_indices = indices.unsqueeze(2).repeat(1, 1, b, 1)
            expanded_indices = torch.cat(
                [expanded_indices, topk_indices.unsqueeze(-1)], dim=3
            )

            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)

            flat_topk = topk_indices.view(-1)
            flat_discrete = self.depth_emb(flat_topk)
            flat_discrete = self.depth_emb_ln(flat_discrete)
            next_tokens = flat_discrete.view(B, b, b, 1, D)

            expanded_sequences = torch.cat([expanded_sequences, next_tokens], dim=3)

            expanded_scores = scores.unsqueeze(2) + topk_log_probs

            expanded_scores = expanded_scores.view(B, -1)
            expanded_sequences = expanded_sequences.view(
                B, -1, expanded_sequences.size(-2), D
            )
            expanded_indices = expanded_indices.view(B, -1, expanded_indices.size(-1))

            topk_scores, topk_indices = torch.topk(expanded_scores, b, dim=-1)

            sequences = expanded_sequences[arranged_batch, topk_indices]
            indices = expanded_indices[arranged_batch, topk_indices]
            scores = topk_scores

        if self.filter_preds:
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)
            scores[is_in_query] = -torch.inf
            topk_scores, topk_indices = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged_batch, topk_indices]

        return indices
