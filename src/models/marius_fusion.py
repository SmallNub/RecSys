import math
from dataclasses import dataclass
import torch
import torch.nn as nn
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


class RoPE:
    def __init__(self, head_dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.inv_freq = inv_freq
        self._cache = {}

    def get(self, seq_len, device):
        key = (seq_len, device)
        if key in self._cache:
            return self._cache[key]

        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))

        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        self._cache[key] = (cos, sin)
        return cos, sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, d_model, d_head, use_rope=True):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.use_rope = use_rope

        self.n_heads = d_model // d_head
        self.n_kv_heads = max(1, self.n_heads // 4)

        self.q_proj = nn.Linear(d_model, self.n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if self.use_rope:
            self.rope = RoPE(d_head)

    def forward(self, x, attn_mask=None, is_causal=False):
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "b l (h d) -> b h l d", d=self.d_head)
        k = rearrange(k, "b l (h d) -> b h l d", d=self.d_head)
        v = rearrange(v, "b l (h d) -> b h l d", d=self.d_head)

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if self.use_rope:
            cos, sin = self.rope.get(L, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal
        )

        y = rearrange(y, "b h l d -> b l (h d)")
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = 4 * d_model * 2 // 3
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TemporalBlock(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, d_head, use_rope=True)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, attn_mask=None, is_causal=False):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x


class DepthBlock(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, d_head, use_rope=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, attn_mask=None, is_causal=False):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + self.ffn(self.norm2(x))
        return x


class MARIUS(nn.Module):
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
        self.cosette = cosette
        self.filter_preds = filter_preds

        assert depth_cfg.vocab_size == temporal_cfg.vocab_size, "Vocab size mismatch"

        if self.temporal_cfg.emb_dropout is None:
            self.temporal_cfg.emb_dropout = self.temporal_cfg.dropout
        if self.depth_cfg.emb_dropout is None:
            self.depth_cfg.emb_dropout = self.depth_cfg.dropout

        self.temp_emb = nn.Embedding(
            temporal_cfg.vocab_size,
            temporal_cfg.d_model,
            padding_idx=SpecialTokens.PAD.value,
        )

        self.depth_emb = self.temp_emb if tie_embeddings else nn.Embedding(
            depth_cfg.vocab_size,
            depth_cfg.d_model,
            padding_idx=SpecialTokens.PAD.value,
        )

        self.depth_pos_emb = nn.Embedding(128, depth_cfg.d_model)

        self.temp_proj = nn.Linear(cosette.model.centroids_dim, temporal_cfg.d_model, bias=False)
        self.depth_proj = nn.Linear(cosette.model.centroids_dim, depth_cfg.d_model, bias=False)

        self.dropout_t = nn.Dropout(temporal_cfg.emb_dropout)
        self.dropout_d = nn.Dropout(depth_cfg.emb_dropout)

        self.temp_tf = nn.ModuleList([
            TemporalBlock(temporal_cfg.d_model, temporal_cfg.d_head) for _ in range(temporal_cfg.n_layers)
        ])

        self.depth_tf = nn.ModuleList([
            DepthBlock(depth_cfg.d_model, depth_cfg.d_head) for _ in range(depth_cfg.n_layers)
        ])

        self.temp_final_norm = nn.RMSNorm(temporal_cfg.d_model)
        self.depth_final_norm = nn.LayerNorm(depth_cfg.d_model)
        self.fuse_norm = nn.LayerNorm(depth_cfg.d_model)

        self.mid_proj = nn.Linear(temporal_cfg.d_model, depth_cfg.d_model, bias=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n or "depth_pos_emb" in n or "norm" in n
        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        B, L, K = input.shape

        discrete = self.temp_emb(input).sum(dim=-2)
        if self.training:
            drop = (torch.rand(B, L, 1, device=input.device) > 0.2).to(discrete.dtype)
            discrete = discrete * drop

        continuous = self.temp_proj(self.cosette.decode(input))
        x = self.dropout_t(discrete + continuous)

        valid_mask = (input[:, :, 0] != SpecialTokens.PAD.value)
        causal_mask = torch.tril(torch.ones(L, L, device=input.device, dtype=torch.bool))
        combined_mask = valid_mask.unsqueeze(1).unsqueeze(2) & causal_mask.unsqueeze(0).unsqueeze(1)

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

        dec = self.depth_emb(target[:, :-1])
        K = dec.shape[1]

        if K > 0:
            res = []
            for i in range(K):
                q = self.cosette.model.rq.vq_layers[i]
                idx = target[:, i]

                valid = (idx >= 0) & (idx < getattr(q, "n_centroids", 32000))
                safe = torch.where(valid, idx, torch.zeros_like(idx))

                x = q.get_codebook_entry(safe, shape=None)
                res.append(x * valid.unsqueeze(-1).to(x.dtype))

            res = torch.stack(res, dim=1)
            cont = self.depth_proj(res)
            
            fused_tgt = self.fuse_norm(dec + cont)
            tgt = torch.cat([mid, fused_tgt], dim=1)
        else:
            tgt = mid

        logits, hidden_states = self.depth_forward(tgt)
        return logits, hidden_states, mid, target

    def get_loss(self, batch):
        inp, tgt = batch["input"], batch["target"]
        logits, _, _, tgt = self.train_forward(inp, tgt)
        logits_rearranged = rearrange(logits, "b l v -> b v l")
        return self.criterion(logits_rearranged, tgt), logits

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

        q = self.cosette.model.rq.vq_layers[0]
        num_embeddings = getattr(q, "n_centroids", 32000)
        valid_mask = (topk_indices >= 0) & (topk_indices < num_embeddings)
        safe_indices = torch.where(valid_mask, topk_indices, torch.zeros_like(topk_indices))

        raw_new_tokens = q.get_codebook_entry(safe_indices, shape=None)
        continuous_new = self.depth_proj(raw_new_tokens)
        v_mask = valid_mask.unsqueeze(-1).to(continuous_new.dtype)

        fused_new = self.fuse_norm(discrete_new + (continuous_new * v_mask))
        sequences = torch.cat([sequences, fused_new.unsqueeze(2)], dim=2)  

        for i in range(2, L + 1):
            logits, _ = self.depth_forward(sequences.view(B * b, i, D))
            last_logits = logits[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)  

            topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

            topk_log_probs = topk_log_probs.view(B, b, b)
            topk_indices = topk_indices.view(B, b, b)

            expanded_indices = indices.unsqueeze(2).repeat(1, 1, b, 1)
            expanded_indices = torch.cat([expanded_indices, topk_indices.unsqueeze(-1)], dim=3)

            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)

            flat_topk = topk_indices.view(-1)
            flat_discrete = self.depth_emb(flat_topk)

            quantizer = self.cosette.model.rq.vq_layers[i - 1]
            num_embeddings = getattr(quantizer, "n_centroids", 32000)
            valid_mask = (flat_topk >= 0) & (flat_topk < num_embeddings)
            safe_indices = torch.where(valid_mask, flat_topk, torch.zeros_like(flat_topk))

            flat_decoded = quantizer.get_codebook_entry(safe_indices, shape=None)
            flat_projected = self.depth_proj(flat_decoded)
            flat_valid = valid_mask.unsqueeze(-1).to(flat_projected.dtype)

            flat_fused = self.fuse_norm(flat_discrete + (flat_projected * flat_valid))
            next_tokens = flat_fused.view(B, b, b, 1, D)

            expanded_sequences = torch.cat([expanded_sequences, next_tokens], dim=3)

            expanded_scores = scores.unsqueeze(2) + topk_log_probs
            
            expanded_scores = expanded_scores.view(B, -1)
            expanded_sequences = expanded_sequences.view(B, -1, expanded_sequences.size(-2), D)
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
