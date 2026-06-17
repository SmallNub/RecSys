from dataclasses import dataclass
import math
import torch
import torch.nn as nn
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


class RoPE(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def get(self, seq_len, device, dtype):
        key = (seq_len, device, dtype)
        if key in self._cache:
            return self._cache[key]

        # SMALL IMPROVEMENT: Clear cache if switching devices to prevent GPU memory leaks
        if len(self._cache) > 16:
            self._cache.clear()

        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))

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
        
        # OPTIMIZATION: Fused QKV layer
        self.qkv_proj = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if self.use_rope:
            self.rope = RoPE(d_head)

    def forward(self, x, attn_mask=None, is_causal=False):
        B, L, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", d=self.d_head)
        k = rearrange(k, "b l (h d) -> b h l d", d=self.d_head)
        v = rearrange(v, "b l (h d) -> b h l d", d=self.d_head)

        if self.use_rope:
            cos, sin = self.rope.get(L, x.device, x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )

        y = rearrange(y, "b h l d -> b l (h d)")
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = 4 * d_model * 2 // 3
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        
        # OPTIMIZATION: Fused SwiGLU projection
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
    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        tie_embeddings=False,
        filter_preds=False,
        ssl_weight=0.1,
        ssl_mask_prob=0.2,
        ssl_temperature=0.07,
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds
        self.ssl_weight = ssl_weight
        self.ssl_mask_prob = ssl_mask_prob
        self.ssl_temperature = ssl_temperature

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

        self.dropout_t = nn.Dropout(temporal_cfg.emb_dropout)
        self.dropout_d = nn.Dropout(depth_cfg.emb_dropout)

        self.temp_tf = nn.ModuleList([
            TemporalBlock(temporal_cfg.d_model, temporal_cfg.d_head, dropout_p=temporal_cfg.dropout)
            for _ in range(temporal_cfg.n_layers)
        ])

        self.depth_tf = nn.ModuleList([
            DepthBlock(depth_cfg.d_model, depth_cfg.d_head, dropout_p=depth_cfg.dropout)
            for _ in range(depth_cfg.n_layers)
        ])

        self.temp_final_norm = nn.RMSNorm(temporal_cfg.d_model)
        self.depth_final_norm = nn.LayerNorm(depth_cfg.d_model)

        self.mid_proj = nn.Linear(temporal_cfg.d_model, depth_cfg.d_model, bias=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # SMALL IMPROVEMENT: Automatically apply professional transformer weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        SMALL IMPROVEMENT: Fixed scaling initialization for Deep Residual Transformers.
        Prevents variance explosion in early stages of training.
        """
        if isinstance(module, nn.Linear):
            # Standard initialization for input mappings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        # Scale output projections by number of total layers to stabilize residual stream updates
        for block in self.temp_tf:
            if hasattr(block.attn, 'o_proj'):
                torch.nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.temporal_cfg.n_layers))
            if hasattr(block.ffn, 'w3'):
                torch.nn.init.normal_(block.ffn.w3.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.temporal_cfg.n_layers))
                
        for block in self.depth_tf:
            if hasattr(block.attn, 'o_proj'):
                torch.nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.depth_cfg.n_layers))
            if hasattr(block.ffn, 'w3'):
                torch.nn.init.normal_(block.ffn.w3.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.depth_cfg.n_layers))

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n or "depth_pos_emb" in n or "norm" in n
        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        B, L, K = input.shape

        discrete = self.temp_emb(input).sum(dim=-2)
        x = self.dropout_t(discrete)

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

    def train_forward_fused(self, temporal, target):
        """
        OPTIMIZATION: Accepts a precomputed temporal representation 
        to seamlessly allow single-pass batched computation pipelines.
        """
        mid = self.mid_proj(temporal)
        mid = rearrange(mid, "b l d -> (b l) 1 d")

        target = rearrange(target, "b l k -> (b l) k")
        keep = target[:, 0] != -100

        mid = mid[keep]
        target = target[keep]

        safe_target = target[:, :-1].clone()
        safe_target[safe_target == -100] = SpecialTokens.PAD.value
        dec = self.depth_emb(safe_target)
        K = dec.shape[1]

        if K > 0:
            tgt = torch.cat([mid, dec], dim=1)
        else:
            tgt = mid

        logits, hidden_states = self.depth_forward(tgt)
        return logits, hidden_states, mid, target

    def _augment_sequence(self, input):
        B, L, K = input.shape
        mask = torch.rand((B, L, 1), device=input.device) > self.ssl_mask_prob
        augmented = input.clone()
        augmented = torch.where(mask, augmented, SpecialTokens.PAD.value)
        return augmented

    def compute_ssl_loss(self, out1, out2):
        B = out1.shape[0]
        emb1 = F.normalize(out1.mean(dim=1), dim=-1)
        emb2 = F.normalize(out2.mean(dim=1), dim=-1)
        
        similarity_matrix = torch.matmul(emb1, emb2.T) / self.ssl_temperature
        labels = torch.arange(B, device=out1.device)
        
        loss_view1 = F.cross_entropy(similarity_matrix, labels)
        loss_view2 = F.cross_entropy(similarity_matrix.T, labels)
        return (loss_view1 + loss_view2) / 2.0

    def get_loss(self, batch):
        inp, tgt = batch["input"], batch["target"]
        
        # OPTIMIZATION: Check if SSL step is active and compress 3 passes down into 1 fused pass
        if self.training and self.ssl_weight > 0.0:
            aug_inp1 = self._augment_sequence(inp)
            aug_inp2 = self._augment_sequence(inp)
            
            # Pack all 3 views into a single batch dimension allocation
            fused_inp = torch.cat([inp, aug_inp1, aug_inp2], dim=0)
            fused_temporal_out = self.temporal_forward(fused_inp)
            
            # Slice the combined matrix back out effortlessly
            B = inp.shape[0]
            temporal, out1, out2 = torch.split(fused_temporal_out, [B, B, B], dim=0)
            
            # Compute recommendation cross-entropy
            logits, _, _, target_rearranged = self.train_forward_fused(temporal, tgt)
            logits_rearranged = rearrange(logits, "b l v -> b v l")
            rec_loss = self.criterion(logits_rearranged, target_rearranged)
            
            # Compute contrastive profile alignment
            ssl_loss = self.compute_ssl_loss(out1, out2)
            
            return rec_loss + (self.ssl_weight * ssl_loss), logits
            
        else:
            # Fallback for evaluation / validation validation sets
            temporal = self.temporal_forward(inp)
            logits, _, _, target_rearranged = self.train_forward_fused(temporal, tgt)
            logits_rearranged = rearrange(logits, "b l v -> b v l")
            rec_loss = self.criterion(logits_rearranged, target_rearranged)
            return rec_loss, logits

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
        sequences = torch.cat([sequences, discrete_new.unsqueeze(2)], dim=2)

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
            next_tokens = flat_discrete.view(B, b, b, 1, D)

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
