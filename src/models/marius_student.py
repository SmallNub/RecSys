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
        top_k=10,
        truncation_weight=0.25,
        listwise_weight=0.40,
        repr_weight=0.35,
        # Balanced 3-Phase Schedule Configuration
        phase1_duration=20000,
        phase2_duration=15000,
        phase3_duration=5000,
        distill_temp=3.0,          
    ):
        super().__init__()
        self.temporal_cfg = temporal_cfg
        self.depth_cfg = depth_cfg
        self.filter_preds = filter_preds
        
        self.top_k = top_k
        self.truncation_weight = truncation_weight
        self.listwise_weight = listwise_weight
        self.repr_weight = repr_weight

        # Set up explicit milestones
        self.p1_end = phase1_duration
        self.p2_end = phase1_duration + phase2_duration
        self.total_steps = phase1_duration + phase2_duration + phase3_duration
        
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))

        assert depth_cfg.vocab_size == temporal_cfg.vocab_size, "Vocab size mismatch"

        if self.temporal_cfg.emb_dropout is None:
            self.temporal_cfg.emb_dropout = self.temporal_cfg.dropout
        if self.depth_cfg.emb_dropout is None:
            self.depth_cfg.emb_dropout = self.depth_cfg.dropout

        # Setup Distillation attributes
        self.teacher = load_model("outputs/checkpoints/marius/MARIUS_teacher_260619_133025")
        self.distill_temp = distill_temp
        if self.teacher is not None:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

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

        self.temp_emb_ln = nn.LayerNorm(temporal_cfg.d_model)
        self.depth_emb_ln = nn.LayerNorm(depth_cfg.d_model)

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
        
        self.listwise_temp = nn.Parameter(torch.tensor(0.0))
        self.repr_temp = nn.Parameter(torch.tensor(0.0))

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
            return any(k in n for k in ["_emb", "norm"]) or n.endswith("_temp")
        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        B, L, K = input.shape

        discrete = self.temp_emb(input).sum(dim=-2)
        discrete = self.temp_emb_ln(discrete)
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

    def compute_softmax_loss_at_k(self, logits, targets, k=10):
        mask = (targets != -100) & (targets != SpecialTokens.PAD.value)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        flat_logits = logits[mask]
        flat_targets = targets[mask]

        pos_scores = flat_logits.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
        topk_scores, _ = torch.topk(flat_logits, k=k, dim=-1)
        
        max_scores = torch.max(topk_scores, dim=-1)[0]
        shifted_topk = topk_scores - max_scores.unsqueeze(-1)
        shifted_pos = pos_scores - max_scores

        denom = torch.log(torch.exp(shifted_pos) + torch.exp(shifted_topk).sum(dim=-1) + 1e-8) + max_scores
        return torch.mean(denom - pos_scores)

    def compute_plackett_luce_loss(self, logits, targets):
        mask = (targets != -100) & (targets != SpecialTokens.PAD.value)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)

        flat_logits = logits[mask]
        flat_targets = targets[mask]

        pos_scores = flat_logits.gather(dim=-1, index=flat_targets.unsqueeze(-1))
        topk_neg, _ = torch.topk(flat_logits, k=32, dim=-1)
        
        slate = torch.cat([pos_scores, topk_neg], dim=-1)
        t = torch.exp(self.listwise_temp).clamp(min=1e-3, max=5.0)
        scaled_slate = slate / t

        max_val = torch.max(scaled_slate, dim=-1, keepdim=True)[0]
        norm_slate = scaled_slate - max_val
        denom = torch.log(torch.exp(norm_slate).sum(dim=-1) + 1e-8) + max_val.squeeze(-1)
        log_prob = scaled_slate[:, 0] - denom

        return torch.mean(-log_prob)

    def compute_infonce_loss(self, hidden_states, targets):
        mask = (targets != -100) & (targets != SpecialTokens.PAD.value)
        if not mask.any():
            return torch.tensor(0.0, device=hidden_states.device)

        flat_h = hidden_states[mask]
        flat_targets = targets[mask]

        pos_emb = self.depth_emb(flat_targets)
        
        flat_h_norm = F.normalize(flat_h, p=2, dim=-1)
        pos_emb_norm = F.normalize(pos_emb, p=2, dim=-1)

        similarity_matrix = torch.matmul(flat_h_norm, pos_emb_norm.T)
        
        t = torch.exp(self.repr_temp).clamp(min=1e-2, max=1.0)
        logits = similarity_matrix / t
        
        collision_mask = flat_targets.unsqueeze(0) == flat_targets.unsqueeze(1)
        identity_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(collision_mask & ~identity_mask, -1e9)
        
        labels = torch.arange(logits.shape[0], device=hidden_states.device)
        return F.cross_entropy(logits, labels)

    def get_loss(self, batch):
        inp, tgt = batch["input"], batch["target"]
        
        # 1. Base Forward Passes
        temporal = self.temporal_forward(inp)
        logits, hidden_states, _, target_rearranged = self.train_forward_fused(temporal, tgt)
        
        # 2. Compute Raw Underlying Losses
        logits_rearranged = rearrange(logits, "b l v -> b v l")
        ce_loss = self.criterion(logits_rearranged, target_rearranged)
        
        trunc_loss = self.compute_softmax_loss_at_k(logits, target_rearranged, k=self.top_k)
        listwise_loss = self.compute_plackett_luce_loss(logits, target_rearranged)
        repr_loss = self.compute_infonce_loss(hidden_states, target_rearranged)
        
        total_structural = (
            self.truncation_weight * trunc_loss + 
            self.listwise_weight * listwise_loss + 
            self.repr_weight * repr_loss
        )

        # 3. Comprehensive 3-Phase Scheduling Core
        ce_floor = 0.15
        p2_distill_floor = 0.40

        if self.training:
            self.current_step += 1
            step = min(self.current_step.item(), self.total_steps)
            
            if step <= self.p1_end:
                # --- PHASE 1: Stable Replication Mode ---
                w_ce = 1.0
                w_distill = 1.0
                w_structural = 0.0
                
            elif step <= self.p2_end:
                # --- PHASE 2: Hand-Off Smooth Mixing Window ---
                p2_progress = (step - self.p1_end) / float(self.p2_end - self.p1_end)
                # Cosine wave maps from 1.0 smoothly down to 0.0
                cos_f = 0.5 * (1.0 + math.cos(math.pi * p2_progress))
                
                w_ce = ce_floor + (1.0 - ce_floor) * cos_f
                w_distill = p2_distill_floor + (1.0 - p2_distill_floor) * cos_f
                w_structural = (1.0 - ce_floor) * (1.0 - cos_f)
                
            else:
                # --- PHASE 3: Autonomous Ranking Refinement ---
                p3_progress = (step - self.p2_end) / float(self.total_steps - self.p2_end)
                cos_f3 = 0.5 * (1.0 + math.cos(math.pi * p3_progress))
                
                w_ce = ce_floor
                w_structural = 1.0 - ce_floor
                # Slowly evaporate the remaining distillation cushion to absolute 0
                w_distill = p2_distill_floor * cos_f3
        else:
            # Inference evaluation assumes perfect Phase 3 criteria
            w_ce = ce_floor
            w_distill = 0.0
            w_structural = 1.0 - ce_floor

        # 4. Mix Active Standard Gradients
        final_loss = (w_ce * ce_loss) + (w_structural * total_structural)

        # 5. Inject Teacher Logit Matching (Active across Phase 1, Phase 2, and Phase 3 fade)
        if self.teacher is not None and self.training and w_distill > 0.0:
            with torch.no_grad():
                if hasattr(self.teacher, "train_forward_fused"):
                    t_temporal = self.teacher.temporal_forward(inp)
                    teacher_logits, _, _, _ = self.teacher.train_forward_fused(t_temporal, tgt)
                else:
                    teacher_logits, _ = self.teacher.train_forward(inp, tgt)

            mask = target_rearranged != -100
            if mask.any():
                flat_student = logits[mask]
                flat_teacher = teacher_logits[mask]

                soft_student = F.log_softmax(flat_student / self.distill_temp, dim=-1)
                soft_teacher = F.softmax(flat_teacher / self.distill_temp, dim=-1)
                distill_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.distill_temp ** 2)
                
                final_loss += w_distill * distill_loss

        return final_loss, logits

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
            expanded_indices = torch.cat([expanded_indices, topk_indices.unsqueeze(-1)], dim=3)

            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)

            flat_topk = topk_indices.view(-1)
            flat_discrete = self.depth_emb(flat_topk)
            flat_discrete = self.depth_emb_ln(flat_discrete)
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
