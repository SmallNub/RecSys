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
    neftune_alpha: float = 5.0  
    loss_lambda: float = 0.1     
    aux_lambda: float = 0.25     # Scaling coefficient for the dense continuous target regression


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff)
        self.w2 = torch.nn.Linear(d_model, d_ff)
        self.w3 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SwiGLUTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SwiGLU(d_model, int(d_model * 8 / 3))
        
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x1 = self.norm1(src)
        attn_out, _ = self.self_attn(
            query=x1, key=x1, value=x1, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        src = src + self.dropout1(attn_out)
        src = src + self.dropout2(self.ffn(self.norm2(src)))
        return src


class EmbeddingAdapter(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim)
        self.norm = torch.nn.LayerNorm(out_dim)
        self.ffn = SwiGLU(out_dim, int(out_dim * 8 / 3))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.proj.weight, a=0.436)

    def forward(self, x):
        x = self.proj(x)
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        return x + residual


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

        assert self.depth_cfg.vocab_size == self.temporal_cfg.vocab_size, "Vocab size mismatch"

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

        self.temp_proj = EmbeddingAdapter(
            in_dim=self.cosette.model.in_dim, 
            out_dim=self.temporal_cfg.d_model,
            dropout=self.temporal_cfg.dropout
        )
        self.depth_proj = EmbeddingAdapter(
            in_dim=self.cosette.model.centroids_dim, 
            out_dim=self.depth_cfg.d_model,
            dropout=self.depth_cfg.dropout
        )

        self.temp_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.temporal_cfg.seq_len, self.temporal_cfg.d_model))
        )
        self.depth_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.depth_cfg.seq_len, self.depth_cfg.d_model))
        )

        self.temp_dropout = torch.nn.Dropout(self.temporal_cfg.emb_dropout)
        self.depth_dropout = torch.nn.Dropout(self.depth_cfg.emb_dropout)

        self.temp_tf_layers = torch.nn.ModuleList([
            SwiGLUTransformerEncoderLayer(
                d_model=self.temporal_cfg.d_model,
                nhead=self.temporal_cfg.d_model // self.temporal_cfg.d_head,
                dropout=self.temporal_cfg.dropout
            ) for _ in range(self.temporal_cfg.n_layers)
        ])
        self.temp_final_norm = torch.nn.LayerNorm(self.temporal_cfg.d_model)

        # INFERENCE-FREE AUXILIARY DENSE SIGNAL HEAD
        # Predicts continuous states at t+1; bypassed entirely during search loops
        self.aux_dense_norm = torch.nn.LayerNorm(self.temporal_cfg.d_model)
        self.aux_dense_head = torch.nn.Sequential(
            torch.nn.Linear(self.temporal_cfg.d_model, self.temporal_cfg.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.temporal_cfg.d_model, self.temporal_cfg.d_model)
        )

        self.depth_tf = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.depth_cfg.d_model,
                nhead=self.depth_cfg.d_model // self.depth_cfg.d_head,
                dim_feedforward=self.depth_cfg.d_model * 4,
                dropout=self.depth_cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.depth_cfg.n_layers,
            norm=torch.nn.LayerNorm(self.depth_cfg.d_model),
        )

        self.register_buffer(
            "standard_causal_mask",
            torch.triu(torch.ones((self.depth_cfg.seq_len, self.depth_cfg.seq_len), dtype=torch.bool), diagonal=1),
        )

        self.mid_proj = torch.nn.Linear(self.temporal_cfg.d_model, self.depth_cfg.d_model)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n
        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def get_vectorized_multimodal_mask(self, L, device):
        total_len = 2 * L
        mask = torch.ones((total_len, total_len), dtype=torch.bool, device=device)
        causal_tril = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
        
        mask[:L, :L] = ~causal_tril        
        mask[:L, L:] = True                
        mask[L:, :L] = ~causal_tril        
        mask[L:, L:] = ~causal_tril        
        return mask

    def apply_neftune(self, embeddings, alpha):
        if self.training and alpha > 0:
            dims = embeddings.size(-1)
            scale = alpha / (dims ** 0.5)
            noise = (torch.rand_like(embeddings) * 2.0 - 1.0) * scale
            return embeddings + noise
        return embeddings

    def compute_contrastive_loss(self, continuous_features, discrete_features, temperature=0.07):
        c_proj = F.normalize(continuous_features.mean(dim=1), dim=-1)
        d_proj = F.normalize(discrete_features.mean(dim=1), dim=-1)

        similarity_matrix = torch.matmul(c_proj, d_proj.T) / temperature
        labels = torch.arange(continuous_features.size(0), device=continuous_features.device)

        loss_c = F.cross_entropy(similarity_matrix, labels)
        loss_d = F.cross_entropy(similarity_matrix.T, labels)
        return (loss_c + loss_d) / 2.0

    def temporal_forward(self, input):
        B, L, K = input.shape

        discrete_embs = self.temp_emb(input).sum(dim=-2)
        if self.training:
            mask = (torch.rand(B, L, 1, device=input.device) > 0.2).to(discrete_embs.dtype)
            discrete_embs = discrete_embs * mask
        
        decoded_features = self.cosette.decode(input)
        continuous_embs = self.temp_proj(decoded_features)

        # Retain references for cross-modal tracking
        self._aux_loss_continuous = continuous_embs
        self._aux_loss_discrete = discrete_embs

        continuous_embs = self.apply_neftune(continuous_embs, self.temporal_cfg.neftune_alpha)
        discrete_embs = self.apply_neftune(discrete_embs, self.temporal_cfg.neftune_alpha)

        tgt = discrete_embs + self.temp_pos_emb[:, :L, :]
        tgt = self.temp_dropout(tgt)

        unified_sequence = torch.cat([continuous_embs, tgt], dim=1)
        mm_mask = self.get_vectorized_multimodal_mask(L, input.device)
        
        padding_mask = input[:, :, 0] == SpecialTokens.PAD.value
        extended_padding_mask = torch.cat([
            torch.zeros((B, L), dtype=torch.bool, device=input.device),
            padding_mask
        ], dim=1)

        x = unified_sequence
        for layer in self.temp_tf_layers:
            x = layer(x, src_mask=mm_mask, src_key_padding_mask=extended_padding_mask)
        unified_out = self.temp_final_norm(x)

        # If training, calculate the auxiliary continuous look-ahead projection
        if self.training:
            temporal_tokens = unified_out[:, L:, :]
            # Step t predicts continuous feature space at step t+1
            preds = self.aux_dense_head(self.aux_dense_norm(temporal_tokens[:, :-1, :]))
            targets = self._aux_loss_continuous[:, 1:, :]
            
            mse_loss = F.mse_loss(preds, targets)
            cosine_loss = 1.0 - F.cosine_similarity(preds, targets, dim=-1).mean()
            self._dense_aux_loss = mse_loss + 0.5 * cosine_loss
        else:
            self._dense_aux_loss = 0.0

        return unified_out[:, L:, :]

    def depth_forward(self, tgt, memory):
        K = tgt.shape[1]
        tgt = tgt + self.depth_pos_emb[:, :K, :]
        tgt = self.depth_dropout(tgt)

        unified_depth = torch.cat([memory, tgt], dim=1)
        B, M_len, _ = memory.shape
        total_depth_len = M_len + K
        
        depth_mask = torch.ones((total_depth_len, total_depth_len), dtype=torch.bool, device=tgt.device)
        causal_tril = torch.tril(torch.ones((max(M_len, K), max(M_len, K)), dtype=torch.bool, device=tgt.device))
        
        depth_mask[:M_len, :M_len] = ~causal_tril[:M_len, :M_len]
        depth_mask[:M_len, M_len:] = True
        depth_mask[M_len:, :M_len] = ~causal_tril[:K, :M_len]
        depth_mask[M_len:, M_len:] = ~causal_tril[:K, :K]

        depth_preds = self.depth_tf(unified_depth, mask=depth_mask)
        depth_preds = depth_preds[:, M_len:, :] 

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

        dec_embs_orig = self.depth_emb(target[:, :-1])
        K_minus_1 = dec_embs_orig.shape[1]
        
        tgt = torch.cat([mid_tokens, dec_embs_orig], dim=1)

        if K_minus_1 > 0:
            raw_res_list = []
            for i in range(K_minus_1):
                quantizer = self.cosette.model.rq.vq_layers[i]
                layer_indices = target[:, i]

                num_embeddings = getattr(quantizer, "n_centroids", 32000)
                valid_mask = (layer_indices >= 0) & (layer_indices < num_embeddings)
                safe_indices = torch.where(valid_mask, layer_indices, torch.zeros_like(layer_indices))

                x_res = quantizer.get_codebook_entry(safe_indices, shape=None)
                v_mask = valid_mask.unsqueeze(-1).to(x_res.dtype)
                raw_res_list.append(x_res * v_mask)

            raw_tensor = torch.stack(raw_res_list, dim=1)
            continuous_tensor = self.depth_proj(raw_tensor)
            
            dummy_zero = torch.zeros((continuous_tensor.shape[0], 1, continuous_tensor.shape[2]), 
                                     device=continuous_tensor.device, dtype=continuous_tensor.dtype)
            memory = torch.cat([dummy_zero, continuous_tensor], dim=1)
        else:
            memory = torch.zeros((tgt.shape[0], 1, tgt.shape[2]), device=tgt.device, dtype=tgt.dtype)

        depth_logits = self.depth_forward(tgt, memory)  
        return depth_logits, target

    def get_loss(self, batch):
        input, target = batch["input"], batch["target"]
        logits, m_target = self.train_forward(input, target)
        logits = rearrange(logits, "B k v -> B v k")
        
        # 1. Main Autoregressive Task Loss
        ce_loss = self.criterion(logits, m_target)
        
        # 2. Batch-Level Structural Alignment Loss
        align_loss = self.compute_contrastive_loss(self._aux_loss_continuous, self._aux_loss_discrete)
        
        # 3. Step-by-Step Dense Continuous Prediction Target Loss
        dense_loss = self._dense_aux_loss
        
        # Aggregated Loss Execution
        total_loss = (
            ce_loss 
            + (self.temporal_cfg.loss_lambda * align_loss) 
            + (self.temporal_cfg.aux_lambda * dense_loss)
        )
        return total_loss, logits

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode."
        input = batch["input"]
        L = batch["target"].shape[-1]

        if self.filter_preds:
            keep_final = n_results
            n_results += self.temporal_cfg.seq_len

        temporal_tokens = self.temporal_forward(input)  
        mid_tokens = self.mid_proj(temporal_tokens)[:, -1, :]  

        B, b, D = input.shape[0], n_results, self.depth_cfg.d_model

        sequences = mid_tokens[:, None, :]  
        memory = torch.zeros((B, 1, D), device=sequences.device, dtype=sequences.dtype) 
        
        depth_logits = self.depth_forward(sequences, memory)  
        log_probs = F.log_softmax(depth_logits[:, -1, :], dim=-1)  

        topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)  

        indices = topk_indices.unsqueeze(2)  
        scores = topk_log_probs  
        sequences = sequences.unsqueeze(1).repeat(1, b, 1, 1)  
        memory = memory.unsqueeze(1).repeat(1, b, 1, 1)        

        discrete_new = self.depth_emb(topk_indices)

        quantizer = self.cosette.model.rq.vq_layers[0]
        num_embeddings = getattr(quantizer, "n_centroids", 32000)
        valid_mask = (topk_indices >= 0) & (topk_indices < num_embeddings)
        safe_indices = torch.where(valid_mask, topk_indices, torch.zeros_like(topk_indices))

        raw_new_tokens = quantizer.get_codebook_entry(safe_indices, shape=None)
        continuous_new = self.depth_proj(raw_new_tokens)
        v_mask = valid_mask.unsqueeze(-1).to(continuous_new.dtype)
        continuous_new = continuous_new * v_mask 

        sequences = torch.concat([sequences, discrete_new.unsqueeze(2)], dim=2) 
        memory = torch.concat([memory, continuous_new.unsqueeze(2)], dim=2)       

        arranged = torch.arange(B, device=sequences.device).view(-1, 1)

        for i in range(2, L + 1):
            last_logits = self.depth_forward(
                tgt=sequences.view(B * b, i, D),
                memory=memory.view(B * b, i, D)
            )[:, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)  

            topk_log_probs, topk_indices = torch.topk(log_probs, b, dim=-1)

            topk_log_probs = topk_log_probs.view(B, b, b)
            topk_indices = topk_indices.view(B, b, b)

            expanded_indices = indices.unsqueeze(2).repeat(1, 1, b, 1)
            expanded_indices = torch.cat([expanded_indices, topk_indices.unsqueeze(-1)], dim=3)

            expanded_sequences = sequences.unsqueeze(2).repeat(1, 1, b, 1, 1)
            expanded_memory = memory.unsqueeze(2).repeat(1, 1, b, 1, 1)
            
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

            next_tokens = flat_discrete.view(B, b, b, 1, D)
            next_mem = flat_projected.view(B, b, b, 1, D)
            
            expanded_sequences = torch.cat([expanded_sequences, next_tokens], dim=3)
            expanded_memory = torch.cat([expanded_memory, next_mem], dim=3)

            expanded_scores = scores.unsqueeze(2) + topk_log_probs
            expanded_scores = expanded_scores.view(B, -1)
            
            expanded_sequences = expanded_sequences.view(B, -1, expanded_sequences.size(-2), D)
            expanded_memory = expanded_memory.view(B, -1, expanded_memory.size(-2), D)
            expanded_indices = expanded_indices.view(B, -1, expanded_indices.size(-1))

            topk_scores, topk_indices = torch.topk(expanded_scores, b, dim=-1)

            sequences = expanded_sequences[arranged, topk_indices]
            memory = expanded_memory[arranged, topk_indices]
            indices = expanded_indices[arranged, topk_indices]
            scores = topk_scores  

        if self.filter_preds:
            is_in_query = indices[:, :, None, :] == input[:, None, :, :]
            is_in_query = is_in_query.all(dim=-1).any(dim=-1)  
            scores[is_in_query] = -torch.inf
            topk_scores, topk_indices = torch.topk(scores, keep_final, dim=-1)
            indices = indices[arranged, topk_indices]

        return indices
