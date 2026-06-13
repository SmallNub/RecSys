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

        assert self.depth_cfg.vocab_size == self.temporal_cfg.vocab_size, "Vocab size mismatch"

        if self.temporal_cfg.emb_dropout is None:
            self.temporal_cfg.emb_dropout = self.temporal_cfg.dropout
        if self.depth_cfg.emb_dropout is None:
            self.depth_cfg.emb_dropout = self.depth_cfg.dropout

        # Clean Discrete Token Embeddings
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

        # Baseline Linear Projections for Continuous Feature Fusion
        self.temp_proj = torch.nn.Linear(self.cosette.model.in_dim, self.temporal_cfg.d_model)
        self.depth_proj = torch.nn.Linear(self.cosette.model.centroids_dim, self.depth_cfg.d_model)

        self.temp_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.temporal_cfg.seq_len, self.temporal_cfg.d_model))
        )
        self.depth_pos_emb = torch.nn.Parameter(
            torch.randn((1, self.depth_cfg.seq_len, self.depth_cfg.d_model))
        )

        self.temp_dropout = torch.nn.Dropout(self.temporal_cfg.emb_dropout)
        self.depth_dropout = torch.nn.Dropout(self.depth_cfg.emb_dropout)

        # Native PyTorch Encoders
        self.temp_tf = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.temporal_cfg.d_model,
                nhead=self.temporal_cfg.d_model // self.temporal_cfg.d_head,
                dim_feedforward=self.temporal_cfg.d_model * 4,
                dropout=self.temporal_cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.temporal_cfg.n_layers,
            norm=torch.nn.LayerNorm(self.temporal_cfg.d_model),
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

        # VECTORIZATION FIX: Pre-register the static temporal causal mask
        self.register_buffer(
            "temp_causal_mask",
            torch.triu(torch.ones((self.temporal_cfg.seq_len, self.temporal_cfg.seq_len), dtype=torch.bool), diagonal=1),
        )

        # VECTORIZATION FIX: Pre-compile all possible depth cross-attention masks to GPU memory
        # Since depth sequence length grows step-by-step during search, we store each matrix size directly.
        for n in range(1, self.depth_cfg.seq_len + 2):
            mask = torch.ones((2 * n, 2 * n), dtype=torch.bool)
            causal_tril = torch.tril(torch.ones((n, n), dtype=torch.bool))
            mask[:n, :n] = ~causal_tril
            mask[:n, n:] = True
            mask[n:, :n] = ~causal_tril
            mask[n:, n:] = ~causal_tril
            self.register_buffer(f"depth_static_mask_step_{n}", mask)

        self.mid_proj = torch.nn.Linear(self.temporal_cfg.d_model, self.depth_cfg.d_model)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def get_param_groups(self):
        def _select_no_decay(n):
            return "temp_emb" in n or "depth_emb" in n
        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]
        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def temporal_forward(self, input):
        B, L, K = input.shape

        discrete_embs = self.temp_emb(input).sum(dim=-2)
        if self.training:
            mask = (torch.rand(B, L, 1, device=input.device) > 0.2).to(discrete_embs.dtype)
            discrete_embs = discrete_embs * mask
        
        decoded_features = self.cosette.decode(input)
        continuous_embs = self.temp_proj(decoded_features)

        # Element-wise addition keeps sequence footprint at exactly L
        x = discrete_embs + continuous_embs + self.temp_pos_emb[:, :L, :]
        x = self.temp_dropout(x)
        
        padding_mask = input[:, :, 0] == SpecialTokens.PAD.value
        mask = self.temp_causal_mask[:L, :L]

        temporal_out = self.temp_tf(
            src=x,
            mask=mask,
            src_key_padding_mask=padding_mask
        )
        return temporal_out

    def depth_forward(self, tgt, memory):
        N = tgt.shape[1]  # Because memory and tgt are perfectly aligned symmetrical blocks

        unified_depth = torch.cat([memory, tgt], dim=1)

        # Zero-allocation mask acquisition via static lookup table
        depth_mask = getattr(self, f"depth_static_mask_step_{N}")

        depth_preds = self.depth_tf(unified_depth, mask=depth_mask)
        depth_preds = depth_preds[:, N:, :]  # Extract target slice predictions

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
        loss = self.criterion(logits, m_target)
        return loss, logits

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
