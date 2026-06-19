import torch

from src.models import SpecialTokens


class FullSoftmax(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.temperature = temperature

    def __call__(self, embs, target, model):
        voc = model.get_embs()  # Normalize everything
        logits = torch.einsum("b l d, v d -> b v l", embs, voc) / self.temperature
        loss = self.criterion(logits, target)
        return loss, logits


class SampledSoftmax(torch.nn.Module):
    def __init__(self, n_items, temperature=1.0, pad_idx=-100): # Note: Paper recommends low temp like 0.05
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.n_items = n_items
        self.temperature = temperature
        self.pad_idx = pad_idx

    def __call__(self, embs, target, model):
        B, L, D = embs.shape
        
        # 1. Identify valid (non-padded) positions
        valid_mask = (target != self.pad_idx).view(-1) 
        
        embs_flat = embs.view(B * L, D)
        target_flat = target.view(B * L)
        target_clean = torch.where(target_flat == self.pad_idx, 0, target_flat)

        # 2. Positive scores (B*L x 1)
        pos_embs = model.get_embs(target_clean)
        target_scores = (embs_flat * pos_embs).sum(dim=-1, keepdim=True)

        # 3. FIX: Independent negative sampling per sequence step (B*L x n_items)
        vocab_size = model.embedding.weight.shape[0]
        samples = torch.randint(0, vocab_size, size=(B * L, self.n_items), device=embs.device)
        
        # Gather negative embeddings and compute scores via batched matrix multiplication
        noise_embs = model.get_embs(samples.view(-1)).view(B * L, self.n_items, D)
        noise_scores = torch.bmm(embs_flat.unsqueeze(1), noise_embs.transpose(1, 2)).squeeze(1)

        # 4. Remove accidentally sampled targets
        reject_samples = target_clean[:, None] == samples
        noise_scores = noise_scores.masked_fill(reject_samples, -1e9)

        # 5. Combine, scale by temperature, and apply masked loss
        logits = torch.cat([target_scores, noise_scores], dim=1) / self.temperature
        labels = torch.zeros(B * L, dtype=torch.long, device=embs.device)
        
        raw_loss = self.cross_entropy(logits, labels)
        
        # FIX: Zero out padding loss and average only over valid elements
        masked_loss = (raw_loss * valid_mask.float()).sum() / torch.clamp(valid_mask.sum(), min=1)

        return masked_loss, logits


class L2Norm(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.clamp(
            torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
            min=self.eps,
        )


class SASRec(torch.nn.Module):
    def __init__(
        self,
        n_layers,
        d_head,
        d_model,
        vocab_size,
        dropout,
        seq_len,
        normalization,  # layer | l2 | None
        criterion=SampledSoftmax(n_items=30_000),
        norm_first=True,
        activation="relu",
        filter_preds=True,
        emb_dropout=None,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.n_heads = d_model // d_head
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.emb_dropout = emb_dropout if emb_dropout is not None else dropout
        self.normalization = normalization
        self.filter_preds = filter_preds

        if not self.filter_preds:
            print("Warning: filter_preds is set to False. Official SASRec uses True.")

        assert normalization in ["layer", "l2", None]

        L = seq_len

        self.embedding = torch.nn.Embedding(
            vocab_size, d_model, padding_idx=SpecialTokens.PAD.value
        )
        self.position_embeddings = torch.nn.Embedding(L, d_model)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((L, L), dtype=torch.bool), diagonal=1).view(L, L),
        )

        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_model * 4,
                dropout=self.dropout,
                activation=activation,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
            # Final norm at the output of the Transformer
            norm={"l2": L2Norm(), "layer": torch.nn.LayerNorm(d_model), None: None}[
                normalization
            ],
        )

        if self.emb_dropout > 0:
            self.edrop = torch.nn.Dropout(self.emb_dropout)
        self.inp_norm = torch.nn.LayerNorm(d_model)

        self.criterion = criterion

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_param_groups(self):
        # Ignore weight decay for embedding
        def _select_no_decay(n):
            return "embedding" in n

        no_decay = [p for n, p in self.named_parameters() if _select_no_decay(n)]
        decay = [p for n, p in self.named_parameters() if not _select_no_decay(n)]

        return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb += self.position_embeddings.weight[None, : x.shape[1], :]
        x_emb = self.inp_norm(x_emb)
        if self.emb_dropout > 0:
            x_emb = self.edrop(x_emb)

        B, L, D = x_emb.shape

        x_enc = self.encoder(
            x_emb,
            mask=self.causal_mask[:L, :L],
            src_key_padding_mask=(x == SpecialTokens.PAD.value),
        )
        # Encoder already includes L2Norm or LayerNorm or Nothing at the end

        return x_enc

    def _l2_norm(self, embs):
        return embs / torch.clamp(
            torch.linalg.norm(embs, ord=2, dim=-1, keepdim=True),
            min=1e-6,
        )

    def get_loss(self, batch):
        query, target = batch["query"], batch["target"]

        embs = self(query)  # B x L x D; Already normalized
        loss, logits = self.criterion(embs, target, model=self)

        return loss, logits

    def get_embs(self, idx=None):
        if idx is None:
            e = self.embedding.weight[:]
        else:
            e = self.embedding(idx)

        # Get the target embeddings for the given indices
        if self.normalization == "layer" or self.normalization is None:
            return e
        elif self.normalization == "l2":
            return self._l2_norm(e)

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode (dropout)."

        query = batch["query"]
        embs = self(query)[:, -1, :]  # Get last embedding, already normalized
        voc = self.get_embs()  # Get everything
        logits = embs @ voc.T  # / temperature doesn't change the order anyway

        if self.filter_preds:
            B = query.shape[0]
            logits[torch.arange(B, device=query.device)[:, None], query] = float("-inf")

        # Get the top n_results
        _, top_idx = logits.topk(n_results, dim=-1)

        return top_idx
