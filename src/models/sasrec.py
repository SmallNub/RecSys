import torch

from src.models import SpecialTokens


class FullSoftmax(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.temperature = temperature

    def __call__(self, embs, target, model):
        voc = model.get_embs()
        logits = torch.einsum("b l d, v d -> b v l", embs, voc) / self.temperature
        loss = self.criterion(logits, target)
        return loss, logits


class SampledSoftmax(torch.nn.Module):
    def __init__(self, n_items, temperature=1.0):
        super().__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.n_items = n_items
        self.emb_table = None
        self.temperature = temperature

    def __call__(self, embs, target, model):
        B, L, D = embs.shape
        embs = embs.view(B * L, D)
        target = target.view(B * L)

        target_scores = (embs * model.get_embs(target)).sum(dim=-1)  # B*L
        samples = torch.randint(
            low=0,
            high=model.embedding.weight.shape[0],
            size=(self.n_items,),
            device=embs.device,
        )
        noise_scores = embs @ model.get_embs(samples).T

        reject_samples = target[:, None] == samples[None, :]
        noise_scores -= 1e6 * reject_samples.float()

        logits = (
            torch.cat([target_scores[:, None], noise_scores], dim=1) / self.temperature
        )
        loss = self.cross_entropy(logits, torch.zeros_like(target))

        return loss, logits


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
        normalization,
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

        return x_enc

    def _l2_norm(self, embs):
        return embs / torch.clamp(
            torch.linalg.norm(embs, ord=2, dim=-1, keepdim=True),
            min=1e-6,
        )

    def get_loss(self, batch):
        query, target = batch["query"], batch["target"]

        embs = self(query)
        loss, logits = self.criterion(embs, target, model=self)

        return loss, logits

    def get_embs(self, idx=None):
        if idx is None:
            e = self.embedding.weight[:]
        else:
            e = self.embedding(idx)

        if self.normalization == "layer" or self.normalization is None:
            return e
        elif self.normalization == "l2":
            return self._l2_norm(e)

    def search(self, batch, n_results):
        assert self.training is False, "Not in evaluation mode (dropout)."

        query = batch["query"]
        embs = self(query)[:, -1, :]
        voc = self.get_embs()
        logits = embs @ voc.T

        if self.filter_preds:
            B = query.shape[0]
            logits[torch.arange(B, device=query.device)[:, None], query] = float("-inf")

        _, top_idx = logits.topk(n_results, dim=-1)

        return top_idx
