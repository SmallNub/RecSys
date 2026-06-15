import os
import torch
import torch.nn as nn
from src.models.cosette import COSETTE


class CosetteWrapper(nn.Module):
    def __init__(self, model_path: str):
        """
        Initializes the frozen COSETTE model wrapper for inference-only
        embedding <-> ID conversions.

        :param model_path: Path to the saved semantic model checkpoint (.pt / .pth)
        """
        super().__init__()
        print("Loading frozen COSETTE checkpoint...")
        self.model = self._load_model_from_checkpoint(model_path)

        # Buffer tracking device migration dynamically within PyTorch Lightning loops
        self.register_buffer("_device_marker", torch.empty(0))

        self.eval()
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Returns the current device context of this module."""
        return self._device_marker.device

    def _load_model_from_checkpoint(self, model_path: str) -> torch.nn.Module:
        """Helper to safely unpack config parameters and load state weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {model_path}")

        ckpt = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )

        model_cfg = ckpt.get("config")
        state_dict = ckpt.get("state_dict")
        state_dict.pop("embeddings", None)  # Pre-emptively remove training block

        layers = model_cfg.model.layers
        n_centroids_list = model_cfg.centroids.n_centroids_list
        in_dim = getattr(model_cfg.model, "in_dim", 768)

        model = COSETTE(
            embs_block=None,
            in_dim=in_dim,
            layers=layers,
            n_centroids_list=n_centroids_list,
            dropout_prob=getattr(model_cfg.optim, "dropout_prob", 0.0),
            loss_weights={"quantization": 1, "reconstruction": 1, "contrastive": 1},
            tau=getattr(model_cfg.loss, "tau", 1.0),
            bias=getattr(model_cfg.loss, "bias", 0.0),
            freeze_tau=True,
            freeze_bias=True,
            kmeans_init=getattr(model_cfg.centroids, "kmeans_init", False),
            kmeans_iters=getattr(model_cfg.centroids, "kmeans_iters", 10),
            sk_epsilons=getattr(model_cfg.centroids, "sk_epsilons", None),
            sk_iters=getattr(model_cfg.centroids, "sk_iters", 100),
        )

        model.load_state_dict(state_dict, strict=False)
        return model

    @torch.no_grad()
    def encode(self, latents: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        """
        Encodes high-dimensional continuous embeddings to multi-codebook indices (IDs).
        """
        latents = latents.to(self.device)
        x_e = self.model.encoder(latents)
        _, _, indices, _ = self.model.rq(x_e, use_sk=use_sk)
        return indices

    @torch.no_grad()
    def get_codebook_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Maps a sequence of IDs to their continuous codebook vectors without summing
        or passing through the decoder MLP.
        Returns tensor of shape (..., K, centroids_dim)
        """
        indices = indices.to(self.device)
        orig_shape = list(indices.shape)

        if len(orig_shape) > 2:
            indices = indices.view(-1, orig_shape[-1])

        num_present_quantizers = indices.shape[-1]
        valid_mask = indices >= 0

        safe_indices = torch.where(valid_mask, indices, torch.zeros_like(indices))
        embs = []

        for i in range(num_present_quantizers):
            quantizer = self.model.rq.vq_layers[i]
            num_embeddings = getattr(quantizer, "n_centroids", 32000)

            valid_idx = safe_indices[:, i] < num_embeddings
            valid_mask[:, i] = valid_mask[:, i] & valid_idx

            safe_idx = torch.where(valid_idx, safe_indices[:, i], torch.zeros_like(safe_indices[:, i]))
            x_res = quantizer.get_codebook_entry(safe_idx, shape=None)
            embs.append(x_res)

        embs = torch.stack(embs, dim=1)  # B x K x centroids_dim
        embs = embs * valid_mask.unsqueeze(-1).to(embs.dtype)

        if len(orig_shape) > 2:
            target_shape = orig_shape[:-1] + [num_present_quantizers, self.model.centroids_dim]
            embs = embs.view(target_shape)

        return embs

    @torch.no_grad()
    def get_codebook_entry(self, indices: torch.Tensor, quantizer_index: int = 0) -> torch.Tensor:
        """
        Fetches continuous embeddings specifically for one quantizer level.
        Returns tensor of shape (..., centroids_dim)
        """
        indices = indices.to(self.device)
        quantizer = self.model.rq.vq_layers[quantizer_index]
        num_embeddings = getattr(quantizer, "n_centroids", 32000)

        valid_mask = indices >= 0
        valid_idx = indices < num_embeddings
        valid_mask = valid_mask & valid_idx

        safe_indices = torch.where(valid_mask, indices, torch.zeros_like(indices))
        embs = quantizer.get_codebook_entry(safe_indices, shape=None)

        embs = embs * valid_mask.unsqueeze(-1).to(embs.dtype)
        return embs

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts discrete IDs back to their continuous, unquantized representation
        by summing vectors across all active quantizers, skipping the MLP decoder.
        """
        codebook_embs = self.get_codebook_embeddings(indices)

        # Sum vectors directly across all active quantizers (unquantized representation)
        x_q = codebook_embs.sum(dim=-2)

        # Multiplicative masking instead of mutating tensor assignments
        valid_rows = (indices >= 0).all(dim=-1, keepdim=True)
        unquantized_latents = x_q * valid_rows.to(x_q.dtype)

        return unquantized_latents

    @torch.no_grad()
    def get_codebooks(self) -> torch.Tensor:
        """
        Fetches full structural arrays for all residual layers codebook weights.
        """
        return self.model.rq.get_codebook()
