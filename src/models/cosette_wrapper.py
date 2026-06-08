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

        self.eval()
        self.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        """Returns the current device context of this module."""
        return next(self.parameters()).device

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
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes codebook discrete IDs back into continuous reconstruction latents.
        Safely ignores negative boundary values and padding indexes.
        """
        # Ensure correct runtime device alignment
        indices = indices.to(self.device)
        orig_shape = indices.shape

        if len(orig_shape) > 2:
            indices = indices.view(-1, orig_shape[-1])

        # Create a mask tracking valid indices vs padding/out-of-bound indices
        # Assumes codebook values must be >= 0 and less than centroid limits
        valid_mask = indices >= 0
        for i, quantizer in enumerate(self.model.rq.vq_layers):
            # Check maximum dictionary boundaries per quantizer layer
            num_embeddings = (
                quantizer.codebook.shape[0] if hasattr(quantizer, "codebook") else 32000
            )
            valid_mask[:, i] = valid_mask[:, i] & (indices[:, i] < num_embeddings)

        # Broad row-level mask: an item is valid only if all its codebook slots are valid
        valid_rows = valid_mask.all(dim=-1)

        # Clamping step: replace invalid indices with zero placeholder to prevent CUDA assertions
        safe_indices = torch.where(valid_mask, indices, torch.zeros_like(indices))

        x_q = 0
        for i, quantizer in enumerate(self.model.rq.vq_layers):
            layer_indices = safe_indices[:, i]
            x_res = quantizer.get_codebook_entry(layer_indices, shape=None)
            x_q = x_q + x_res

        # Pass combined components through the Decoder MLP
        reconstructed_latents = self.model.decoder(x_q)

        # Zero out the reconstructed vectors for invalid/padded rows
        reconstructed_latents[~valid_rows] = 0.0

        if len(orig_shape) > 2:
            target_shape = list(orig_shape[:-1]) + [self.model.in_dim]
            reconstructed_latents = reconstructed_latents.view(target_shape)

        return reconstructed_latents

    @torch.no_grad()
    def get_codebooks(self) -> torch.Tensor:
        """
        Fetches full structural arrays for all residual layers codebook weights.
        """
        return self.model.rq.get_codebook()
