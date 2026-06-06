import os
import torch
from cosette import COSETTE


class CosetteWrapper:
    def __init__(self, model_path: str, device: str = None):
        """
        Initializes the COSETTE model wrapper for simple embedding <-> ID conversions.

        :param model_path: Path to the saved semantic model checkpoint (.pt / .pth)
        :param device: 'cuda' or 'cpu'. Automatically detects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading COSETTE checkpoint onto {self.device}...")
        self.model = self._load_model_from_checkpoint(model_path)
        self.model.eval()

    def _load_model_from_checkpoint(self, model_path: str) -> torch.nn.Module:
        """Helper to safely unpack config parameters and load state weights."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {model_path}")

        ckpt = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )

        model_cfg = ckpt.get("config")
        state_dict = ckpt.get("state_dict")
        state_dict.pop("embeddings")

        # Build layer-dimensions from saved configuration templates
        # Default fallback values handled safely if config format slightly shifts
        layers = model_cfg.model.layers
        n_centroids_list = model_cfg.centroids.n_centroids_list

        model = COSETTE(
            embs_block=None,  # We pass continuous arrays explicitly at inference
            in_dim=768,  # Or derived via config fields
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

        model.load_state_dict(state_dict)
        return model.to(self.device)

    @torch.no_grad()
    def encode(self, latents: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        """
        Encodes high-dimensional continuous embeddings to multi-codebook indices (IDs).

        :param latents: torch.Tensor of shape (Batch, Input_Dim)
        :param use_sk: True to activate Sinkhorn optimization constraint clustering
        :return: Discrete IDs matrix of shape (Batch, Number_of_Quantizers)
        """
        latents = latents.to(self.device).float()

        # 1. Project through the multi-layer perceptron encoder
        x_e = self.model.encoder(latents)

        # 2. Extract indices via Residual Vector Quantizer layer
        _, _, indices, _ = self.model.rq(x_e, use_sk=use_sk)

        return indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes codebook discrete IDs back into continuous reconstruction latents.

        :param indices: torch.Tensor of shape (Batch, Number_of_Quantizers)
        :return: Reconstructed latents tensor of shape (Batch, Input_Dim)
        """
        indices = indices.to(self.device).long()
        shape = indices.shape

        # Unpack layers recursively from the Residual Quantizer sequence
        x_q = 0
        for i, quantizer in enumerate(self.model.rq.vq_layers):
            # Extract specific quantization column entries
            layer_indices = indices[..., i]
            # Match codebook weight structures
            x_res = quantizer.get_codebook_entry(layer_indices, shape=None)
            x_q = x_q + x_res

        # Reshape to expected linear sizing if indices nested
        if len(shape) > 2:
            x_q = x_q.view(-1, self.model.centroids_dim)

        # Pass combined quantization residuals through the Decoder MLP
        reconstructed_latents = self.model.decoder(x_q)
        return reconstructed_latents

    def get_codebooks(self) -> torch.Tensor:
        """
        Fetches full structural arrays for all residual layers codebook weights.

        :return: Vector stack of codebooks tensor shaped (Num_Quantizers, Centroids_per_layer, Code_Dim)
        """
        return self.model.rq.get_codebook()


if __name__ == "__main__":
    # 1. Initialize model wrapper
    MODEL_PATH = "outputs/checkpoints/cosette/20260606_151039/last_model.pth"
    engine = CosetteWrapper(model_path=MODEL_PATH)

    # 2. Mock a continuous input latent vector (Batch Size = 4, Dim = 128)
    mock_latents = torch.randn(4, 768)

    # 3. Encode Latent -> Semantic ID
    semantic_ids = engine.encode(mock_latents, use_sk=False)
    print("Encoded IDs Shape:", semantic_ids.shape) 
    print("Example Discrete Code IDs:\n", semantic_ids)

    # 4. Decode Semantic ID -> Latent Vector Space
    reconstructed_embeddings = engine.decode(semantic_ids)
    print("Decoded Latents Shape:", reconstructed_embeddings.shape)

    # 5. Access Underlyling Codebooks
    codebooks = engine.get_codebooks()
    print("Stacked Codebooks Shape:", codebooks.shape)
