from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu"):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            activation_func = nn.ReLU()
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def kmeans(samples, num_clusters, num_iters=10):
    """
    Pure PyTorch GPU-accelerated implementation of KMeans.
    Maintains full compatibility with autocast/mixed precision tracking.
    """
    device = samples.device
    orig_dtype = samples.dtype

    # Convert to float32 internally for numerical stability during distance calculations
    x = samples.float()
    num_samples = x.size(0)

    if num_samples < num_clusters:
        raise ValueError(
            f"Number of samples ({num_samples}) must be greater than or equal to num_clusters ({num_clusters})"
        )

    # 1. Randomly initialize centroids from existing samples
    permutation = torch.randperm(num_samples, device=device)
    centroids = x[permutation[:num_clusters]].clone()

    for _ in range(num_iters):
        # 2. Compute pairwise L2 distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * <x, c>
        dists = (
            torch.sum(x**2, dim=1, keepdim=True)
            + torch.sum(centroids**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(x, centroids.t())
        )

        # 3. Assign each sample to its closest centroid
        labels = torch.argmin(dists, dim=-1)

        # 4. Update centroids using vectorized index accumulation
        counts = torch.bincount(labels, minlength=num_clusters).view(-1, 1).float()
        
        sum_centroids = torch.zeros_like(centroids)
        sum_centroids.index_add_(0, labels, x)

        # Handle empty clusters: if count is 0, preserve the old centroid location
        centroids = torch.where(counts > 0, sum_centroids / counts.clamp(min=1), centroids)

    return centroids.to(dtype=orig_dtype)


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block

    # make the matrix sum to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # columns must sum to 1 so that Q is an assignment
    return Q


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_centroids,
        centroids_dim,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.01,
        sk_iters=100,
    ):
        super().__init__()
        self.n_centroids = n_centroids
        self.centroids_dim = centroids_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_centroids, self.centroids_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(
                -1.0 / self.n_centroids, 1.0 / self.n_centroids
            )
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        print("Initializing VQ with GPU-accelerated KMeans...")
        centers = kmeans(
            data,
            self.n_centroids,
            self.kmeans_iters,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        latent = x.view(-1, self.centroids_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate L2 Norm between latent space and embedded weights
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, self.embedding.weight.t())
        )
        
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # Compute losses for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator trick to preserve gradients
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, d


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_centroids_list,
        centroids_dim,
        sk_epsilons,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
    ):
        super().__init__()
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = centroids_dim
        self.num_quantizers = len(n_centroids_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_centroids,
                    centroids_dim,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_centroids, sk_epsilon in zip(n_centroids_list, sk_epsilons)
            ]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_distances = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, mean_losses, all_indices, all_distances


class SigLIPLoss(torch.nn.Module):
    def __init__(
        self,
        tau,
        bias,
        freeze_tau=False,
        freeze_bias=False,
    ):
        super(SigLIPLoss, self).__init__()
        self.tau = torch.nn.Parameter(
            torch.tensor(tau, dtype=torch.float32), requires_grad=not freeze_tau
        )
        self.bias = torch.nn.Parameter(
            torch.tensor(bias, dtype=torch.float32), requires_grad=not freeze_bias
        )

    def _siglip_loss(self, logits, items, timelines):
        with torch.no_grad():
            mask = torch.full(
                (len(items), len(items)),
                -1,
                dtype=torch.float32,
                device=self.tau.device,
            )
            pos = (items[:, None, None] == timelines[None, :, :]).any(axis=2)
            pos = pos.to(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            pos = torch.matmul(pos, pos.T) > 0
            mask[pos] = 1.0

        logsig = F.logsigmoid(mask * logits)
        loss = -(logsig.sum(dim=1) / (mask == 1).sum(dim=1)).mean()
        return loss

    def forward(self, xa, xb, items, timelines):
        xa = F.normalize(xa, dim=-1)
        xb = F.normalize(xb, dim=-1)
        logits = torch.mm(xa, xb.T) * self.tau.exp() + self.bias
        loss = self._siglip_loss(logits, items, timelines)
        return loss


class COSETTE(torch.nn.Module):
    def __init__(
        self,
        embs_block,
        in_dim,
        layers,
        n_centroids_list,
        dropout_prob,
        tau,
        bias,
        freeze_tau,
        freeze_bias,
        loss_weights=None,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
    ):
        super(COSETTE, self).__init__()
        if loss_weights is None:
            loss_weights = {}

        self.in_dim = in_dim
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = layers[-1]

        self.layers = layers
        self.dropout = dropout_prob
        self.loss_weights = loss_weights

        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers
        self.decode_layer_dims = self.encode_layer_dims[::-1]

        self.embeddings = (
            torch.nn.Parameter(embs_block, requires_grad=False)
            if embs_block is not None
            : None
        )

        self.encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout)
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout)

        self.rq = ResidualVectorQuantizer(
            n_centroids_list=self.n_centroids_list,
            centroids_dim=self.centroids_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        if self.loss_weights.get("contrastive", 0) > 0:
            self.siglip = SigLIPLoss(
                tau=tau,
                bias=bias,
                freeze_tau=freeze_tau,
                freeze_bias=freeze_bias,
            )

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def training_loss(self, items, timelines):
        assert self.embeddings is not None, "Embeddings must be provided for training."

        loss = 0.0
        metrics = defaultdict(float)

        # Classic reconstruction LOSS
        if self.loss_weights.get("reconstruction", 0) > 0:
            idxs = torch.randint(
                0,
                self.embeddings.shape[0],
                (len(timelines),),
                device=self.embeddings.device,
            )
            embeddings = F.embedding(idxs, self.embeddings)

            # 1. Forward Pass
            x = self.encoder(embeddings)
            x_q, rq_loss, _, _ = self.rq(x, use_sk=True)
            x_hat = self.decoder(x_q)

            # 2. Base Reconstruction Loss
            recon_loss = F.mse_loss(x_hat, embeddings, reduction="mean")
            loss += recon_loss * self.loss_weights["reconstruction"]
            loss += rq_loss * self.loss_weights["quantization"]

            metrics["reconstruction_loss"] = recon_loss.item()
            metrics["quantization_loss"] = rq_loss.item()

            if self.loss_weights.get("reconstruction_l1_loss", 0) > 0:
                recon_l1_loss = F.l1_loss(x_hat, embeddings, reduction="mean")
                loss += recon_l1_loss * self.loss_weights.get("reconstruction_l1_loss", 0)
                metrics["reconstruction_l1_loss"] = recon_l1_loss.item()

            # 3. Latent Consistency Loss
            if self.loss_weights.get("latent_consistency", 0) > 0:
                x_tilde = self.encoder(x_hat)
                latent_cons_loss = F.mse_loss(x_tilde, x_q.detach(), reduction="mean")

                loss += latent_cons_loss * self.loss_weights["latent_consistency"]
                metrics["latent_consistency_loss"] = latent_cons_loss.item()

                if self.loss_weights.get("latent_consistency_l1_loss", 0) > 0:
                    latent_cons_l1_loss = F.l1_loss(x_tilde, x_q.detach(), reduction="mean")
                    loss += latent_cons_l1_loss * self.loss_weights.get("latent_consistency_l1_loss", 0)
                    metrics["latent_consistency_l1_loss"] = latent_cons_l1_loss.item()

        # 4. Contrastive training
        if self.loss_weights.get("contrastive", 0) > 0:
            embeddings = F.embedding(items, self.embeddings)
            x = self.encoder(embeddings)
            x_q, sig_rq_loss, _, _ = self.rq(x, use_sk=True)

            contrastive_loss = self.siglip(x_q, x_q, items, timelines)

            loss += sig_rq_loss * self.loss_weights["quantization"]
            loss += contrastive_loss * self.loss_weights["contrastive"]

            metrics["contrastive_loss"] = contrastive_loss.item()
            metrics["quantization_loss"] += sig_rq_loss.item()
            metrics["tau"] = self.siglip.tau.item()
            metrics["bias"] = self.siglip.bias.item()
            metrics["n_items_in_batch"] = len(items)

        return loss, metrics

    @torch.no_grad()
    def get_indices(self, embeddings=None, use_sk=False):
        if embeddings is None:
            embeddings = self.embeddings

        x_e = self.encoder(embeddings)
        _, _, indices, distances = self.rq(x_e, use_sk=use_sk)
        return indices, distances
