from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn


# ==========================================
# 1. CORE NETWORK BUILDING BLOCKS
# ==========================================

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ln(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.drop(x)
        return x + residual


class MLPBlock(nn.Module):
    def __init__(self, layers, dropout=0.0, apply_final_ln=False):
        super().__init__()
        self.layers = layers
        self.blocks = nn.ModuleList()

        for input_size, output_size in zip(self.layers[:-1], self.layers[1:]):
            self.blocks.append(LinearBlock(input_size, output_size, dropout))

        self.final_ln = nn.LayerNorm(layers[-1]) if apply_final_ln else nn.Identity()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final_ln(x)


# ==========================================
# 2. HYPERSPHERICAL QUANTIZATION UTILITIES
# ==========================================

def spherical_kmeans(samples, num_clusters, num_iters=10):
    device = samples.device
    orig_dtype = samples.dtype

    # Project inputs onto the unit hypersphere
    x = F.normalize(samples.float(), p=2, dim=-1)
    num_samples = x.size(0)

    if num_samples < num_clusters:
        repeats = (num_clusters // num_samples) + 1
        x = x.repeat(repeats, 1)
        num_samples = x.size(0)

    permutation = torch.randperm(num_samples, device=device)
    centroids = x[permutation[:num_clusters]].clone()

    for _ in range(num_iters):
        # On a unit sphere, squared Euclidean distance simplifies to: 2 - 2 * cos_sim
        dist_matrix = 2.0 - 2.0 * torch.matmul(x, centroids.t())

        labels = torch.argmin(dist_matrix, dim=-1)
        
        sum_centroids = torch.zeros_like(centroids)
        sum_centroids.index_add_(0, labels, x)

        # Map centroid averages back to the surface of the hypersphere
        centroids = F.normalize(sum_centroids, p=2, dim=-1)

    return centroids.to(dtype=orig_dtype)


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    B, K = distances.shape

    shifted_dists = distances - distances.min()
    Q = torch.exp(-shifted_dists / epsilon)
    Q /= (Q.sum() + 1e-12)

    for _ in range(sinkhorn_iterations):
        Q /= (torch.sum(Q, dim=1, keepdim=True) * B + 1e-12)
        Q /= (torch.sum(Q, dim=0, keepdim=True) * K + 1e-12)

    return Q * B


# ==========================================
# 3. QUANTIZER MODULES
# ==========================================

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_centroids,
        centroids_dim,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.03,  # Tuned down for bounded hyperspherical distances
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
            # Initialized uniformly; dynamically normalized during forward pass
            self.embedding.weight.data.uniform_(-1.0, 1.0)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        # Always expose dynamically normalized unit vectors
        return F.normalize(self.embedding.weight, p=2, dim=-1)

    def get_codebook_entry(self, indices, shape=None):
        weight = self.get_codebook()
        z_q = F.embedding(indices, weight)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        print("Initializing Spherical VQ with GPU-accelerated Spherical KMeans...")
        centers = spherical_kmeans(data, self.n_centroids, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = (max_distance - middle).clamp(min=1e-5)
        return (distances - middle) / amplitude

    def forward(self, x, use_sk=True):
        # 1. Map input representations to the unit hypersphere
        latent = F.normalize(x.view(-1, self.centroids_dim), p=2, dim=-1)

        if not self.initted and self.training:
            self.init_emb(latent)

        # 2. Grab normalized codebook vectors
        weight = self.get_codebook()

        # 3. Compute spherical distance matrix
        d = 2.0 - 2.0 * torch.matmul(latent, weight.t())

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_norm = self.center_distance_for_constraint(d)
            Q = sinkhorn_algorithm(d_norm, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # 4. Extract quantized values from normalized weight space
        x_q = F.embedding(indices, weight).view(x.shape)
        latent_shaped = latent.view(x.shape)

        # 5. Quantizer Losses
        commitment_loss = F.mse_loss(x_q.detach(), latent_shaped)
        codebook_loss = F.mse_loss(x_q, latent_shaped.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # 6. Straight-Through Estimator back to normalized latent
        x_q = latent_shaped + (x_q - latent_shaped).detach()
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


# ==========================================
# 4. LOSS FUNCTIONS & COSETTE MAIN WRAPPER
# ==========================================

class SigLIPLoss(torch.nn.Module):
    def __init__(self, tau, bias, freeze_tau=False, freeze_bias=False):
        super(SigLIPLoss, self).__init__()
        self.tau = torch.nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=not freeze_tau)
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=not freeze_bias)

    def _siglip_loss(self, logits, items, timelines):
        with torch.no_grad():
            mask = torch.full((len(items), len(items)), -1.0, dtype=logits.dtype, device=self.tau.device)
            pos = (items[:, None, None] == timelines[None, :, :]).any(dim=2)
            pos = pos.to(dtype=logits.dtype)
            pos = torch.matmul(pos, pos.T) > 0
            mask[pos] = 1.0

        logsig = F.logsigmoid(mask * logits)

        denominator = (mask == 1).sum(dim=1).clamp(min=1)
        loss = -(logsig.sum(dim=1) / denominator).mean()
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
        self.loss_weights = loss_weights if loss_weights is not None else {}

        self.in_dim = in_dim
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = layers[-1]
        self.layers = layers
        self.dropout = dropout_prob
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons if sk_epsilons is not None else [0.03] * len(n_centroids_list)
        self.sk_iters = sk_iters

        self.encode_layer_dims = [self.in_dim] + self.layers
        self.decode_layer_dims = self.encode_layer_dims[::-1]

        self.embeddings = torch.nn.Parameter(embs_block, requires_grad=False) if embs_block is not None else None

        self.encoder = MLPBlock(layers=self.encode_layer_dims, dropout=self.dropout, apply_final_ln=True)
        self.decoder = MLPBlock(layers=self.decode_layer_dims, dropout=self.dropout, apply_final_ln=False)

        self.rq = ResidualVectorQuantizer(
            n_centroids_list=self.n_centroids_list,
            centroids_dim=self.centroids_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        if self.loss_weights.get("contrastive", 0) > 0:
            self.siglip = SigLIPLoss(tau=tau, bias=bias, freeze_tau=freeze_tau, freeze_bias=freeze_bias)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def training_loss(self, items, timelines):
        assert self.embeddings is not None, "Embeddings must be provided for training."

        loss = 0.0
        metrics = defaultdict(float)

        # --- RECONSTRUCTION COMPLEX LOSSES ---
        if self.loss_weights.get("reconstruction", 0) > 0:
            idxs = torch.randint(0, self.embeddings.shape[0], (len(timelines),), device=self.embeddings.device)
            embeddings = F.embedding(idxs, self.embeddings)

            x = self.encoder(embeddings)
            x_q, rq_loss, _, _ = self.rq(x, use_sk=True)
            x_hat = self.decoder(x_q)

            recon_loss = F.mse_loss(x_hat, embeddings, reduction="mean")
            loss += recon_loss * self.loss_weights["reconstruction"]
            loss += rq_loss * self.loss_weights.get("quantization", 0)

            metrics["reconstruction_loss"] = recon_loss.item()
            metrics["quantization_loss"] = rq_loss.item()

            if self.loss_weights.get("reconstruction_l1_loss", 0) > 0:
                recon_l1_loss = F.l1_loss(x_hat, embeddings, reduction="mean")
                loss += recon_l1_loss * self.loss_weights.get("reconstruction_l1_loss", 0)
                metrics["reconstruction_l1_loss"] = recon_l1_loss.item()

            if self.loss_weights.get("latent_consistency", 0) > 0:
                x_tilde = self.encoder(x_hat)
                latent_cons_loss = F.mse_loss(x_tilde, x_q.detach(), reduction="mean")
                loss += latent_cons_loss * self.loss_weights["latent_consistency"]
                metrics["latent_consistency_loss"] = latent_cons_loss.item()

                if self.loss_weights.get("latent_consistency_l1_loss", 0) > 0:
                    latent_cons_l1_loss = F.l1_loss(x_tilde, x_q.detach(), reduction="mean")
                    loss += latent_cons_l1_loss * self.loss_weights.get("latent_consistency_l1_loss", 0)
                    metrics["latent_consistency_l1_loss"] = latent_cons_l1_loss.item()

        # --- CONTRASTIVE LOSS MATRIX ---
        if self.loss_weights.get("contrastive", 0) > 0:
            embeddings = F.embedding(items, self.embeddings)
            x = self.encoder(embeddings)
            x_q, sig_rq_loss, _, _ = self.rq(x, use_sk=True)

            contrastive_loss = self.siglip(x_q, x_q, items, timelines)

            loss += sig_rq_loss * self.loss_weights.get("quantization", 0)
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
