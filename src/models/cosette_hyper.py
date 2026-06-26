"""COSETTE model architecture and submodules (Hyperbolic version using Poincare ball)."""

from collections import defaultdict
import math

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn

MIN_NORM = 1e-15


def project(x, c=1.0):
    """Projects points to ensure they strictly remain inside the Poincare ball."""
    norm = torch.norm(x, dim=-1, keepdim=True)
    maxnorm = (1.0 - 1e-5) / math.sqrt(c)
    # Project vector back into the Poincare ball if it exceeds the maximum allowed norm
    cond = norm > maxnorm
    projected = x / norm.clamp_min(MIN_NORM) * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c=1.0):
    """Performs Mobius addition of x and y in the Poincare ball."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    # Mobius addition formula: ( (1 + 2c<x,y> + c|y|^2)x + (1 - c|x|^2)y ) / (1 + 2c<x,y> + c^2|x|^2|y|^2 )
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c**2 * x2 * y2

    return project(num / den.clamp_min(MIN_NORM), c)


def expmap0(u, c=1.0):
    """Maps a vector u from the Euclidean tangent space at origin to the Poincare ball."""
    sqrt_c = math.sqrt(c)
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    return project(torch.tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm), c)


def logmap0(y, c=1.0):
    """Maps a point y from the Poincare ball to the Euclidean tangent space at origin."""
    sqrt_c = math.sqrt(c)
    y_norm = torch.norm(y, dim=-1, keepdim=True).clamp(
        min=MIN_NORM, max=(1.0 - 1e-5) / sqrt_c
    )
    return (torch.atanh(sqrt_c * y_norm) * y) / (sqrt_c * y_norm)


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
    device = samples.device
    x = samples.cpu().detach().float().numpy()
    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)
    centers = cluster.cluster_centers_
    return torch.from_numpy(centers).to(device)


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)
    B = Q.shape[0]
    K = Q.shape[1]

    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K
    Q *= B
    return Q


class HyperbolicVectorQuantizer(nn.Module):
    """Vector Quantizer that operates within hyperbolic space."""
    def __init__(
        self,
        n_centroids,
        centroids_dim,
        c=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.01,
        sk_iters=100,
    ):
        super().__init__()
        self.n_centroids = n_centroids
        self.centroids_dim = centroids_dim
        self.c = c
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_centroids, self.centroids_dim)
        if not kmeans_init:
            self.initted = True
            max_val = (1.0 / self.n_centroids) * (1.0 / math.sqrt(self.c))
            self.embedding.weight.data.uniform_(-max_val, max_val)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return project(self.embedding.weight, self.c)

    def init_emb(self, data):
        print("Initializing VQ with KMeans...")
        centers = kmeans(data, self.n_centroids, self.kmeans_iters)
        centers = project(centers, self.c)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        return (distances - middle) / amplitude

    def forward(self, x, use_sk=True):
        latent = project(x.view(-1, self.centroids_dim), self.c)
        codebook = project(self.embedding.weight, self.c)

        if not self.initted and self.training:
            self.init_emb(latent)
            codebook = project(self.embedding.weight, self.c)

        x_expand = latent.unsqueeze(1)
        emb_expand = codebook.unsqueeze(0)

        # Calculate hyperbolic distance: 2/sqrt(c) * atanh(sqrt(c) * ||-x + y||)
        minus_x = -x_expand
        m_add = mobius_add(minus_x, emb_expand, self.c)
        m_add_norm = torch.norm(m_add, dim=-1).clamp(max=1.0 - 1e-5)
        d = (2.0 / math.sqrt(self.c)) * torch.atanh(math.sqrt(self.c) * m_add_norm)

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d).double()
            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)
            indices = torch.argmax(Q, dim=-1)

        x_q = codebook[indices].view(x.shape)

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices, d


class HyperbolicResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_centroids_list,
        centroids_dim,
        sk_epsilons,
        c=1.0,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
    ):
        super().__init__()
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = centroids_dim
        self.c = c
        self.num_quantizers = len(n_centroids_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        self.vq_layers = nn.ModuleList(
            [
                HyperbolicVectorQuantizer(
                    n_centroids,
                    centroids_dim,
                    c=self.c,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                )
                for n_centroids, sk_epsilon in zip(n_centroids_list, sk_epsilons)
            ]
        )

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_distances = []

        x_q_agg = None
        residual = x

        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, use_sk=use_sk)
            residual = mobius_add(residual, -x_res, c=self.c)

            if x_q_agg is None:
                x_q_agg = x_res
            else:
                x_q_agg = mobius_add(x_q_agg, x_res, c=self.c)

            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)

        x_q_agg = x + (x_q_agg - x).detach()

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q_agg, mean_losses, all_indices, all_distances


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
            # N x N
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
            mask[pos] = 1.0  # Positive items

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
    """Hyperbolic variation of the COSETTE model for learning semantic IDs in a Poincare ball space."""
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
        loss_weights={},
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
        c=1.0,
    ):
        super(COSETTE, self).__init__()

        self.in_dim = in_dim
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = layers[-1]
        self.c = c

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
            else None
        )

        self.encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout)
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout)

        self.rq = HyperbolicResidualVectorQuantizer(
            n_centroids_list=self.n_centroids_list,
            centroids_dim=self.centroids_dim,
            c=self.c,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        if self.loss_weights.get("contrastive", 0.0) > 0:
            self.siglip = SigLIPLoss(
                tau=tau,
                bias=bias,
                freeze_tau=freeze_tau,
                freeze_bias=freeze_bias,
            )

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def training_loss(self, items, timelines, c=1.0):
        self.rq.c = c
        assert self.embeddings is not None, "Embeddings must be provided for training."

        loss = 0.0
        metrics = defaultdict(float)

        if self.loss_weights.get("reconstruction", 0.0) > 0:
            idxs = torch.randint(
                0,
                self.embeddings.shape[0],
                (len(timelines) if timelines is not None else 256,),
                device=self.embeddings.device,
            )

            embeddings = F.embedding(idxs, self.embeddings)

            z_e = self.encoder(embeddings)
            z_h = expmap0(z_e, c=c)
            z_q_h, rq_loss, _, _ = self.rq(z_h, use_sk=True)
            z_q_e = logmap0(z_q_h, c=c)
            x_hat = self.decoder(z_q_e)

            recon_loss = F.mse_loss(x_hat, embeddings, reduction="mean")

            loss += recon_loss * self.loss_weights["reconstruction"]
            loss += rq_loss * self.loss_weights["quantization"]

            metrics["reconstruction_loss"] = recon_loss.item()
            metrics["quantization_loss"] = rq_loss.item()

        if self.loss_weights.get("contrastive", 0.0) > 0:
            embeddings = F.embedding(items, self.embeddings)

            z_e = self.encoder(embeddings)
            z_h = expmap0(z_e, c=c)
            z_q_h, sig_rq_loss, _, _ = self.rq(z_h, use_sk=True)
            z_q_e = logmap0(z_q_h, c=c)

            contrastive_loss = self.siglip(z_q_e, z_q_e, items, timelines)

            loss += sig_rq_loss * self.loss_weights["quantization"]
            loss += contrastive_loss * self.loss_weights["contrastive"]

            metrics["contrastive_loss"] = contrastive_loss.item()
            metrics["quantization_loss"] += sig_rq_loss.item()
            metrics["tau"] = self.siglip.tau.item()
            metrics["bias"] = self.siglip.bias.item()
            metrics["n_items_in_batch"] = len(items)

        return loss, metrics

    @torch.no_grad()
    def get_indices(self, embeddings=None, use_sk=False, c=1.0):
        self.rq.c = c
        if embeddings is None:
            embeddings = self.embeddings

        z_e = self.encoder(embeddings)
        z_h = expmap0(z_e, c=c)
        _, _, indices, distances = self.rq(z_h, use_sk=use_sk)
        return indices, distances
