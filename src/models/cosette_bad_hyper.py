from collections import defaultdict
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


class PoincareBall(nn.Module):
    """Poincare ball operations for hyperbolic space."""
    
    def __init__(self, c=1.0, learnable=True):
        super().__init__()
        # Parameterize as log(c) to keep c > 0
        self.c_log = nn.Parameter(
            torch.tensor(c, dtype=torch.float32).log(),
            requires_grad=learnable
        )
        self.eps = 1e-5
    
    @property
    def c(self):
        return self.c_log.exp()
    
    def exp_map_zero(self, x):
        """Exponential map at origin: Euclidean -> Poincare ball."""
        c = self.c
        x_norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True).clamp(min=self.eps))
        return torch.tanh(c.sqrt() * x_norm) * x / (c.sqrt() * x_norm)
    
    def project(self, x):
        """Project onto Poincare ball to avoid boundary numerical issues."""
        c = self.c
        max_norm = (1.0 - self.eps) / c.sqrt()
        x_norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True).clamp(min=self.eps))
        cond = x_norm > max_norm
        return torch.where(cond, x / x_norm * max_norm, x)
    
    def mobius_add(self, x, y):
        """Mobius addition on Poincare ball."""
        c = self.c
        xy = (x * y).sum(dim=-1, keepdim=True)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
        denom = (1 + 2*c*xy + c**2 * x2 * y2).clamp(min=self.eps)
        return self.project(num / denom)
        
    def pairwise_distance(self, x, y):
        """Computes pairwise Poincare distance using matrix multiplication."""
        c = self.c
        
        x_sqnorm = (x * x).sum(dim=-1)  # [N]
        y_sqnorm = (y * y).sum(dim=-1)  # [M]
        
        # Compute ||x - y||^2 using standard expansion: ||x||^2 + ||y||^2 - 2xy
        xy_inner = torch.matmul(x, y.transpose(-1, -2))  # [N, M]
        sq_dist = x_sqnorm.unsqueeze(1) + y_sqnorm.unsqueeze(0) - 2 * xy_inner
        sq_dist = sq_dist.clamp(min=self.eps)
        
        # Denominators: (1 - c||x||^2) and (1 - c||y||^2)
        x_denom = (1.0 - c * x_sqnorm).clamp(min=self.eps)  # [N]
        y_denom = (1.0 - c * y_sqnorm).clamp(min=self.eps)  # [M]
        denominator = x_denom.unsqueeze(1) * y_denom.unsqueeze(0)
        
        arg = 1.0 + 2 * c * sq_dist / denominator
        arg = arg.clamp(min=1.0 + self.eps)
        
        return (1.0 / c.sqrt()) * torch.acosh(arg)


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
    tensor_centers = torch.from_numpy(centers).to(device)
    return tensor_centers


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
        print("Initializing VQ with KMeans...")
        centers = kmeans(data, self.n_centroids, self.kmeans_iters)
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

        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

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


class HyperbolicSigLIPLoss(torch.nn.Module):
    def __init__(
        self,
        tau,
        bias,
        freeze_tau=False,
        freeze_bias=False,
        c=1.0,
        learnable=True,
    ):
        super().__init__()

        self.tau = nn.Parameter(
            torch.tensor(tau, dtype=torch.float32),
            requires_grad=not freeze_tau,
        )

        self.bias = nn.Parameter(
            torch.tensor(bias, dtype=torch.float32),
            requires_grad=not freeze_bias,
        )

        self.hyp = PoincareBall(
            c=c,
            learnable=learnable,
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
            pos = pos.to(logits.dtype)
            pos = torch.matmul(pos, pos.T) > 0

            mask[pos] = 1.0

        logsig = F.logsigmoid(mask * logits)

        loss = -(logsig.sum(dim=1) / (mask == 1).sum(dim=1)).mean()

        return loss

    def forward(self, xa, xb, items, timelines):
        orig_dtype = xa.dtype

        # Hyperbolic operations are more stable in FP32
        xa_f32 = xa.float()
        xb_f32 = xb.float()

        # Direct mapping into Poincare ball
        xa_h = self.hyp.project(
            self.hyp.exp_map_zero(xa_f32)
        )

        xb_h = self.hyp.project(
            self.hyp.exp_map_zero(xb_f32)
        )

        # Pairwise hyperbolic distances
        dist_matrix = self.hyp.pairwise_distance(
            xa_h,
            xb_h,
        )

        # Convert distance to similarity
        logits = -dist_matrix * self.tau.exp() + self.bias

        # Return to model precision
        logits = logits.to(orig_dtype)

        loss = self._siglip_loss(
            logits,
            items,
            timelines,
        )

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
        loss_weights={},
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
        hyperbolic_c=1.0,
        learnable_c=False
    ):
        super(COSETTE, self).__init__()

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
            else None
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

        if self.loss_weights["contrastive"] > 0:
            self.siglip = HyperbolicSigLIPLoss(
                tau=tau,
                bias=bias,
                freeze_tau=freeze_tau,
                freeze_bias=freeze_bias,
                c=hyperbolic_c,
                learnable=learnable_c,
            )

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def training_loss(self, items, timelines):
        assert self.embeddings is not None, "Embeddings must be provided for training."

        loss = 0.0
        metrics = defaultdict(float)

        if self.loss_weights["reconstruction"] > 0:
            idxs = torch.randint(
                0,
                self.embeddings.shape[0],
                (len(timelines),),
                device=self.embeddings.device,
            )

            embeddings = F.embedding(idxs, self.embeddings)
            x = self.encoder(embeddings)
            x_q, rq_loss, _, _ = self.rq(x, use_sk=True)
            x_hat = self.decoder(x_q)

            recon_loss = F.mse_loss(x_hat, embeddings, reduction="mean")

            loss += recon_loss * self.loss_weights["reconstruction"]
            loss += rq_loss * self.loss_weights["quantization"]
            metrics["reconstruction_loss"] = recon_loss.item()
            metrics["quantization_loss"] = rq_loss.item()

        if self.loss_weights["contrastive"] > 0:
            embeddings = F.embedding(items, self.embeddings)
            x = self.encoder(embeddings)
            x_q, sig_rq_loss, _, _ = self.rq(x, use_sk=True)

            contrastive_loss = self.siglip(x_q, x_q, items, timelines)

            loss += sig_rq_loss * self.loss_weights["quantization"]
            loss += contrastive_loss * self.loss_weights["contrastive"]

            metrics["contrastive_loss"] = contrastive_loss.item()
            metrics["hyperbolic_c"] = self.siglip.hyp.c.item()
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