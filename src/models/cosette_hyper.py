import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn

# ==========================================
# 1. HYPERBOLIC MATH UTILITIES
# ==========================================

class RiemannianGradient(torch.autograd.Function):
    """Scales Euclidean gradients to Riemannian gradients for standard optimizers."""
    @staticmethod
    def forward(ctx, x, c):
        ctx.save_for_backward(x, torch.tensor(c, dtype=torch.float32))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, c_tensor = ctx.saved_tensors
        c = c_tensor.item()
        # Conformal factor scale: ((1 - c * ||x||^2)^2) / 4
        x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
        scale = ((1.0 - c * x_sqnorm) ** 2) / 4.0
        return grad_output * scale, None

def project(x, c=1.0):
    """Safely projects points to strictly remain inside the Poincare ball."""
    max_norm = 1.0 / math.sqrt(c) - 1e-5
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-15)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)

def expmap0(v, c=1.0):
    """Maps Euclidean tangent vectors at the origin to the Poincare ball."""
    sqrt_c = math.sqrt(c)
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-15)
    hyp_x = torch.tanh(sqrt_c * v_norm) * (v / (sqrt_c * v_norm))
    return project(hyp_x, c)

def logmap0(y, c=1.0):
    """Maps points in the Poincare ball to the Euclidean tangent space at origin."""
    sqrt_c = math.sqrt(c)
    y_norm = torch.norm(y, dim=-1, keepdim=True).clamp_min(1e-15)
    return torch.atanh(sqrt_c * y_norm) * (y / (sqrt_c * y_norm))

def mobius_add(x, y, c=1.0):
    """Hyperbolic vector addition."""
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    res = num / den.clamp_min(1e-15)
    return project(res, c)

def pairwise_poincare_distance(x, y, c=1.0):
    """Optimized pairwise distance matrix calculation using arccosh."""
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
    y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.matmul(x, y.T)
    
    sqdist = x_sqnorm + y_sqnorm.T - 2 * xy
    den = (1 - c * x_sqnorm) * (1 - c * y_sqnorm.T)
    
    cosh_sq = 1 + 2 * c * sqdist / den.clamp_min(1e-15)
    dist = torch.acosh(cosh_sq.clamp_min(1.0 + 1e-7)) / math.sqrt(c)
    return dist

# ==========================================
# 2. HELPER FUNCTIONS & BASE LAYERS
# ==========================================

class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu"):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if idx != (len(self.layers) - 2):
                mlp_modules.append(nn.ReLU())

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

class HyperbolicMLP(nn.Module):
    """Wraps MLPLayers to safely operate in the tangent space."""
    def __init__(self, layers, dropout=0.0):
        super().__init__()
        self.mlp = MLPLayers(layers, dropout)
        
    def forward(self, x, c=1.0):
        tangent_x = logmap0(x, c)
        tangent_out = self.mlp(tangent_x)
        hyp_out = expmap0(tangent_out, c)
        return project(hyp_out, c)

def kmeans(samples, num_clusters, num_iters=10):
    device = samples.device
    x = samples.cpu().detach().float().numpy()
    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)
    centers = cluster.cluster_centers_
    return torch.from_numpy(centers).to(device)

def hyperbolic_kmeans(samples, num_clusters, c=1.0, num_iters=10):
    """Initializes centroids in tangent space to prevent manifold escape."""
    tangent_samples = logmap0(samples, c)
    tangent_centers = kmeans(tangent_samples, num_clusters, num_iters)
    return expmap0(tangent_centers, c)

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

# ==========================================
# 3. QUANTIZERS
# ==========================================

class VectorQuantizer(nn.Module):
    def __init__(self, n_centroids, centroids_dim, beta=0.25, kmeans_init=True, kmeans_iters=10, sk_epsilon=0.01, sk_iters=100):
        super().__init__()
        self.n_centroids = n_centroids
        self.centroids_dim = centroids_dim
        self.beta = beta
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        
        self.embedding = nn.Embedding(self.n_centroids, self.centroids_dim)
        self.initted = not kmeans_init
        if self.initted:
            self.embedding.weight.data.uniform_(-1e-3, 1e-3)

    def init_emb(self, data, c=1.0):
        centers = hyperbolic_kmeans(data, self.n_centroids, c, 10)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        return (distances - middle) / amplitude

    def forward(self, x, c=1.0, use_sk=True):
        latent = x.view(-1, self.centroids_dim)
        
        if not self.initted and self.training:
            self.init_emb(latent, c)
            
        emb_weights = RiemannianGradient.apply(self.embedding.weight, c)
        
        d = pairwise_poincare_distance(latent, emb_weights, c)
        
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_norm = self.center_distance_for_constraint(d)
            d_norm = d_norm.double()
            Q = sinkhorn_algorithm(d_norm, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        x_q = emb_weights[indices].view(x.shape)
        
        batch_dist = pairwise_poincare_distance(x.view(-1, self.centroids_dim), x_q.view(-1, self.centroids_dim).detach(), c).diag()
        commitment_dist = pairwise_poincare_distance(x.view(-1, self.centroids_dim).detach(), x_q.view(-1, self.centroids_dim), c).diag()
        
        codebook_loss = batch_dist.pow(2).mean()
        commitment_loss = commitment_dist.pow(2).mean()
        loss = codebook_loss + self.beta * commitment_loss

        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        
        return x_q, loss, indices, d

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_centroids_list, centroids_dim, sk_epsilons, kmeans_init=True, kmeans_iters=100, sk_iters=100):
        super().__init__()
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                n_centroids, 
                centroids_dim, 
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_epsilon=sk_epsilon, 
                sk_iters=sk_iters
            )
            for n_centroids, sk_epsilon in zip(n_centroids_list, sk_epsilons)
        ])

    def forward(self, x, c=1.0, use_sk=True):
        all_losses, all_indices, all_distances = [], [], []
        
        x_q = torch.zeros_like(x)
        residual = x
        
        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, c=c, use_sk=use_sk)
            
            residual = mobius_add(-x_res, residual, c)
            x_q = mobius_add(x_q, x_res, c)

            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, mean_losses, all_indices, all_distances

# ==========================================
# 4. LOSS & MAIN MODEL
# ==========================================

class SigLIPLoss(nn.Module):
    def __init__(self, tau, bias, freeze_tau=False, freeze_bias=False):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=not freeze_tau)
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=not freeze_bias)

    def _siglip_loss(self, logits, items, timelines):
        with torch.no_grad():
            mask = torch.full((len(items), len(items)), -1, dtype=torch.float32, device=self.tau.device)
            pos = (items[:, None, None] == timelines[None, :, :]).any(axis=2)
            pos = pos.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
            pos = torch.matmul(pos, pos.T) > 0
            mask[pos] = 1.0

        logsig = F.logsigmoid(mask * logits)
        loss = -(logsig.sum(dim=1) / (mask == 1).sum(dim=1)).mean()
        return loss

    def forward(self, xa, xb, items, timelines, c=1.0):
        distances = pairwise_poincare_distance(xa, xb, c)
        logits = -distances * self.tau.exp() + self.bias
        return self._siglip_loss(logits, items, timelines)

class COSETTE(nn.Module):
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
    ):
        super().__init__()

        self.in_dim = in_dim
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = layers[-1]
        self.loss_weights = loss_weights

        # Embeddings strictly projected to the Poincare ball
        if embs_block is not None:
            self.embeddings = nn.Parameter(project(embs_block), requires_grad=False)
        else:
            self.embeddings = None

        encode_layer_dims = [self.in_dim] + layers
        decode_layer_dims = encode_layer_dims[::-1]

        self.encoder = HyperbolicMLP(encode_layer_dims, dropout_prob)
        self.decoder = HyperbolicMLP(decode_layer_dims, dropout_prob)

        self.rq = ResidualVectorQuantizer(
            n_centroids_list=self.n_centroids_list,
            centroids_dim=self.centroids_dim,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
        )

        if self.loss_weights.get("contrastive", 0) > 0:
            self.siglip = SigLIPLoss(tau=tau, bias=bias, freeze_tau=freeze_tau, freeze_bias=freeze_bias)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def training_loss(self, items, timelines, c=1.0):
        assert self.embeddings is not None, "Embeddings must be provided for training."

        loss = 0.0
        metrics = defaultdict(float)

        if self.loss_weights.get("reconstruction", 0) > 0:
            idxs = torch.randint(0, self.embeddings.shape[0], (len(timelines),), device=self.embeddings.device)
            emb = F.embedding(idxs, self.embeddings)
            
            x = self.encoder(emb, c)
            x_q, rq_loss, _, _ = self.rq(x, c=c, use_sk=True)
            x_hat = self.decoder(x_q, c)

            recon_dist = pairwise_poincare_distance(x_hat, emb, c).diag()
            recon_loss = recon_dist.pow(2).mean()

            loss += recon_loss * self.loss_weights["reconstruction"]
            loss += rq_loss * self.loss_weights["quantization"]
            metrics["reconstruction_loss"] = recon_loss.item()
            metrics["quantization_loss"] = rq_loss.item()

        if self.loss_weights.get("contrastive", 0) > 0:
            emb = F.embedding(items, self.embeddings)
            x = self.encoder(emb, c)
            x_q, sig_rq_loss, _, _ = self.rq(x, c=c, use_sk=True)

            x_q_scaled = RiemannianGradient.apply(x_q, c)
            contrastive_loss = self.siglip(x_q_scaled, x_q_scaled, items, timelines, c)

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
        if embeddings is None:
            embeddings = self.embeddings

        x_e = self.encoder(embeddings, c)
        _, _, indices, distances = self.rq(x_e, c=c, use_sk=use_sk)
        return indices, distances