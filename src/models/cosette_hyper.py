from collections import defaultdict

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn

# ==========================================
# Phase 1: Optimized Lorentz Manifold Primitives
# ==========================================

def lorentz_inner(u, v):
    """Optimized: single sum and subtraction"""
    return -u[..., 0] * v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)

def distance_lorentz(x, y):
    inner = lorentz_inner(x, y)
    inner = torch.clamp(inner, max=-1.0 - 1e-7)
    return torch.acosh(-inner)

def project_to_manifold(x):
    """Forced float32 internally to prevent bfloat16 NaN collapse."""
    x_f = x.float()
    x_norm_sq = (x_f ** 2).sum(dim=-1, keepdim=True)
    x0 = torch.sqrt(1.0 + x_norm_sq)
    return torch.cat([x0, x_f], dim=-1).to(x.dtype)

def log_map(x, y):
    inner = lorentz_inner(x, y)
    inner = torch.clamp(inner, max=-1.0 - 1e-7)
    dist = torch.acosh(-inner)
    
    # Taylor expansion fallback for stability to avoid 0/0
    sinh_dist = torch.sinh(dist)
    scale = torch.where(dist > 1e-6, dist / torch.clamp(sinh_dist, min=1e-6), 1.0)
    
    v = y + inner.unsqueeze(-1) * x
    return scale.unsqueeze(-1) * v

def exp_map(x, v):
    norm_v_sq = lorentz_inner(v, v)
    norm_v_sq = torch.clamp(norm_v_sq, min=0.0) # Prevent NaNs from fp precision
    norm_v = torch.sqrt(norm_v_sq)
    
    # Taylor expansion of sinh(x)/x around 0 is 1 + x^2/6
    scale = torch.where(norm_v > 1e-6, torch.sinh(norm_v) / norm_v, 1.0 + norm_v_sq / 6.0)
    return torch.cosh(norm_v).unsqueeze(-1) * x + scale.unsqueeze(-1) * v

def parallel_transport_to_origin(x, v):
    """Analytically simplified: No zero-tensors allocated. Fixed sign bug."""
    x0 = x[..., 0:1]
    v0 = v[..., 0:1]
    factor = -v0 / (1.0 + x0)
    
    out = v + factor * x
    out[..., 0:1] += factor # (x + O)_0 is x0 + 1. We add the +1 part here.
    return out

def parallel_transport_from_origin(x, v):
    """Analytically simplified: O -> x transport."""
    x0 = x[..., 0:1]
    # v is in T_O, so v_0 is 0. Inner product is just the spatial part.
    inner_xv = (x[..., 1:] * v[..., 1:]).sum(dim=-1, keepdim=True)
    factor = inner_xv / (1.0 + x0)
    
    out = v + factor * x
    out[..., 0:1] += factor
    return out

# ==========================================
# Standard Modules
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


class VectorQuantizer(nn.Module):
    """Standard Euclidean Quantizer for Levels 2..N"""
    def __init__(self, n_centroids, centroids_dim, beta=0.25, kmeans_init=False, kmeans_iters=10, sk_epsilon=0.01, sk_iters=100):
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
            self.embedding.weight.data.uniform_(-1.0 / self.n_centroids, 1.0 / self.n_centroids)
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
        centers = kmeans(data, self.n_centroids, self.kmeans_iters)
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
            d_centered = self.center_distance_for_constraint(d)
            Q = sinkhorn_algorithm(d_centered.double(), self.sk_epsilon, self.sk_iters)
            indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).view(x.shape)
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices, d


# ==========================================
# Phase 1: Hyperbolic Quantizer Components
# ==========================================

class HyperbolicVectorQuantizer(nn.Module):
    """Lorentz Quantizer for Level 1 Root Taxonomy."""
    def __init__(self, n_centroids, centroids_dim, beta=0.25, kmeans_init=False, kmeans_iters=10, sk_epsilon=0.01, sk_iters=100):
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
            self.embedding.weight.data.uniform_(-1.0 / self.n_centroids, 1.0 / self.n_centroids)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def init_emb(self, data_R):
        print("Initializing Hyperbolic VQ with KMeans (on R^d projection)...")
        centers = kmeans(data_R, self.n_centroids, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x_L, use_sk=True):
        centroids_R = self.embedding.weight 
        centroids_L = project_to_manifold(centroids_R) 

        # 1. Fast Distance Matrix
        # Only compute the raw inner product (matrix multiplication is extremely fast on GPUs)
        inner = -x_L[:, 0:1] * centroids_L[:, 0:1].t() + torch.matmul(x_L[:, 1:], centroids_L[:, 1:].t())
        
        if not use_sk or self.sk_epsilon <= 0:
            # OPTIMIZATION: Minimizing acosh(-inner) is identical to maximizing inner.
            indices = torch.argmax(inner, dim=-1)
            d = -inner # Proxy distance to maintain API output
        else:
            d = torch.acosh(torch.clamp(-inner, max=-1.0 - 1e-5))
            d_centered = VectorQuantizer.center_distance_for_constraint(d)
            Q = sinkhorn_algorithm(d_centered.double(), self.sk_epsilon, self.sk_iters)
            indices = torch.argmax(Q, dim=-1)

        c1_R = self.embedding(indices) 
        c1 = project_to_manifold(c1_R) 
        
        # 2. Tangent Extraction
        v_target = log_map(c1, x_L) 
        r_target_L = parallel_transport_to_origin(c1, v_target)
        r_target_E = r_target_L[:, 1:] 

        # 3. Exact Codebook Losses (Computed ONLY for the 1 selected centroid, not all 256)
        commitment_loss = distance_lorentz(x_L.detach(), c1).mean()
        codebook_loss = distance_lorentz(x_L, c1.detach()).mean()
        loss = codebook_loss + self.beta * commitment_loss

        return c1, r_target_E, loss, indices, d


class HyperbolicResidualVectorQuantizer(nn.Module):
    """Hybird RVQ: L1 = Lorentz, L2..N = Tangent Euclidean"""
    def __init__(self, n_centroids_list, centroids_dim, sk_epsilons, kmeans_init=False, kmeans_iters=100, sk_iters=100):
        super().__init__()
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = centroids_dim
        self.num_quantizers = len(n_centroids_list)
        
        # Level 1: Hyperbolic Root
        self.hyp_vq = HyperbolicVectorQuantizer(
            n_centroids_list[0], centroids_dim, kmeans_init=kmeans_init, 
            kmeans_iters=kmeans_iters, sk_epsilon=sk_epsilons[0], sk_iters=sk_iters
        )
        
        # Levels 2..N: Euclidean in Tangent Space
        self.euc_vqs = nn.ModuleList([
            VectorQuantizer(n, centroids_dim, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, sk_epsilon=eps, sk_iters=sk_iters)
            for n, eps in zip(n_centroids_list[1:], sk_epsilons[1:])
        ])

    def get_codebook(self):
        all_codebook = [self.hyp_vq.get_codebook()]
        for quantizer in self.euc_vqs:
            all_codebook.append(quantizer.get_codebook())
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        if not self.hyp_vq.initted and self.training:
            self.hyp_vq.init_emb(x)

        # 1. Project base Euclidean vector onto Lorentz Manifold
        x_L = project_to_manifold(x)
        
        # 2. Hyperbolic Root Tokenization
        c1, residual_E, loss0, idx0, d0 = self.hyp_vq(x_L, use_sk=use_sk)
        
        all_losses = [loss0]
        all_indices = [idx0]
        all_distances = [d0]
        
        # 3. Euclidean Tangent Space Tokenization
        r_hat_E = 0
        for quantizer in self.euc_vqs:
            x_res, loss, indices, distance = quantizer(residual_E, use_sk=use_sk)
            residual_E = residual_E - x_res
            r_hat_E = r_hat_E + x_res
            
            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)
            
        # 4. Reconstruct and Map Back to Manifold
        r_hat_L = torch.cat([torch.zeros_like(r_hat_E[:, 0:1]), r_hat_E], dim=-1)
        v_hat = parallel_transport_from_origin(c1, r_hat_L)
        x_q_L = exp_map(c1, v_hat)
        
        # Straight-through estimator on the manifold
        x_q_L = x_L + (x_q_L - x_L).detach()
        
        # 5. Inverse projection back to flat R^d space for the Decoder
        x_q = x_q_L[:, 1:]
        
        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, mean_losses, all_indices, all_distances


# ==========================================
# Wrapping it up in COSETTE
# ==========================================

class SigLIPLoss(torch.nn.Module):
    def __init__(self, tau, bias, freeze_tau=False, freeze_bias=False):
        super(SigLIPLoss, self).__init__()
        self.tau = torch.nn.Parameter(torch.tensor(tau, dtype=torch.float32), requires_grad=not freeze_tau)
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32), requires_grad=not freeze_bias)

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

    def forward(self, xa, xb, items, timelines):
        xa = F.normalize(xa, dim=-1)
        xb = F.normalize(xb, dim=-1)
        logits = torch.mm(xa, xb.T) * self.tau.exp() + self.bias
        loss = self._siglip_loss(logits, items, timelines)
        return loss


class COSETTE(torch.nn.Module):
    def __init__(self, embs_block, in_dim, layers, n_centroids_list, dropout_prob, tau, bias, freeze_tau, freeze_bias, loss_weights={}, kmeans_init=False, kmeans_iters=100, sk_epsilons=None, sk_iters=100):
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

        self.embeddings = torch.nn.Parameter(embs_block, requires_grad=False) if embs_block is not None else None
        self.encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout)
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout)

        # Replaced standard RVQ with the new Hyperbolic RVQ
        self.rq = HyperbolicResidualVectorQuantizer(
            n_centroids_list=self.n_centroids_list,
            centroids_dim=self.centroids_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        if self.loss_weights["contrastive"] > 0:
            self.siglip = SigLIPLoss(tau=tau, bias=bias, freeze_tau=freeze_tau, freeze_bias=freeze_bias)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def training_loss(self, items, timelines):
        loss = 0.0
        metrics = defaultdict(float)

        if self.loss_weights["reconstruction"] > 0:
            idxs = torch.randint(0, self.embeddings.shape[0], (len(timelines),), device=self.embeddings.device)
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