from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn

# ==========================================
# Phase 1: Pure Poincaré Manifold Primitives
# ==========================================
MIN_NORM = 1e-15
EPS = 1e-5

def project_to_poincare(x, c=1.0):
    """Ensures vectors strictly remain inside the Poincaré ball of radius 1/sqrt(c)."""
    max_norm = (1.0 / c) ** 0.5 - EPS
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)

def mobius_add(x, y, c=1.0):
    """Computes x ⊕_c y in the Poincaré ball. To subtract y, pass -y."""
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + (c ** 2) * x2 * y2

    res = num / denom.clamp_min(MIN_NORM)
    return project_to_poincare(res, c)

def poincare_distance(x, y, c=1.0):
    """Computes the hyperbolic distance between x and y in the Poincaré ball."""
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    diff2 = torch.sum((x - y) ** 2, dim=-1, keepdim=True)
    
    denom = (1 - c * x2) * (1 - c * y2)
    denom = denom.clamp_min(MIN_NORM)
    
    arg = 1 + 2 * c * diff2 / denom
    arg = arg.clamp_min(1.0 + EPS) # Prevent acosh(1) exploding gradients
    
    dist = torch.acosh(arg)
    return dist.squeeze(-1)

def exp_map_0(v, c=1.0):
    """Maps a Euclidean vector v from the tangent space at the origin into the Poincaré ball."""
    norm_v = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    sqrt_c = c ** 0.5
    res = torch.tanh(sqrt_c * norm_v) * (v / (sqrt_c * norm_v))
    return project_to_poincare(res, c)

def log_map_0(y, c=1.0):
    """Maps a Poincaré point y back to the Euclidean tangent space at the origin."""
    norm_y = torch.norm(y, p=2, dim=-1, keepdim=True).clamp_min(MIN_NORM)
    sqrt_c = c ** 0.5
    arg = (sqrt_c * norm_y).clamp_max(1.0 - EPS)
    res = torch.atanh(arg) * (y / (sqrt_c * norm_y))
    return res

# ==========================================
# Standard Modules & Utilities
# ==========================================

class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu"):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout

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

def center_distance_for_constraint(distances):
    max_distance = distances.max()
    min_distance = distances.min()
    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1e-5
    return (distances - middle) / amplitude

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
# Phase 2: Full Hyperbolic Quantizer
# ==========================================

class HyperbolicResidualVectorQuantizer(nn.Module):
    def __init__(self, n_centroids_list, centroids_dim, beta=0.25, sk_epsilons=None, sk_iters=100):
        super().__init__()
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = centroids_dim
        self.beta = beta
        self.sk_epsilons = sk_epsilons if sk_epsilons else [0.0] * len(n_centroids_list)
        self.sk_iters = sk_iters
        
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.empty(n, centroids_dim)) for n in n_centroids_list
        ])
        
        # Tight uniform initialization inside the manifold near the origin
        for cb in self.codebooks:
            cb.data.uniform_(-1e-3, 1e-3)

    def get_codebook(self):
        return torch.stack([cb for cb in self.codebooks])

    def forward(self, x_poincare, use_sk=True, c=1.0):
        x = project_to_poincare(x_poincare, c)
        
        residual = x
        quantized_sum = torch.zeros_like(x)
        
        all_indices = []
        all_distances = []
        codebook_loss = 0.0
        commitment_loss = 0.0

        for l, codebook in enumerate(self.codebooks):
            c_weights = project_to_poincare(codebook, c)
            
            # Distance is strictly Poincare for routing/selection
            r_exp = residual.unsqueeze(1)
            c_exp = c_weights.unsqueeze(0)
            d = poincare_distance(r_exp, c_exp, c) 
            
            # Centroid Assignment
            sk_eps = self.sk_epsilons[l]
            if not use_sk or sk_eps <= 0:
                indices = torch.argmin(d, dim=-1)
            else:
                d_centered = center_distance_for_constraint(d)
                Q = sinkhorn_algorithm(d_centered.double(), sk_eps, self.sk_iters)
                indices = torch.argmax(Q, dim=-1)
            
            selected_codes = c_weights[indices] 
            
            # L2 Loss computation (Equation 3 from HypRQ-VAE paper)
            commitment_loss += F.mse_loss(residual.detach(), selected_codes)
            codebook_loss += F.mse_loss(residual, selected_codes.detach())
            
            # Residual Update via Möbius Math
            quantized_sum = mobius_add(quantized_sum, selected_codes, c)
            residual = mobius_add(residual, -selected_codes, c)
            
            all_indices.append(indices)
            all_distances.append(d)
        
        # PURE HYPERBOLIC Straight-Through Estimator
        # \hat{z} = z ⊕ (\hat{z} ⊖ z).detach()
        diff = mobius_add(quantized_sum, -x, c)
        x_q = mobius_add(x, diff.detach(), c)
        
        loss = codebook_loss + self.beta * commitment_loss
        
        all_indices = torch.stack(all_indices, dim=-1) 
        all_distances = torch.stack(all_distances, dim=1) 
        
        return x_q, loss, all_indices, all_distances, quantized_sum

# ==========================================
# Native Hyperbolic SigLIP Contrastive Loss
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
            pos = pos.to(torch.float32)
            pos = torch.matmul(pos, pos.T) > 0
            mask[pos] = 1.0  

        logsig = F.logsigmoid(mask * logits)
        loss = -(logsig.sum(dim=1) / (mask == 1).sum(dim=1)).mean()
        return loss

    def forward(self, xa_P, xb_P, items, timelines, c=1.0):
        # Pairwise Poincare distances
        xa_exp = xa_P.unsqueeze(1)
        xb_exp = xb_P.unsqueeze(0)
        dist = poincare_distance(xa_exp, xb_exp, c)
        
        # Convert hyperbolic distance to similarity space [-1, 1]
        sim = (2.0 / (1.0 + dist)) - 1.0
        
        logits = sim * self.tau.exp() + self.bias
        loss = self._siglip_loss(logits, items, timelines)
        return loss

# ==========================================
# COSETTE Architect
# ==========================================

class COSETTE(torch.nn.Module):
    def __init__(self, embs_block, in_dim, layers, n_centroids_list, dropout_prob, tau, bias, freeze_tau, freeze_bias, loss_weights={}, kmeans_init=False, kmeans_iters=100, sk_epsilons=None, sk_iters=100):
        super(COSETTE, self).__init__()

        self.in_dim = in_dim
        self.n_centroids_list = n_centroids_list
        self.centroids_dim = layers[-1]
        self.layers = layers
        self.dropout = dropout_prob
        self.loss_weights = loss_weights
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.encode_layer_dims = [self.in_dim] + self.layers
        self.decode_layer_dims = self.encode_layer_dims[::-1]

        self.embeddings = torch.nn.Parameter(embs_block, requires_grad=False) if embs_block is not None else None
        self.encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout)
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout)

        self.rq = HyperbolicResidualVectorQuantizer(
            n_centroids_list=self.n_centroids_list,
            centroids_dim=self.centroids_dim,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        if self.loss_weights.get("contrastive", 0) > 0:
            self.siglip = SigLIPLoss(tau=tau, bias=bias, freeze_tau=freeze_tau, freeze_bias=freeze_bias)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    def training_loss(self, items, timelines, c=1.0):
        loss = 0.0
        metrics = defaultdict(float)

        if self.loss_weights.get("reconstruction", 0) > 0:
            idxs = torch.randint(0, self.embeddings.shape[0], (len(timelines),), device=self.embeddings.device)
            embeddings = F.embedding(idxs, self.embeddings)
            x_e = self.encoder(embeddings)
            
            # ==== ISOLATE HYPERBOLIC MANIFOLD FROM BFLOAT16 ====
            with torch.autocast(device_type="cuda", enabled=False):
                x_f = x_e.float()
                x_p = exp_map_0(x_f, c=c) 
                x_q_f, rq_loss, _, _, _ = self.rq(x_p, use_sk=True, c=c)
                x_q_euc = log_map_0(x_q_f, c=c) 
            # ===================================================

            x_q = x_q_euc.to(x_e.dtype)
            x_hat = self.decoder(x_q)

            recon_loss = F.mse_loss(x_hat, embeddings, reduction="mean")
            loss += recon_loss * self.loss_weights["reconstruction"]
            loss += rq_loss * self.loss_weights["quantization"]
            
            metrics["reconstruction_loss"] = recon_loss.item()
            metrics["quantization_loss"] = rq_loss.item()

        if self.loss_weights.get("contrastive", 0) > 0:
            embeddings = F.embedding(items, self.embeddings)
            x_e = self.encoder(embeddings)
            
            # ==== ISOLATE HYPERBOLIC MANIFOLD FROM BFLOAT16 ====
            with torch.autocast(device_type="cuda", enabled=False):
                x_f = x_e.float()
                x_p = exp_map_0(x_f, c=c)
                _, sig_rq_loss, _, _, x_q_P = self.rq(x_p, use_sk=True, c=c)
                contrastive_loss = self.siglip(x_q_P, x_q_P, items, timelines, c=c)
            # ===================================================

            loss += sig_rq_loss * self.loss_weights["quantization"]
            loss += contrastive_loss * self.loss_weights["contrastive"]

            metrics["contrastive_loss"] = contrastive_loss.item()
            metrics["quantization_loss"] += sig_rq_loss.item()
            metrics["tau"] = self.siglip.tau.item()
            metrics["bias"] = self.siglip.bias.item()
            metrics["n_items_in_batch"] = len(items)

        metrics["manifold_curvature"] = c
        return loss, metrics

    @torch.no_grad()
    def get_indices(self, embeddings=None, use_sk=False, c=1.0):
        if embeddings is None:
            embeddings = self.embeddings
        
        x_e = self.encoder(embeddings)
        
        # Mapping isolation is required here for evaluation
        x_f = x_e.float()
        x_p = exp_map_0(x_f, c=c)
        
        _, _, indices, distances, _ = self.rq(x_p, use_sk=use_sk, c=c)
        return indices, distances