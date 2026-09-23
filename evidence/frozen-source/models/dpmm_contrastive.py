"""
DPMMODEContrastive: DPMMODE with momentum contrastive learning.

Core components:
- Momentum Contrastive (MoCo) learning for representation quality
- InfoNCE loss for self-supervised contrastive pre-training
- Data augmentation strategies for single-cell data

Single-phase training:
- Phase 1: Train AE/VAE with DPMM clustering + contrastive learning (fit method)

"""

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture

try:
    from .base_model import BaseModel
except ImportError:
    from utils.base_model import BaseModel


def weight_init(m):
    """Xavier normal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "mish":
        return nn.Mish()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class _Layer1D(nn.Module):
    """Normalization + Activation + Dropout layer"""

    def __init__(
        self, dim: int, norm: str | None = None, act: str | None = None, drop: float = 0.0
    ):
        super().__init__()
        layers = []
        if norm == "bn":
            layers.append(nn.BatchNorm1d(dim))
        elif norm == "ln":
            layers.append(nn.LayerNorm(dim))
        if act:
            layers.append(_act(act))
        if drop and drop > 0:
            layers.append(nn.Dropout(drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """Configurable MLP with flexible normalization, activation, and dropout"""

    def __init__(
        self,
        features: list,
        hid_act: str = "mish",
        out_act: str | None = None,
        norm: str | None = None,
        hid_norm: str | None = None,
        drop: float = 0.0,
        hid_drop: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i in range(1, len(features)):
            is_last = i == len(features) - 1
            cur_norm = norm if is_last else hid_norm
            cur_act = out_act if is_last else hid_act
            cur_drop = drop if is_last else hid_drop

            layers.append(nn.Linear(features[i - 1], features[i]))
            if cur_norm or cur_act or cur_drop:
                layers.append(_Layer1D(features[i], norm=cur_norm, act=cur_act, drop=cur_drop))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DataAugmentation(nn.Module):
    """Enhanced data augmentation strategies for single-cell data

    Integrates augmentation from:
    - scSimGCL: Feature dropout + noise
    - scHSC: Laplacian smoothing based augmentation
    - scGPCL: Heterogeneous dropout
    """

    def __init__(
        self,
        noise_prob: float = 0.2,
        noise_std: float = 0.1,
        mask_prob: float = 0.1,
        scale_range: tuple[float, float] = (0.8, 1.2),
        feature_dropout: float = 0.2,  # From scSimGCL
        edge_dropout: float = 0.2,  # From scSimGCL
    ):
        super().__init__()
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.scale_range = scale_range
        self.feature_dropout = feature_dropout
        self.edge_dropout = edge_dropout

    def forward(self, x: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        """Apply random augmentation with scSimGCL-style dropout"""
        batch_size, n_features = x.shape
        device = x.device
        x_aug = x.clone()

        # scSimGCL-style feature dropout (Bernoulli mask)
        if self.feature_dropout > 0:
            feature_mask = torch.bernoulli(
                torch.ones(batch_size, n_features, device=device) * (1 - self.feature_dropout)
            )
            x_aug = x_aug * feature_mask

        # Gaussian noise (enhanced)
        if self.noise_prob > 0 and torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise

        # Additional feature masking
        if self.mask_prob > 0:
            mask = torch.rand(batch_size, n_features, device=device) > self.mask_prob
            x_aug = x_aug * mask.float()

        # Random scaling
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = torch.empty(batch_size, 1, device=device).uniform_(*self.scale_range)
            x_aug = x_aug * scale

        if return_mask and self.feature_dropout > 0:
            return x_aug, feature_mask
        return x_aug


class MomentumEncoder(nn.Module):
    """Momentum-updated encoder for MoCo"""

    def __init__(self, encoder: nn.Module, latent_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, embedding_dim)
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and project to embedding space"""
        z = self.encoder(x)
        z_proj = self.projector(z)
        return F.normalize(z_proj, dim=1)


class MomentumContrast(nn.Module):
    """Enhanced Momentum Contrast (MoCo) module with multi-level contrastive learning

    Integrates contrastive strategies from:
    - MoCo: Momentum encoder + memory queue
    - scAGCL: Symmetric contrastive loss + projection
    - scGPCL: Instance-level + Prototype-level contrastive
    - scHSC: Hard sample weighting
    """

    def __init__(
        self,
        latent_dim: int,
        embedding_dim: int = 128,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.2,
        device: torch.device = torch.device("cuda"),
        use_prototype: bool = True,  # From scGPCL
        n_prototypes: int = 10,
    ):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.device = device
        self.embedding_dim = embedding_dim
        self.use_prototype = use_prototype
        self.n_prototypes = n_prototypes

        # Query projector (scAGCL-style 2-layer MLP)
        self.query_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embedding_dim),
        )

        # Key projector (momentum-updated copy)
        self.key_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embedding_dim),
        )

        # Initialize key projector with query projector weights
        for param_q, param_k in zip(
            self.query_projector.parameters(), self.key_projector.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Initialize queue
        self.register_buffer("queue", torch.randn(embedding_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Prototype parameters (from scGPCL)
        if use_prototype:
            self.prototypes = nn.Parameter(torch.randn(n_prototypes, embedding_dim))
            nn.init.xavier_uniform_(self.prototypes)

        self.apply(weight_init)

    @torch.no_grad()
    def _momentum_update(self):
        """Update key projector with momentum"""
        for param_q, param_k in zip(
            self.query_projector.parameters(), self.key_projector.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        else:
            part1_size = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:part1_size].T
            part2_size = batch_size - part1_size
            self.queue[:, :part2_size] = keys[part1_size:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(
        self, z_query: torch.Tensor, z_key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute contrastive logits and labels

        Args:
            z_query: Query latent vectors [batch, latent_dim]
            z_key: Key latent vectors [batch, latent_dim]

        Returns:
            logits: Contrastive logits [batch, 1 + queue_size]
            labels: Target labels (zeros for positive pairs)
        """
        # Project queries
        q = self.query_projector(z_query)
        q = F.normalize(q, dim=1)

        # Project keys with momentum update
        with torch.no_grad():
            self._momentum_update()
            k = self.key_projector(z_key)
            k = F.normalize(k, dim=1)

        # Positive logits: [batch, 1]
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: [batch, queue_size]
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits: [batch, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: positive pairs have index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def symmetric_contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """scAGCL-style symmetric contrastive loss"""
        h1 = self.query_projector(z1)
        h2 = self.query_projector(z2)
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)

        # Compute similarity
        sim_11 = torch.mm(h1, h1.T) / self.temperature
        sim_12 = torch.mm(h1, h2.T) / self.temperature
        sim_22 = torch.mm(h2, h2.T) / self.temperature

        batch_size = h1.size(0)  # noqa: F841

        # Symmetric loss (scAGCL)
        pos_sim = torch.diag(sim_12)

        # Loss from view 1
        neg_sim_1 = torch.cat([sim_11, sim_12], dim=1)
        neg_sim_1 = neg_sim_1 - torch.diag(torch.diag(sim_11)).repeat(1, 2)[:, : neg_sim_1.size(1)]
        l1 = -pos_sim + torch.logsumexp(neg_sim_1, dim=1)

        # Loss from view 2
        neg_sim_2 = torch.cat([sim_22, sim_12.T], dim=1)
        neg_sim_2 = neg_sim_2 - torch.diag(torch.diag(sim_22)).repeat(1, 2)[:, : neg_sim_2.size(1)]
        l2 = -pos_sim + torch.logsumexp(neg_sim_2, dim=1)

        return (l1.mean() + l2.mean()) / 2

    def prototype_contrastive_loss(
        self, z: torch.Tensor, cluster_assignments: torch.Tensor | None = None
    ) -> torch.Tensor:
        """scGPCL-style prototype contrastive loss"""
        if not self.use_prototype:
            return torch.tensor(0.0, device=z.device)

        h = self.query_projector(z)
        h = F.normalize(h, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)

        # Similarity to prototypes
        sim = torch.mm(h, prototypes.T) / self.temperature

        if cluster_assignments is not None:
            # Supervised prototype loss
            pos_mask = F.one_hot(cluster_assignments, num_classes=self.n_prototypes).float()
            pos_sim = (sim * pos_mask).sum(dim=1)
            loss = -pos_sim + torch.logsumexp(sim, dim=1)
        else:
            # Self-supervised: assign to nearest prototype
            assignments = sim.argmax(dim=1)
            pos_mask = F.one_hot(assignments, num_classes=self.n_prototypes).float()
            pos_sim = (sim * pos_mask).sum(dim=1)
            loss = -pos_sim + torch.logsumexp(sim, dim=1)

        return loss.mean()


class InformationBottleneck(nn.Module):
    """Compress and reconstruct latent space"""

    def __init__(self, latent_dim: int, bottleneck_dim: int, norm: str = "bn", drop: float = 0.1):
        super().__init__()
        self.compress = MLP(
            [latent_dim, latent_dim // 2, bottleneck_dim],
            norm=norm,
            hid_norm=norm,
            hid_drop=drop,
            out_act=None,
        )
        self.expand = MLP(
            [bottleneck_dim, latent_dim // 2, latent_dim],
            norm=norm,
            hid_norm=norm,
            hid_drop=drop,
            out_act=None,
        )

    def forward(self, z):
        z_bottleneck = self.compress(z)
        z_reconstructed = self.expand(z_bottleneck)
        return z_bottleneck, z_reconstructed


class DPMMContrastiveAutoEncoder(nn.Module):
    """Autoencoder with MoCo contrastive learning, optional VAE, bottleneck, and ODE components."""

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list,
        latent_dim: int,
        decoder_dims: list,
        norm: str,
        drop: float,
        use_bottleneck: bool = False,
        bottleneck_dim: int | None = None,
        use_vae: bool = False,
        var_eps: float = 1e-4,
        # Contrastive learning params
        use_moco: bool = True,
        moco_embedding_dim: int = 128,
        moco_queue_size: int = 4096,
        moco_momentum: float = 0.999,
        moco_temperature: float = 0.2,
        use_prototype: bool = True,  # scGPCL-style prototype learning
        n_prototypes: int = 10,
        # Augmentation params
        aug_noise_prob: float = 0.2,
        aug_noise_std: float = 0.1,
        aug_mask_prob: float = 0.1,
        aug_feature_dropout: float = 0.2,  # scSimGCL-style
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.use_vae = use_vae
        self.var_eps = var_eps
        self.latent_dim = latent_dim
        self.use_moco = use_moco
        self.device = device

        # Encoder
        if use_vae:
            self.encoder_backbone = MLP(
                [input_dim] + encoder_dims, norm=norm, hid_norm=norm, hid_drop=drop, out_act="mish"
            )
            self.mu_head = nn.Linear(encoder_dims[-1], latent_dim)
            self.var_head = nn.Linear(encoder_dims[-1], latent_dim)
        else:
            self.encoder = MLP(
                [input_dim] + encoder_dims + [latent_dim],
                norm=norm,
                hid_norm=norm,
                hid_drop=drop,
                out_act=None,
            )

        self.decoder = MLP(
            [latent_dim] + decoder_dims + [input_dim],
            norm=norm,
            hid_norm=norm,
            hid_drop=drop,
            out_act=None,
        )

        self.use_bottleneck = use_bottleneck
        self.apply(weight_init)

        if use_bottleneck:
            if bottleneck_dim is None:
                bottleneck_dim = max(latent_dim // 2, 8)
            self.bottleneck = InformationBottleneck(latent_dim, bottleneck_dim, norm, drop)

        # Enhanced contrastive learning components (integrating scSimGCL, scAGCL, scGPCL, scHSC)
        if use_moco:
            self.augmentation = DataAugmentation(
                noise_prob=aug_noise_prob,
                noise_std=aug_noise_std,
                mask_prob=aug_mask_prob,
                feature_dropout=aug_feature_dropout,  # scSimGCL-style
            )
            self.moco = MomentumContrast(
                latent_dim=latent_dim,
                embedding_dim=moco_embedding_dim,
                queue_size=moco_queue_size,
                momentum=moco_momentum,
                temperature=moco_temperature,
                device=device,
                use_prototype=use_prototype,  # scGPCL-style
                n_prototypes=n_prototypes,
            )

    def encode_vae(self, x: torch.Tensor):
        """VAE encoding: returns (mu, var)"""
        h = self.encoder_backbone(x)
        mu = self.mu_head(h)
        var = torch.nn.functional.softplus(self.var_head(h)) + self.var_eps
        return mu, var

    def encode_ae(self, x: torch.Tensor):
        """AE encoding: returns latent directly"""
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Unified encode method"""
        if self.use_vae:
            mu, var = self.encode_vae(x)
            return self.reparameterize(mu, var)
        else:
            return self.encode_ae(x)

    def forward(self, x, x_aug_q=None, x_aug_k=None):
        """Forward pass with optional contrastive learning"""
        return self._forward_normal(x, x_aug_q, x_aug_k)

    def _forward_normal(self, x, x_aug_q=None, x_aug_k=None):
        if self.use_vae:
            mu, var = self.encode_vae(x)
            z = self.reparameterize(mu, var)
        else:
            z = self.encode_ae(x)
            mu, var = None, None

        # Enhanced contrastive learning with scSimGCL-style augmentation
        moco_logits, moco_labels = None, None
        z_aug_q, z_aug_k = None, None

        if self.use_moco and self.training:
            if x_aug_q is None:
                x_aug_q = self.augmentation(x)
            if x_aug_k is None:
                x_aug_k = self.augmentation(x)

            z_aug_q = self.encode(x_aug_q)
            z_aug_k = self.encode(x_aug_k)
            moco_logits, moco_labels = self.moco(z_aug_q, z_aug_k)

        if self.use_bottleneck:
            z_le, z_ld = self.bottleneck(z)
            x_hat = self.decoder(z)
            x_le_hat = self.decoder(z_ld)
            return x_hat, z, x_le_hat, z_le, mu, var, moco_logits, moco_labels, z_aug_q, z_aug_k
        else:
            x_hat = self.decoder(z)
            return x_hat, z, None, None, mu, var, moco_logits, moco_labels, z_aug_q, z_aug_k


class DPMMODEContrastiveModel(BaseModel):
    """DPMMODEContrastive: AE/VAE with DPMM clustering, MoCo contrastive learning, and optional ODE dynamics."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        encoder_dims: list | None = None,
        decoder_dims: list | None = None,
        norm_type: str = "bn",
        dropout_rate: float = 0.1,
        # DPMM params
        dpmm_warmup_ratio: float = 0.6,
        dpmm_loss_weight: float = 1.0,
        dpmm_refit_interval: int = 10,
        n_components: int = 50,
        weight_concentration_prior: float = 1.0,
        dpmm_loss_type: Literal["nll", "kl", "energy", "student_t", "mmd", "soft_nll"] = "kl",
        student_t_df: float = 3.0,
        mmd_bandwidth: float = 1.0,
        model_name: str = "DPMMODEContrastive",
        use_bottleneck: bool = False,
        bottleneck_dim: int | None = None,
        use_vae: bool = False,
        kl_weight: float = 0.1,
        # Contrastive learning parameters
        use_moco: bool = True,
        moco_weight: float = 1.0,
        moco_embedding_dim: int = 128,
        moco_queue_size: int = 4096,
        moco_momentum: float = 0.999,
        moco_temperature: float = 0.2,
        # Augmentation parameters
        aug_noise_prob: float = 0.2,
        aug_noise_std: float = 0.1,
        aug_mask_prob: float = 0.1,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_dims or [],
            model_name=model_name,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ae = DPMMContrastiveAutoEncoder(
            input_dim=input_dim,
            encoder_dims=encoder_dims or [256, 128],
            latent_dim=latent_dim,
            decoder_dims=decoder_dims or [128, 256],
            norm=norm_type,
            drop=dropout_rate,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            use_vae=use_vae,
            use_moco=use_moco,
            moco_embedding_dim=moco_embedding_dim,
            moco_queue_size=moco_queue_size,
            moco_momentum=moco_momentum,
            moco_temperature=moco_temperature,
            aug_noise_prob=aug_noise_prob,
            aug_noise_std=aug_noise_std,
            aug_mask_prob=aug_mask_prob,
            device=self.device,
        )

        self.dpmm_warmup_ratio = dpmm_warmup_ratio
        self.dpmm_loss_weight = dpmm_loss_weight
        self.dpmm_refit_interval = dpmm_refit_interval
        self.n_components = n_components
        self.weight_concentration_prior = float(weight_concentration_prior)
        self.dpmm_loss_type = dpmm_loss_type
        self.student_t_df = student_t_df
        self.mmd_bandwidth = mmd_bandwidth
        self.dpmm_params = None
        self.dpmm_fitted = False
        self.recon_loss_fn = nn.MSELoss()
        self.use_vae = use_vae
        self.kl_weight = kl_weight

        # Safety guard: VAE Gaussian prior N(0,I) conflicts with DPMM clustering prior
        if self.use_vae and self.dpmm_loss_weight > 0:
            import warnings

            warnings.warn(
                f"{self.model_name}: use_vae=True with dpmm_loss_weight={self.dpmm_loss_weight} "
                f"creates conflicting KL objectives (Gaussian N(0,I) vs DPMM mixture). "
                f"Forcing dpmm_loss_weight=0.0 and dpmm_warmup_ratio=1.0.",
                UserWarning,
                stacklevel=2,
            )
            self.dpmm_loss_weight = 0.0
            self.dpmm_warmup_ratio = 1.0

        # Contrastive learning
        self.use_moco = use_moco
        self.moco_weight = moco_weight
        self.moco_loss_fn = nn.CrossEntropyLoss()

        # Track training phases

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to latent space with optional ODE."""
        if self.ae.use_vae:
            mu, var = self.ae.encode_vae(x)
            z = mu
        else:
            z = self.ae.encode_ae(x)
        return z

    def extract_latent(self, data_loader, device="cuda", return_reconstructions: bool = False):
        """Extract latent representations from DataLoader."""
        self.eval()
        self.to(device)
        latents, recons = [], []

        with torch.no_grad():
            for batch_data in data_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)

                z = self.encode(x, **batch_kwargs)

                latents.append(z.cpu().numpy())

                if return_reconstructions:
                    recons.append(self.decode(z).cpu().numpy())

        result = {"latent": np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result["reconstruction"] = np.concatenate(recons, axis=0)
        return result

    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode from latent space"""
        return self.ae.decoder(z)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass (Phase 1: AE/VAE + MoCo training without ODE)"""
        if self.ae.use_bottleneck:
            x_hat, z, x_le_hat, z_ld, mu, var, moco_logits, moco_labels, z_aug_q, z_aug_k = self.ae(
                x
            )
            recon = (x_hat, x_le_hat)
        else:
            x_hat, z, _, _, mu, var, moco_logits, moco_labels, z_aug_q, z_aug_k = self.ae(x)
            recon = (x_hat,)

        result = {"reconstruction": recon, "latent": z}
        if mu is not None:
            result["mu"] = mu
            result["var"] = var
        if moco_logits is not None:
            result["moco_logits"] = moco_logits
            result["moco_labels"] = moco_labels
        if z_aug_q is not None:
            result["z_aug_q"] = z_aug_q
            result["z_aug_k"] = z_aug_k
        return result

    def compute_loss(
        self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Compute loss: reconstruction + DPMM + contrastive (Phase 1)"""
        loss_dict = {}

        # Reconstruction loss
        if self.ae.use_bottleneck:
            x_hat, x_le_hat = outputs["reconstruction"]
            recon_ae = self.recon_loss_fn(x_hat, x)
            recon_bottleneck = self.recon_loss_fn(x_le_hat, x)
            recon = (recon_ae + recon_bottleneck) / 2.0
            loss_dict["recon_ae"] = recon_ae
            loss_dict["recon_bottleneck"] = recon_bottleneck
        else:
            (x_hat,) = outputs["reconstruction"]
            recon = self.recon_loss_fn(x_hat, x)
            loss_dict["recon_ae"] = recon

        loss_dict["recon_loss"] = recon

        # DPMM clustering loss
        dpmm = torch.tensor(0.0, device=x.device)
        if self.dpmm_fitted and self.dpmm_params is not None:
            if self.dpmm_loss_type == "nll":
                dpmm = self._dpmm_loss_nll(outputs["latent"])
            elif self.dpmm_loss_type == "kl":
                dpmm = self._dpmm_loss_kl(outputs["latent"])
            elif self.dpmm_loss_type == "energy":
                dpmm = self._dpmm_loss_energy(outputs["latent"])
            elif self.dpmm_loss_type == "student_t":
                dpmm = self._dpmm_loss_student_t(outputs["latent"])
            elif self.dpmm_loss_type == "mmd":
                dpmm = self._dpmm_loss_mmd(outputs["latent"])
            elif self.dpmm_loss_type == "soft_nll":
                dpmm = self._dpmm_loss_soft_nll(outputs["latent"])
        loss_dict["dpmm_loss"] = dpmm

        # VAE KL divergence loss
        kl_vae = torch.tensor(0.0, device=x.device)
        if self.use_vae and "mu" in outputs and "var" in outputs:
            kl_vae = self._kl_gaussian(outputs["mu"], outputs["var"])
            loss_dict["kl_vae"] = kl_vae

        # Contrastive loss (enhanced with symmetric and prototype loss)
        moco_loss = torch.tensor(0.0, device=x.device)
        sym_loss = torch.tensor(0.0, device=x.device)
        proto_loss = torch.tensor(0.0, device=x.device)

        if self.use_moco and "moco_logits" in outputs and outputs["moco_logits"] is not None:
            moco_loss = self.moco_loss_fn(outputs["moco_logits"], outputs["moco_labels"])
            loss_dict["moco_loss"] = moco_loss

            # Add symmetric contrastive loss (scAGCL-style)
            if "z_aug_q" in outputs and "z_aug_k" in outputs:
                sym_loss = self.ae.moco.symmetric_contrastive_loss(
                    outputs["z_aug_q"], outputs["z_aug_k"]
                )
                loss_dict["symmetric_cl_loss"] = sym_loss

            # Add prototype contrastive loss (scGPCL-style)
            if self.ae.moco.use_prototype:
                proto_loss = self.ae.moco.prototype_contrastive_loss(outputs["latent"])
                loss_dict["prototype_cl_loss"] = proto_loss

        # Total loss with all contrastive components
        contrastive_total = moco_loss + 0.5 * sym_loss + 0.3 * proto_loss
        total = (
            recon
            + self.dpmm_loss_weight * dpmm
            + self.kl_weight * kl_vae
            + self.moco_weight * contrastive_total
        )
        loss_dict["total_loss"] = total

        return loss_dict

    def _update_dpmm_params(self, bgm: BayesianGaussianMixture, device: torch.device):
        """Extract DPMM parameters from fitted sklearn model"""
        means = torch.as_tensor(bgm.means_, dtype=torch.float32, device=device)
        weight_concentration = torch.as_tensor(
            bgm.weight_concentration_, dtype=torch.float32, device=device
        )
        precisions_cholesky = torch.as_tensor(
            bgm.precisions_cholesky_, dtype=torch.float32, device=device
        )
        weights = torch.as_tensor(bgm.weights_, dtype=torch.float32, device=device)

        precisions_cholesky = torch.clamp(precisions_cholesky, min=1e-6, max=1e6)

        self.dpmm_params = {
            "means": means,
            "weight_concentration": weight_concentration,
            "precisions_cholesky": precisions_cholesky,
            "weights": weights,
            "covariances": 1.0 / (precisions_cholesky**2 + 1e-10),
        }
        self.dpmm_fitted = True

    def _dpmm_loss_kl(self, z: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under DPMM"""
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        weights = dp["weights"]
        log_weights = torch.log(weights + eps)
        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff**2 * precisions.unsqueeze(0), dim=2)

        log_gauss = -0.5 * (n_features * math.log(2.0 * math.pi) + mahalanobis) + log_det
        log_prob = torch.logsumexp(log_gauss + log_weights, dim=1)

        nll = -log_prob
        return torch.clamp(nll.mean(), min=0.0)

    def _dpmm_loss_nll(self, z: torch.Tensor) -> torch.Tensor:
        """NLL with stick-breaking weights"""
        return self._dpmm_loss_kl(z)

    def _dpmm_loss_energy(self, z: torch.Tensor) -> torch.Tensor:
        """Energy-based distance to nearest cluster"""
        dp = self.dpmm_params

        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        precisions = dp["precisions_cholesky"] ** 2
        weighted_dist = torch.sum(diff**2 * precisions.unsqueeze(0), dim=2)

        temperature = 1.0
        soft_assign = torch.softmax(-weighted_dist / temperature, dim=1)
        energy = torch.sum(soft_assign * weighted_dist, dim=1).mean()

        return energy

    def _dpmm_loss_student_t(self, z: torch.Tensor) -> torch.Tensor:
        """Student's t-distribution loss"""
        dp = self.dpmm_params
        n_features = z.size(1)
        df = self.student_t_df
        eps = 1e-10

        weights = dp["weights"]
        log_weights = torch.log(weights + eps)

        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        precisions = dp["precisions_cholesky"] ** 2
        mahalanobis = torch.sum(diff**2 * precisions.unsqueeze(0), dim=2)

        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)

        log_student_t = (
            torch.lgamma(torch.tensor((df + n_features) / 2.0, device=z.device))
            - torch.lgamma(torch.tensor(df / 2.0, device=z.device))
            - (n_features / 2.0) * math.log(df * math.pi)
            + log_det
            - ((df + n_features) / 2.0) * torch.log(1.0 + mahalanobis / df)
        )

        log_prob = torch.logsumexp(log_student_t + log_weights, dim=1)
        nll = -log_prob
        return torch.clamp(nll.mean(), min=0.0)

    def _dpmm_loss_mmd(self, z: torch.Tensor) -> torch.Tensor:
        """Maximum Mean Discrepancy"""
        dp = self.dpmm_params
        n_samples = z.size(0)

        component_samples = torch.multinomial(dp["weights"], n_samples, replacement=True)
        means = dp["means"][component_samples]
        stds = torch.sqrt(dp["covariances"][component_samples])
        dpmm_samples = means + stds * torch.randn_like(means)

        def rbf_kernel(x, y, bandwidth):
            xx = torch.sum(x**2, dim=1, keepdim=True)
            yy = torch.sum(y**2, dim=1, keepdim=True)
            xy = torch.mm(x, y.T)
            dists = xx + yy.T - 2 * xy
            return torch.exp(-dists / (2 * bandwidth**2))

        bandwidth = self.mmd_bandwidth
        k_xx = rbf_kernel(z, z, bandwidth).mean()
        k_yy = rbf_kernel(dpmm_samples, dpmm_samples, bandwidth).mean()
        k_xy = rbf_kernel(z, dpmm_samples, bandwidth).mean()

        mmd = k_xx - 2 * k_xy + k_yy
        return torch.relu(mmd)

    def _dpmm_loss_soft_nll(self, z: torch.Tensor) -> torch.Tensor:
        """Softened NLL"""
        dp = self.dpmm_params
        n_features = z.size(1)
        eps = 1e-10

        weights = dp["weights"]
        log_weights = torch.log(weights + eps)
        log_det = torch.sum(torch.log(dp["precisions_cholesky"] + eps), dim=1)
        precisions = dp["precisions_cholesky"] ** 2

        diff = z.unsqueeze(1) - dp["means"].unsqueeze(0)
        mahalanobis = torch.sum(diff**2 * precisions.unsqueeze(0), dim=2)

        log_gauss = -0.5 * (n_features * math.log(2.0 * math.pi) + mahalanobis) + log_det
        log_prob = torch.logsumexp(log_gauss + log_weights, dim=1)

        temperature = 2.0
        prob = torch.exp(log_prob / temperature)
        loss = ((1.0 - prob) ** 2).mean()

        return loss

    def _kl_gaussian(self, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """KL divergence for Gaussian"""
        kl_per_sample = 0.5 * torch.sum(var + mu**2 - 1.0 - torch.log(var + 1e-10), dim=1)
        return kl_per_sample.mean()

    def _refit_dpmm(self, train_loader, device: torch.device, verbose: int = 1) -> bool:
        """Refit DPMM on current latent representations"""
        self.eval()
        z_all = []

        with torch.no_grad():
            for batch in train_loader:
                x, _ = self._prepare_batch(batch, device)
                z = self.encode(x)
                z_all.append(z.cpu().numpy())

        z_all = np.concatenate(z_all, axis=0)
        z_all = z_all + np.random.normal(0, 1e-6, z_all.shape)

        try:
            bgm = BayesianGaussianMixture(
                n_components=self.n_components,
                weight_concentration_prior=self.weight_concentration_prior,
                mean_precision_prior=0.1,
                covariance_type="diag",
                init_params="kmeans",
                max_iter=100,
                tol=1e-3,
                random_state=42,
            )
            bgm.fit(z_all)

            if not np.isfinite(bgm.lower_bound_):
                if verbose >= 1:
                    print("Warning: DPMM fitting resulted in non-finite lower bound")
                return False

            self._update_dpmm_params(bgm, device=device)
            return True

        except Exception as e:
            if verbose >= 1:
                print(f"Warning: DPMM fitting failed with error: {e}")
            return False

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 500,
        lr: float = 1e-4,
        device: str = "cuda",
        save_path: str | None = None,
        patience: int = 20,
        verbose: int = 1,
        verbose_every: int = 1,
        weight_decay: float = 1e-5,
        **kwargs,
    ):
        """Phase 1: Train AE/VAE + MoCo with periodic DPMM refitting

        Args:
            weight_decay: AdamW weight decay (L2 regularization).
        """
        self.to(device)
        self.device = torch.device(device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        dpmm_warmup_epochs = int(epochs * self.dpmm_warmup_ratio)
        best_loss = float("inf")
        patience_counter = 0

        train_losses, recon_losses, dpmm_losses, moco_losses = [], [], [], []

        if verbose_every is None or verbose_every < 1:
            verbose_every = 1

        for epoch in range(epochs):
            if epoch < dpmm_warmup_epochs:
                self.dpmm_fitted = False
            elif (
                epoch == dpmm_warmup_epochs
                or (epoch - dpmm_warmup_epochs) % self.dpmm_refit_interval == 0
            ):
                if verbose >= 1 and ((epoch + 1) % verbose_every == 0 or epoch == 0):
                    print(f"Epoch {epoch + 1}: Refitting DPMM...")
                self._refit_dpmm(train_loader, torch.device(device), verbose=verbose)

            self.train()
            epoch_loss, epoch_recon, epoch_dpmm, epoch_moco = 0.0, 0.0, 0.0, 0.0
            n_batches = 0

            for batch in train_loader:
                x, batch_kwargs = self._prepare_batch(batch, device)

                optimizer.zero_grad()
                out = self.forward(x, **batch_kwargs, **kwargs)
                loss_dict = self.compute_loss(x, out, **batch_kwargs, **kwargs)
                loss = loss_dict["total_loss"]

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += loss_dict["recon_loss"].item()
                epoch_dpmm += loss_dict["dpmm_loss"].item()
                if "moco_loss" in loss_dict:
                    epoch_moco += loss_dict["moco_loss"].item()
                n_batches += 1

            if n_batches == 0:
                continue

            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_dpmm = epoch_dpmm / n_batches
            avg_moco = epoch_moco / n_batches

            train_losses.append(avg_loss)
            recon_losses.append(avg_recon)
            dpmm_losses.append(avg_dpmm)
            moco_losses.append(avg_moco)

            do_print = (verbose >= 1) and (
                ((epoch + 1) % verbose_every == 0) or (epoch == 0) or (epoch + 1 == epochs)
            )

            if do_print:
                phase = "Warmup" if epoch < dpmm_warmup_epochs else f"DPMM[{self.dpmm_loss_type}]"
                print(
                    f"Epoch {epoch + 1:3d}/{epochs} [Phase1-{phase}+MoCo] | "
                    f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | DPMM: {avg_dpmm:.4f} | MoCo: {avg_moco:.4f}"
                )

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                if save_path:
                    torch.save(self.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        return {
            "train_loss": train_losses,
            "recon_loss": recon_losses,
            "dpmm_loss": dpmm_losses,
            "moco_loss": moco_losses,
        }


def create_dpmmode_contrastive_model(
    input_dim: int, latent_dim: int = 32, **kwargs
) -> DPMMODEContrastiveModel:
    """
    Create DPMMODEContrastive model.

    Args:
        input_dim: Number of input features (genes)
        latent_dim: Latent space dimension (16-128)

    Critical Parameters:
        encoder_dims/decoder_dims: Network dimensions (default: [256,128]/[128,256])
        n_components: DPMM components (20-100)
        dpmm_loss_type: 'nll', 'kl', 'energy', 'student_t', 'mmd', 'soft_nll'
        use_moco: Enable MoCo contrastive learning (default: True)
        moco_weight: Contrastive loss weight (default: 1.0)
        moco_temperature: Contrastive temperature (default: 0.2)
    """
    return DPMMODEContrastiveModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
