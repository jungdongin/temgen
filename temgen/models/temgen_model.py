"""
temgen_model.py

Top-level TEMGen model.

Wires together:
    - Image encoder  : CNNFrontend → GeometryTokens → Aggregator (Method 1/2/3)
    - Structure encoder: GraphBuilder → StructureEncoder
    - InfoNCE loss   : on projected embeddings (z_TEM_proj, z_cell_proj)

All hyperparameters are read from an OmegaConf config object loaded from
configs/cuau_101010.yaml. No magic numbers live here.

Forward pass:
    batch = {
        "dp"        : (B, T, 1, H, W)   float32
        "alpha"     : (B, T)             float32  tilt angles in radians
        "frac_coords": list of (N_i, 3)
        "atom_types" : list of (N_i,)
        "lengths"    : (B, 3)
        "angles"     : (B, 3)
    }
    → z_TEM       : (B, D=256)   image latent
    → z_cell      : (B, D=256)   structure latent
    → z_TEM_proj  : (B, 128)     image projection  (InfoNCE)
    → z_cell_proj : (B, 128)     structure projection (InfoNCE)

Usage:
    from omegaconf import OmegaConf
    cfg   = OmegaConf.load("configs/cuau_101010.yaml")
    model = TEMGenModel(cfg)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .image_encoder import (
    CNNFrontend,
    GeometryTokens,
    PerceiverAggregator,
    GeometryAwarePerceiverAggregator,
    CrossViewVoxelAggregator,
)
from .structure_encoder import GraphBuilder, StructureEncoder


# ─── Aggregator registry ──────────────────────────────────────────────────────

AGGREGATOR_REGISTRY = {
    1: PerceiverAggregator,
    2: GeometryAwarePerceiverAggregator,
    3: CrossViewVoxelAggregator,
}


# ─── TEMGen model ─────────────────────────────────────────────────────────────

class TEMGenModel(nn.Module):
    """
    Full TEMGen contrastive learning model.

    Reads all hyperparameters from cfg (OmegaConf DictConfig).
    Selects the image encoder aggregator method via cfg.image_encoder.aggregator_method.

    Args:
        cfg : OmegaConf config loaded from configs/cuau_101010.yaml
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        ie_cfg   = cfg.image_encoder
        se_cfg   = cfg.structure_encoder
        g_cfg    = cfg.graph
        c_cfg    = cfg.contrastive

        # ── Image encoder ─────────────────────────────────────────────────────
        self.cnn_frontend = CNNFrontend(
            d_model    = ie_cfg.d_model,
            pretrained = ie_cfg.pretrained_cnn,
        )

        self.geometry_tokens = GeometryTokens(
            d_model = ie_cfg.d_model,
            K       = ie_cfg.K_fourier,
        )

        method = ie_cfg.aggregator_method
        assert method in AGGREGATOR_REGISTRY, (
            f"aggregator_method must be 1, 2, or 3. Got: {method}"
        )

        if method == 1:
            self.aggregator = PerceiverAggregator(
                d_model    = ie_cfg.d_model,
                n_latents  = ie_cfg.n_latents,
                n_heads    = ie_cfg.n_heads,
                L_cross    = ie_cfg.L_cross,
                L_self     = ie_cfg.L_self,
                d_proj_out = ie_cfg.d_proj,
            )

        elif method == 2:
            self.aggregator = GeometryAwarePerceiverAggregator(
                d_model    = ie_cfg.d_model,
                n_latents  = ie_cfg.n_latents,
                n_heads    = ie_cfg.n_heads,
                L_cross    = ie_cfg.L_cross,
                L_self     = ie_cfg.L_self,
                d_proj_out = ie_cfg.d_proj,
                beta_init  = ie_cfg.beta_init,
                sigma_init = ie_cfg.sigma_init,
            )

        elif method == 3:
            self.aggregator = CrossViewVoxelAggregator(
                d_model    = ie_cfg.d_model,
                n_heads    = ie_cfg.n_heads,
                L_blocks   = ie_cfg.L_blocks,
                K          = ie_cfg.K_fourier,
                d_proj_out = ie_cfg.d_proj,
            )

        # ── Structure encoder ─────────────────────────────────────────────────
        self.graph_builder = GraphBuilder(
            r_c               = g_cfg.r_c,
            K_rbf             = g_cfg.K_rbf,
            max_num_neighbors = g_cfg.max_num_neighbors,
        )

        self.structure_encoder = StructureEncoder(
            hidden_dim     = se_cfg.hidden_dim,
            n_layers       = se_cfg.n_layers,
            rbf_dim        = g_cfg.K_rbf,
            gram_dim       = 9,
            d_out          = se_cfg.d_out,
            d_proj         = se_cfg.d_proj,
            max_atomic_num = se_cfg.max_atomic_num,
        )

        # ── Learnable temperature for InfoNCE ─────────────────────────────────
        self.log_temp = nn.Parameter(
            torch.tensor(c_cfg.log_temp_init, dtype=torch.float32)
        )
        self.temp_min = c_cfg.temp_min
        self.temp_max = c_cfg.temp_max

        # Store method for forward routing
        self._aggregator_method = method

    # ── Image encoder forward ─────────────────────────────────────────────────

    def encode_image(
        self,
        dp   : torch.Tensor,   # (B, T, 1, H, W)
        alpha: torch.Tensor,   # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_TEM      : (B, 256)  image latent
            z_TEM_proj : (B, 128)  projected for InfoNCE
        """
        # CNN feature extraction
        t_cont = self.cnn_frontend(dp)                 # (B, T, P, D)

        # Geometry-aware token assembly
        Z, q_coords = self.geometry_tokens(t_cont, alpha)  # (B, M, D), (B, M, 3)

        # Aggregation (method-dependent)
        if self._aggregator_method == 1:
            z_TEM, z_TEM_proj = self.aggregator(Z, attn_bias=None)

        elif self._aggregator_method == 2:
            # Compute anchor positional bias then pass into aggregator
            attn_bias = self.aggregator._compute_attn_bias(q_coords)
            z_TEM, z_TEM_proj = self.aggregator(Z, attn_bias=attn_bias)

        elif self._aggregator_method == 3:
            # Method 3 doesn't use q_coords in aggregator (voxel grid is static)
            z_TEM, z_TEM_proj = self.aggregator(Z)

        return z_TEM, z_TEM_proj

    # ── Structure encoder forward ─────────────────────────────────────────────

    def encode_structure(
        self,
        frac_coords_list: list,   # list of (N_i, 3)
        atom_types_list : list,   # list of (N_i,)
        lengths_list    : list,   # list of (3,)
        angles_list     : list,   # list of (3,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_cell      : (B, 256)  structure latent
            z_cell_proj : (B, 128)  projected for InfoNCE
        """
        graph = self.graph_builder(
            frac_coords_list, atom_types_list,
            lengths_list, angles_list,
        )
        z_cell, z_cell_proj = self.structure_encoder(graph)
        return z_cell, z_cell_proj

    # ── Full forward ──────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch : dict from cuau_collate_fn with keys:
                dp           (B, T, 1, H, W)
                alpha        (B, T)
                frac_coords  list of (N_i, 3)
                atom_types   list of (N_i,)
                lengths      (B, 3)
                angles       (B, 3)

        Returns dict:
            z_TEM       : (B, 256)
            z_cell      : (B, 256)
            z_TEM_proj  : (B, 128)
            z_cell_proj : (B, 128)
            loss        : scalar  InfoNCE loss
            tau         : scalar  current temperature
        """
        dp    = batch["dp"]       # (B, T, 1, H, W)
        alpha = batch["alpha"]    # (B, T)
        B     = dp.shape[0]

        # Per-sample lists for structure encoder
        frac_coords_list = batch["frac_coords"]   # list of (N_i, 3)
        atom_types_list  = batch["atom_types"]    # list of (N_i,)

        # lengths/angles: (B, 3) → list of (3,) for GraphBuilder
        lengths_list = [batch["lengths"][i] for i in range(B)]
        angles_list  = [batch["angles"][i]  for i in range(B)]

        # Encode both modalities
        z_TEM,  z_TEM_proj  = self.encode_image(dp, alpha)
        z_cell, z_cell_proj = self.encode_structure(
            frac_coords_list, atom_types_list,
            lengths_list, angles_list,
        )

        # InfoNCE loss
        loss, tau = self.info_nce(z_TEM_proj, z_cell_proj)

        return dict(
            z_TEM       = z_TEM,
            z_cell      = z_cell,
            z_TEM_proj  = z_TEM_proj,
            z_cell_proj = z_cell_proj,
            loss        = loss,
            tau         = tau,
        )

    # ── InfoNCE loss ──────────────────────────────────────────────────────────

    def info_nce(
        self,
        z_tem : torch.Tensor,   # (B, 128)
        z_cell: torch.Tensor,   # (B, 128)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric InfoNCE loss with learnable temperature τ.

        L = -½ Σ_i [ log exp(sim(T_i,C_i)/τ) / Σ_j exp(sim(T_i,C_j)/τ)
                    + log exp(sim(C_i,T_i)/τ) / Σ_j exp(sim(C_i,T_j)/τ) ]

        Returns:
            loss : scalar
            tau  : current temperature value (for logging)
        """
        B   = z_tem.shape[0]
        tau = self.log_temp.exp().clamp(self.temp_min, self.temp_max)

        # L2 normalise
        z_tem  = F.normalize(z_tem,  dim=-1)   # (B, 128)
        z_cell = F.normalize(z_cell, dim=-1)   # (B, 128)

        # Similarity matrix
        logits = z_tem @ z_cell.T / tau        # (B, B)

        # Diagonal = positive pairs
        labels = torch.arange(B, device=logits.device)

        loss_tc = F.cross_entropy(logits,   labels)
        loss_ct = F.cross_entropy(logits.T, labels)
        loss    = (loss_tc + loss_ct) / 2.0

        return loss, tau.detach()

    def __repr__(self) -> str:
        method_name = {
            1: "PerceiverAggregator (Method 1)",
            2: "GeometryAwarePerceiverAggregator (Method 2)",
            3: "CrossViewVoxelAggregator (Method 3)",
        }[self._aggregator_method]

        n_img    = sum(p.numel() for p in list(self.cnn_frontend.parameters())
                                         + list(self.geometry_tokens.parameters())
                                         + list(self.aggregator.parameters()))
        n_struct = sum(p.numel() for p in list(self.structure_encoder.parameters()))
        n_total  = sum(p.numel() for p in self.parameters())

        return (
            f"TEMGenModel(\n"
            f"  image_encoder  : CNNFrontend → GeometryTokens → {method_name}\n"
            f"  struct_encoder : GraphBuilder → StructureEncoder\n"
            f"  loss           : symmetric InfoNCE  τ_init={self.log_temp.exp().item():.4f}\n"
            f"  params (image) : {n_img:,}\n"
            f"  params (struct): {n_struct:,}\n"
            f"  params (total) : {n_total:,}\n"
            f")"
        )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from omegaconf import OmegaConf

    # Load config
    cfg = OmegaConf.load("configs/cuau_101010.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test all three aggregator methods
    for method in [1, 2, 3]:
        cfg.image_encoder.aggregator_method = method
        model = TEMGenModel(cfg).to(device)

        if method == 1:
            print(model)
            print()

        B, T, H, W = 2, 15, 409, 409
        N = 640

        # Mock image batch
        dp    = torch.randn(B, T, 1, H, W, device=device)
        import math
        tilt_deg = torch.arange(-7, 8, dtype=torch.float32)
        alpha = tilt_deg.mul(math.pi / 180).unsqueeze(0).expand(B, -1).to(device)

        # Mock structure batch
        frac_coords_list = [torch.rand(N, 3, device=device) for _ in range(B)]
        atom_types_list  = [
            torch.where(torch.rand(N, device=device) > 0.5,
                        torch.full((N,), 29, device=device),
                        torch.full((N,), 79, device=device))
            for _ in range(B)
        ]
        lengths_list = [torch.tensor([15.0, 15.0, 50.0], device=device)] * B
        angles_list  = [torch.tensor([90.0, 90.0, 90.0], device=device)] * B

        batch = dict(
            dp          = dp,
            alpha       = alpha,
            frac_coords = frac_coords_list,
            atom_types  = atom_types_list,
            lengths     = torch.stack(lengths_list),
            angles      = torch.stack(angles_list),
        )

        with torch.no_grad():
            out = model(batch)

        print(f"Method {method}: loss={out['loss'].item():.4f}  "
              f"τ={out['tau'].item():.4f}  "
              f"z_TEM={tuple(out['z_TEM'].shape)}  "
              f"z_cell={tuple(out['z_cell'].shape)}")

        assert out["z_TEM"].shape       == (B, 256)
        assert out["z_cell"].shape      == (B, 256)
        assert out["z_TEM_proj"].shape  == (B, 128)
        assert out["z_cell_proj"].shape == (B, 128)

    print("\nAll methods passed!")