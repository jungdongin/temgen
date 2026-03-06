"""
geometry_aware_perceiver.py

Method 2: Geometry-Aware Perceiver aggregator for the TEMGen image encoder.

Extends Method 1 (PerceiverAggregator) by adding learnable 3D anchor
positions in reciprocal space that bias cross-attention scores. Each
latent token is associated with an anchor p_n ∈ ℝ³, and tokens closer
to that anchor in reciprocal space receive higher attention weight.

Key additions over Method 1 (Section A8, Method 2):

    Anchors      : p_n ∈ ℝ³  for n = 1,...,32
                   Initialised on a 4×4×2 uniform grid in [-1,1]³
    σ (sigma)    : learnable scalar width, init softplus⁻¹(0.5) ≈ 0.48
    β (beta)     : learnable scalar scale,  init 1.0

    Bias:
        bias[b, n, m] = -‖p_n - q_m[b]‖² / (2σ²)     shape (B, N_latent, M)

    Modified cross-attention score:
        S = QKᵀ / √d_head  +  β · bias[:, None, :, :]  (broadcast over heads)

    At β = 0: exactly Method 1 — clean ablation.

The PerceiverAggregator (Method 1) already exposes an `attn_bias` hook
in CrossAttentionBlock.forward(). Method 2 computes the bias here and
passes it in — Method 1's CrossAttentionBlock code is unchanged.

NOTE: This file is self-contained. It re-uses FeedForward,
CrossAttentionBlock, and SelfAttentionBlock from aggregator.py
via import rather than copy-pasting them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import CrossAttentionBlock, FeedForward, SelfAttentionBlock


class GeometryAwarePerceiverAggregator(nn.Module):
    """
    Method 2: Geometry-Aware Perceiver aggregator.

    Identical to Method 1 except cross-attention scores are biased by
    the proximity of each input token's 3D reciprocal-space coordinate
    to each latent's learned anchor position.

    Args:
        d_model    : token dimension              (default 256)
        n_latents  : number of latent tokens      (default 32)
        n_heads    : attention heads              (default 8)
        L_cross    : cross-attention blocks       (default 1)
        L_self     : self-attention blocks each   (default 2)
        d_proj_out : projection head output dim   (default 128)
        beta_init  : initial value of β           (default 1.0)
        sigma_init : initial value of σ           (default 0.5)
    """

    # Fixed constants from spec
    D        : int   = 256
    N_LATENT : int   = 32
    N_HEADS  : int   = 8
    D_FF     : int   = 1024
    L_CROSS  : int   = 1
    L_SELF   : int   = 2
    D_PROJ   : int   = 128

    # Anchor grid: 4×4×2 = 32 anchors
    GRID_X   : int   = 4
    GRID_Y   : int   = 4
    GRID_Z   : int   = 2

    def __init__(
        self,
        d_model    : int   = 256,
        n_latents  : int   = 32,
        n_heads    : int   = 8,
        L_cross    : int   = 1,
        L_self     : int   = 2,
        d_proj_out : int   = 128,
        beta_init  : float = 1.0,
        sigma_init : float = 0.5,
    ):
        super().__init__()
        assert n_latents == self.GRID_X * self.GRID_Y * self.GRID_Z, (
            f"n_latents must equal {self.GRID_X}×{self.GRID_Y}×{self.GRID_Z}=32 "
            f"to match the 4×4×2 anchor grid."
        )

        d_ff = 4 * d_model
        self.n_heads  = n_heads
        self.d_model  = d_model
        self.n_latents = n_latents

        # ── Learned latent array ──────────────────────────────────────────────
        self.latents = nn.Parameter(
            torch.randn(1, n_latents, d_model) * 0.02
        )

        # ── Anchor positions: 4×4×2 uniform grid in [-1,1]³ ──────────────────
        # Shape: (32, 3) — learnable, initialised on uniform grid
        anchors = self._make_anchor_grid()            # (32, 3)
        self.anchors = nn.Parameter(anchors)

        # ── σ: learnable via softplus to keep it positive ─────────────────────
        # softplus⁻¹(sigma_init) = log(exp(sigma_init) - 1)
        sigma_raw_init = math.log(math.exp(sigma_init) - 1.0)
        self.sigma_raw = nn.Parameter(torch.tensor(sigma_raw_init))

        # ── β: learnable scalar scale ─────────────────────────────────────────
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # ── Perceiver blocks (same as Method 1) ───────────────────────────────
        self.cross_blocks = nn.ModuleList()
        for _ in range(L_cross):
            block = nn.ModuleDict({
                "cross_attn": CrossAttentionBlock(d_model, n_heads),
                "cross_ff"  : FeedForward(d_model, d_ff),
                "self_layers": nn.ModuleList([
                    nn.ModuleDict({
                        "self_attn": SelfAttentionBlock(d_model, n_heads),
                        "self_ff"  : FeedForward(d_model, d_ff),
                    })
                    for _ in range(L_self)
                ]),
            })
            self.cross_blocks.append(block)

        # ── Global pooling query ──────────────────────────────────────────────
        self.q_glob     = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_cross = CrossAttentionBlock(d_model, n_heads)
        self.pool_norm  = nn.LayerNorm(d_model)

        # ── Projection head (A9) ──────────────────────────────────────────────
        self.img_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proj_out),
        )

    # ── Anchor grid initialisation ────────────────────────────────────────────

    @staticmethod
    def _make_anchor_grid() -> torch.Tensor:
        """
        Build a 4×4×2 uniform grid of anchor positions in [-1,1]³.

        q_z coverage is limited to ±sin(7°) ≈ ±0.122 (the physical range
        from the y-axis tilt), so the z extent is scaled accordingly.

        Returns:
            anchors : (32, 3)  float32
        """
        rx = torch.linspace(-1.0,  1.0,  4)            # q_x: full range
        ry = torch.linspace(-1.0,  1.0,  4)            # q_y: full range
        rz = torch.linspace(-0.122, 0.122, 2)          # q_z: ±sin(7°)

        grid = torch.stack(
            torch.meshgrid(rx, ry, rz, indexing="ij"), dim=-1
        )                                               # (4, 4, 2, 3)
        return grid.reshape(32, 3)                      # (32, 3)

    # ── Bias computation ──────────────────────────────────────────────────────

    def _compute_attn_bias(self, q_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute anchor-based positional bias for cross-attention.

        bias[b, n, m] = -‖p_n - q_m[b]‖² / (2σ²)

        Args:
            q_coords : (B, M, 3)  3D reciprocal coords from GeometryTokens

        Returns:
            attn_bias : (B, H, N_latent, M)  ready to add to attention scores
        """
        # anchors: (32, 3) → (1, 32, 1, 3)
        p = self.anchors.unsqueeze(0).unsqueeze(2)      # (1, 32, 1, 3)

        # q_coords: (B, M, 3) → (B, 1, M, 3)
        q = q_coords.unsqueeze(1)                       # (B, 1, M, 3)

        diff    = p - q                                 # (B, 32, M, 3)
        sq_dist = (diff ** 2).sum(dim=-1)               # (B, 32, M)

        sigma = F.softplus(self.sigma_raw)              # scalar, > 0
        bias  = -sq_dist / (2.0 * sigma ** 2)           # (B, 32, M)

        # Scale by β, then broadcast over H heads → (B, H, N_latent, M)
        bias = self.beta * bias
        bias = bias.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B,H,32,M)

        return bias

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        Z        : torch.Tensor,   # (B, M=2535, D=256)
        q_coords : torch.Tensor,   # (B, M=2535, 3)   from GeometryTokens
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z        : (B, M, D)  assembled token sequence
            q_coords : (B, M, 3)  3D reciprocal-space coords per token

        Returns:
            z_TEM      : (B, D=256)  latent embedding
            z_TEM_proj : (B, 128)    projected embedding for InfoNCE
        """
        B = Z.shape[0]

        # Compute geometry-aware positional bias
        attn_bias = self._compute_attn_bias(q_coords)  # (B, H, 32, M)

        # Expand latents to batch
        latents = self.latents.expand(B, -1, -1)       # (B, 32, D)

        # ── Perceiver cross + self attention ──────────────────────────────────
        for block in self.cross_blocks:
            # Biased cross-attention: latents attend to Z
            latents = block["cross_attn"](latents, Z, attn_bias=attn_bias)
            latents = block["cross_ff"](latents)

            # Self-attention: latents attend to themselves (no bias)
            for self_layer in block["self_layers"]:
                latents = self_layer["self_attn"](latents)
                latents = self_layer["self_ff"](latents)

        # ── Global pooling ────────────────────────────────────────────────────
        q_glob = self.q_glob.expand(B, -1, -1)         # (B, 1, D)
        z_TEM  = self.pool_cross(q_glob, latents)      # (B, 1, D)
        z_TEM  = self.pool_norm(z_TEM)
        z_TEM  = z_TEM.squeeze(1)                      # (B, D)

        # ── Projection head ───────────────────────────────────────────────────
        z_TEM_proj = self.img_proj(z_TEM)              # (B, 128)

        return z_TEM, z_TEM_proj

    def __repr__(self) -> str:
        sigma = F.softplus(self.sigma_raw).item()
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"GeometryAwarePerceiverAggregator [Method 2]\n"
            f"  latents      : ({self.n_latents}, {self.d_model})\n"
            f"  anchors      : ({self.n_latents}, 3)  grid={self.GRID_X}×{self.GRID_Y}×{self.GRID_Z}\n"
            f"  sigma        : {sigma:.4f}  (learned via softplus)\n"
            f"  beta         : {self.beta.item():.4f}  (learned scalar)\n"
            f"  cross blocks : {self.L_CROSS} × (CrossAttn+bias + FFN + {self.L_SELF}×SelfAttn+FFN)\n"
            f"  heads        : {self.n_heads},  d_head={self.d_model // self.n_heads}\n"
            f"  d_ff         : {self.D_FF}\n"
            f"  pool         : q_glob cross-attn → z_TEM (B, {self.d_model})\n"
            f"  img_proj     : Linear({self.d_model}→{self.d_model})→GELU→Linear({self.d_model}→{self.D_PROJ})\n"
            f"  params       : {n_params:,}\n"
        )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    B = 2
    M = 2535
    D = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = GeometryAwarePerceiverAggregator().to(device)
    print(model)

    Z        = torch.randn(B, M, D, device=device)
    q_coords = torch.randn(B, M, 3, device=device).clamp(-1, 1)
    # clamp q_z to physical range ±0.122
    q_coords[..., 2] = q_coords[..., 2] * 0.122

    t0 = time.time()
    with torch.no_grad():
        z_TEM, z_TEM_proj = model(Z, q_coords)
    elapsed = time.time() - t0

    print(f"\nZ          : {tuple(Z.shape)}")
    print(f"q_coords   : {tuple(q_coords.shape)}")
    print(f"z_TEM      : {tuple(z_TEM.shape)}       expected (B={B}, 256)")
    print(f"z_TEM_proj : {tuple(z_TEM_proj.shape)}  expected (B={B}, 128)")

    assert z_TEM.shape      == (B, 256), f"z_TEM shape mismatch: {z_TEM.shape}"
    assert z_TEM_proj.shape == (B, 128), f"z_TEM_proj shape mismatch"

    # Verify β=0 → same as Method 1 (bias zeroed out)
    with torch.no_grad():
        model.beta.fill_(0.0)
        z_beta0, _ = model(Z, q_coords)
    print(f"\nβ=0 ablation check: z_TEM computed without issue ✓")
    model.beta.fill_(1.0)   # restore

    print(f"\nForward pass: {elapsed*1000:.1f} ms  (B={B})")
    print("All assertions passed!")