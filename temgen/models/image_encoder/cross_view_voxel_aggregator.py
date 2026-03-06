"""
cross_view_voxel_aggregator.py

Method 3: Cross-View Reciprocal Voxel Latent aggregator for the TEMGen
image encoder.

Replaces the Perceiver latent array with a structured voxel grid in 3D
reciprocal space. Each voxel is a learnable query positioned at a fixed
coordinate in (q_x, q_y, q_z) space, and attends to all input tokens Z
via cross-attention, then refines through self-attention.

Architecture (Section A8, Method 3 of CuAu 101010 Encoding Manual):

    Voxel grid   : (N_x, N_y, N_z) = (8, 8, 2)  →  J = 128 voxels
    Voxel coords : q_x ∈ linspace(-1, 1, 8)
                   q_y ∈ linspace(-1, 1, 8)
                   q_z ∈ linspace(-0.122, 0.122, 2)   [±sin(7°)]
    Voxel pos enc: 3 × (1 raw + 2K Fourier) = 3 × 21 = 63  →  Linear(63, 256)
    Voxel tokens : v_j = f_voxel(pos_enc(c_j)) ∈ ℝ^256  — FIXED (no Parameter)
                   (positional structure is the identity; tokens come purely
                    from pos encoding, not a learned embedding table)

    Per block (L_blocks = 2):
        1. Cross-attention: voxels (Q) attend to Z (K, V)
        2. FFN on voxels
        3. Self-attention: voxels attend to each other
        4. FFN on voxels

    Global pooling:
        Learned global query q_glob (1, 256) cross-attends
        to final voxel tokens → z_TEM (B, 256)

    Projection head (A9):
        z_TEM → Linear(256,256) → GELU → Linear(256,128) → z_TEM_proj (B, 128)

Attention config:
    H_attn = 8 heads,  d_head = 256/8 = 32
    FFN hidden = 4 × D = 1024

Cross-attention cost : 128 × 2535 = 324K per head
Self-attention cost  : 128²       =  16K per head
(More expensive than Method 1/2 but captures full 3D voxel structure)

NOTE: Reuses FeedForward, CrossAttentionBlock, SelfAttentionBlock
from aggregator.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import CrossAttentionBlock, FeedForward, SelfAttentionBlock


# ─── Fourier positional encoding (same as geometry_tokens.py) ─────────────────

def fourier_encode(x: torch.Tensor, K: int = 10) -> torch.Tensor:
    """
    φ(x) = [sin(πx), cos(πx), ..., sin(Kπx), cos(Kπx)] ∈ ℝ^(2K)

    Args:
        x   : (...,)
        K   : frequency bands

    Returns:
        enc : (..., 2K)
    """
    freqs  = torch.arange(1, K + 1, device=x.device, dtype=x.dtype)
    angles = x.unsqueeze(-1) * math.pi * freqs       # (..., K)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ─── Voxel positional encoding ────────────────────────────────────────────────

def voxel_pos_enc(coords: torch.Tensor, K: int = 10) -> torch.Tensor:
    """
    Build positional encoding for 3D voxel coordinates.

    For each coordinate dimension c ∈ {x, y, z}:
        enc(c) = [c, sin(πc), cos(πc), ..., sin(Kπc), cos(Kπc)]  ∈ ℝ^(1+2K)

    Concatenated over 3 dims: ℝ^(3 × (1+2K)) = ℝ^63 at K=10.

    Args:
        coords : (J, 3)   voxel coordinates
        K      : Fourier bands (default 10)

    Returns:
        enc : (J, 3*(1+2K))  = (J, 63)
    """
    parts = []
    for d in range(3):
        c   = coords[:, d]                            # (J,)
        enc = torch.cat([
            c.unsqueeze(-1),                          # (J, 1) raw
            fourier_encode(c, K),                     # (J, 2K)
        ], dim=-1)                                    # (J, 1+2K)
        parts.append(enc)
    return torch.cat(parts, dim=-1)                   # (J, 3*(1+2K))


# ─── Main aggregator ──────────────────────────────────────────────────────────

class CrossViewVoxelAggregator(nn.Module):
    """
    Method 3: Cross-View Reciprocal Voxel Latent aggregator.

    Uses a structured 8×8×2 voxel grid in reciprocal space as queries,
    instead of unstructured learned latent tokens (Method 1/2).

    Args:
        d_model    : token dimension              (default 256)
        n_heads    : attention heads              (default 8)
        L_blocks   : number of cross+self blocks  (default 2)
        K          : Fourier bands for pos enc    (default 10)
        d_proj_out : projection head output dim   (default 128)
    """

    # Fixed constants from spec
    D      : int = 256
    N_HEADS: int = 8
    L_BLOCKS: int = 2
    K      : int = 10
    D_PROJ : int = 128
    D_FF   : int = 1024

    # Voxel grid
    NX: int = 8
    NY: int = 8
    NZ: int = 2
    J : int = 128   # NX * NY * NZ

    # Positional encoding dim: 3 × (1 + 2K) = 63
    D_POS: int = 63

    def __init__(
        self,
        d_model    : int = 256,
        n_heads    : int = 8,
        L_blocks   : int = 2,
        K          : int = 10,
        d_proj_out : int = 128,
    ):
        super().__init__()
        d_ff = 4 * d_model
        d_pos = 3 * (1 + 2 * K)                      # 63 at K=10

        assert d_pos == self.D_POS, f"d_pos mismatch: {d_pos} vs {self.D_POS}"

        self.d_model = d_model
        self.n_heads = n_heads

        # ── Voxel coordinate grid (static buffer) ─────────────────────────────
        # q_z range: ±sin(7°) ≈ ±0.122  (physical max from y-axis tilt)
        rx = torch.linspace(-1.0,   1.0,   self.NX)
        ry = torch.linspace(-1.0,   1.0,   self.NY)
        rz = torch.linspace(-0.122, 0.122, self.NZ)
        grid = torch.stack(
            torch.meshgrid(rx, ry, rz, indexing="ij"), dim=-1
        )                                              # (8, 8, 2, 3)
        voxel_coords = grid.reshape(self.J, 3)         # (128, 3)
        self.register_buffer("voxel_coords", voxel_coords)

        # ── Voxel positional encoding projection ──────────────────────────────
        # f_voxel: ℝ^63 → ℝ^256
        self.f_voxel = nn.Linear(d_pos, d_model)

        # ── Attention blocks ──────────────────────────────────────────────────
        # L_blocks × (CrossAttn + FFN + SelfAttn + FFN)
        self.blocks = nn.ModuleList()
        for _ in range(L_blocks):
            block = nn.ModuleDict({
                "cross_attn": CrossAttentionBlock(d_model, n_heads),
                "cross_ff"  : FeedForward(d_model, d_ff),
                "self_attn" : SelfAttentionBlock(d_model, n_heads),
                "self_ff"   : FeedForward(d_model, d_ff),
            })
            self.blocks.append(block)

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

    def _voxel_tokens(self, B: int) -> torch.Tensor:
        """
        Build voxel query tokens from positional encoding.

        Returns:
            v : (B, J=128, D=256)
        """
        # voxel_coords: (J, 3)  [static buffer]
        pos_enc = voxel_pos_enc(self.voxel_coords, K=self.K)  # (J, 63)
        v = self.f_voxel(pos_enc)                              # (J, D)
        v = v.unsqueeze(0).expand(B, -1, -1)                  # (B, J, D)
        return v

    def forward(
        self,
        Z : torch.Tensor,    # (B, M=2535, D=256)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z : (B, M, D)  assembled token sequence from GeometryTokens

        Returns:
            z_TEM      : (B, D=256)  latent embedding
            z_TEM_proj : (B, 128)    projected embedding for InfoNCE
        """
        B = Z.shape[0]

        # Build voxel query tokens from positional encoding
        v = self._voxel_tokens(B)                      # (B, 128, D)

        # ── Cross + self attention blocks ─────────────────────────────────────
        for block in self.blocks:
            # Cross-attention: voxels attend to full token sequence Z
            v = block["cross_attn"](v, Z)              # (B, 128, D)
            v = block["cross_ff"](v)

            # Self-attention: voxels attend to each other
            v = block["self_attn"](v)                  # (B, 128, D)
            v = block["self_ff"](v)

        # ── Global pooling: q_glob cross-attends to voxel tokens ──────────────
        q_glob = self.q_glob.expand(B, -1, -1)         # (B, 1, D)
        z_TEM  = self.pool_cross(q_glob, v)            # (B, 1, D)
        z_TEM  = self.pool_norm(z_TEM)
        z_TEM  = z_TEM.squeeze(1)                      # (B, D)

        # ── Projection head ───────────────────────────────────────────────────
        z_TEM_proj = self.img_proj(z_TEM)              # (B, 128)

        return z_TEM, z_TEM_proj

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"CrossViewVoxelAggregator [Method 3]\n"
            f"  voxel grid   : {self.NX}×{self.NY}×{self.NZ} = {self.J} voxels\n"
            f"  voxel coords : q_x∈[-1,1], q_y∈[-1,1], q_z∈[-0.122,0.122]\n"
            f"  pos enc dim  : {self.D_POS}  (3×(1+2×{self.K}))\n"
            f"  f_voxel      : Linear({self.D_POS}→{self.d_model})\n"
            f"  blocks       : {self.L_BLOCKS} × (CrossAttn + FFN + SelfAttn + FFN)\n"
            f"  heads        : {self.n_heads},  d_head={self.d_model // self.n_heads}\n"
            f"  d_ff         : {self.D_FF}\n"
            f"  cross cost   : {self.J}×2535 = {self.J*2535:,} per head\n"
            f"  self  cost   : {self.J}² = {self.J**2:,} per head\n"
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

    model = CrossViewVoxelAggregator().to(device)
    print(model)

    Z = torch.randn(B, M, D, device=device)

    t0 = time.time()
    with torch.no_grad():
        z_TEM, z_TEM_proj = model(Z)
    elapsed = time.time() - t0

    print(f"\nZ          : {tuple(Z.shape)}")
    print(f"z_TEM      : {tuple(z_TEM.shape)}       expected (B={B}, 256)")
    print(f"z_TEM_proj : {tuple(z_TEM_proj.shape)}  expected (B={B}, 128)")

    assert z_TEM.shape      == (B, 256), f"z_TEM shape mismatch"
    assert z_TEM_proj.shape == (B, 128), f"z_TEM_proj shape mismatch"

    # Verify voxel grid shape and q_z range
    vc = model.voxel_coords
    assert vc.shape == (128, 3), f"voxel_coords shape mismatch: {vc.shape}"
    assert vc[:, 2].abs().max().item() <= 0.123, "q_z out of physical range"
    print(f"\nvoxel_coords : {tuple(vc.shape)}  q_z ∈ [{vc[:,2].min():.3f}, {vc[:,2].max():.3f}] ✓")

    # Verify pos enc dim
    from cross_view_voxel_aggregator import voxel_pos_enc
    pe = voxel_pos_enc(vc)
    assert pe.shape == (128, 63), f"pos enc dim mismatch: {pe.shape}"
    print(f"pos enc      : {tuple(pe.shape)}  (128, 63=3×21) ✓")

    print(f"\nForward pass: {elapsed*1000:.1f} ms  (B={B})")
    print("All assertions passed!")