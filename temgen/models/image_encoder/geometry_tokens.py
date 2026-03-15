"""
geometry_tokens.py

Geometry-aware token generation for the TEMGen image encoder.

Takes content tokens t_cont from CNNFrontend and tilt angles alpha,
and produces three additive embeddings:

    t_geo : (B, T, P, D)  — geometry-aware feature (reciprocal coords + Fourier)
    t_ang : (B, T, P, D)  — per-tilt angle embedding (broadcast over patches)
    Z     : (B, M, D)     — assembled token sequence, M = T*P
    q_coords : (B, M, 3)  — 3D reciprocal-space coords for each token (for Method 2)

Implements Sections A3–A7 of the CuAu 101010 Encoding Manual:

    A3. 2D reciprocal grid        — built lazily on first forward pass
    A4. 3D y-axis rotation        — alpha-dependent, computed per forward pass
    A5. Geometry feature vector   — 5 raw + 5×20 Fourier = 105 dim → Linear → 256
    A6. Angle embedding           — 20 Fourier bands → Linear → 256
    A7. Token assembly            — t_cont + t_geo + t_ang → view(B, M, D)

Key constants:
    T  = 15     tilts
    P  = dynamic  patches per tilt (e.g. 64 for 256×256 input, 169 for 409×409)
    M  = T*P    total tokens
    D  = 256    token dimension
    K  = 10     Fourier bands
    d_geo = 105   geometry feature dim  (5 raw + 5 × 2K)
    d_ang = 20    angle feature dim     (2K)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from temgen.utils import fourier_encode


# ─── Main module ──────────────────────────────────────────────────────────────

class GeometryTokens(nn.Module):
    """
    Geometry-aware token assembler (Sections A3–A7).

    Consumes:
        t_cont   : (B, T, P, D)   content tokens from CNNFrontend
        alpha    : (B, T)         tilt angles in radians

    Produces:
        Z        : (B, M, D)      assembled token sequence  (M = T*P, dynamic)
        q_coords : (B, M, 3)      3D reciprocal coords per token (for Method 2/3)

    Args:
        d_model  : token dimension D (default 256, fixed by spec)
        K        : Fourier frequency bands (default 10, fixed by spec)
    """

    # Fixed constants
    T : int = 15      # tilts
    D : int = 256     # token dimension
    K : int = 10      # Fourier bands

    # Derived dimensions
    # d_geo = 5 raw + 5 × 2K = 5 + 100 = 105
    # d_ang = 2K = 20

    def __init__(self, d_model: int = 256, K: int = 10):
        super().__init__()

        assert d_model == self.D, (
            f"d_model must be {self.D} to match the image encoder spec."
        )

        self._K = K
        d_geo = 5 + 5 * 2 * K    # 105
        d_ang = 2 * K             # 20

        # ── A3: 2D reciprocal grid — built lazily on first forward pass ───────
        # Grid size depends on input image resolution (8×8 for 256, 13×13 for 409)
        self._grid_h: int | None = None
        self._grid_w: int | None = None

        # ── A5: Geometry projection ───────────────────────────────────────────
        # f_geo: ℝ^105 → ℝ^256
        self.f_geo = nn.Sequential(
            nn.Linear(d_geo, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # ── A6: Angle projection ──────────────────────────────────────────────
        # f_ang: ℝ^20 → ℝ^256
        self.f_ang = nn.Linear(d_ang, d_model)

    def _ensure_grid(self, P: int, device: torch.device) -> None:
        """Build or rebuild the 2D reciprocal grid if patch count changed."""
        h = w = int(math.sqrt(P))
        assert h * w == P, f"Patch count {P} is not a perfect square"
        if self._grid_h == h and self._grid_w == w:
            return  # already built
        self._grid_h = h
        self._grid_w = w
        u = torch.linspace(-1, 1, h, device=device)
        v = torch.linspace(-1, 1, w, device=device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")
        self.register_buffer("q_tilde_x", grid_u.flatten())
        self.register_buffer("q_tilde_y", grid_v.flatten())

    # ── A4: y-axis 3D rotation ────────────────────────────────────────────────

    def _rotate_y(self, alpha: torch.Tensor, P: int) -> torch.Tensor:
        """
        Rotate 2D reciprocal-space coords into 3D via y-axis tilt.

        R_y(α): (q̃_x, q̃_y) → (q̃_x cos α,  q̃_y,  -q̃_x sin α)

        Args:
            alpha : (B, T)  tilt angles in radians
            P     : number of patches

        Returns:
            q_prime : (B, T, P, 3)  3D reciprocal coords per token
        """
        B, T = alpha.shape

        cos_a = torch.cos(alpha).unsqueeze(2)          # (B, T, 1)
        sin_a = torch.sin(alpha).unsqueeze(2)          # (B, T, 1)

        # q_tilde buffers: (P,) → (1, 1, P)
        qx = self.q_tilde_x.unsqueeze(0).unsqueeze(0) # (1, 1, P)
        qy = self.q_tilde_y.unsqueeze(0).unsqueeze(0) # (1, 1, P)

        qp_x = qx * cos_a                             # (B, T, P)
        qp_y = qy.expand(B, T, P)                     # (B, T, P)
        qp_z = -qx * sin_a                            # (B, T, P)

        q_prime = torch.stack([qp_x, qp_y, qp_z], dim=-1)  # (B, T, P, 3)
        return q_prime

    # ── A5: Geometry feature vector ───────────────────────────────────────────

    def _geometry_feature(
        self,
        alpha   : torch.Tensor,   # (B, T)
        q_prime : torch.Tensor,   # (B, T, P, 3)
    ) -> torch.Tensor:
        """
        Build γ_geo ∈ ℝ^105 for every (b, i, p) token.

        γ_geo = [q̃_x, q̃_y, q'_x, q'_y, q'_z,
                 φ(q̃_x), φ(q̃_y), φ(q'_x), φ(q'_y), φ(q'_z)]

        where φ(s) = [sin(πs), cos(πs), ..., sin(10πs), cos(10πs)] ∈ ℝ^20

        Returns:
            gamma_geo : (B, T, P, 105)
        """
        B, T = alpha.shape
        P = q_prime.shape[2]

        # Raw 2D coords — broadcast (P,) → (B, T, P)
        raw_x = self.q_tilde_x.view(1, 1, P).expand(B, T, -1)  # (B,T,P)
        raw_y = self.q_tilde_y.view(1, 1, P).expand(B, T, -1)  # (B,T,P)

        # Rotated 3D coords from q_prime
        raw_qp_x = q_prime[..., 0]                                   # (B,T,P)
        raw_qp_y = q_prime[..., 1]                                   # (B,T,P)
        raw_qp_z = q_prime[..., 2]                                   # (B,T,P)

        # Fourier features: each (B,T,P) → (B,T,P,20)
        phi_x   = fourier_encode(raw_x,   self._K)
        phi_y   = fourier_encode(raw_y,   self._K)
        phi_qpx = fourier_encode(raw_qp_x, self._K)
        phi_qpy = fourier_encode(raw_qp_y, self._K)
        phi_qpz = fourier_encode(raw_qp_z, self._K)

        # Assemble γ_geo: 5 raw scalars + 5 × 20 Fourier = 105
        gamma_geo = torch.cat([
            raw_x.unsqueeze(-1),    # (B,T,P,1)
            raw_y.unsqueeze(-1),    # (B,T,P,1)
            raw_qp_x.unsqueeze(-1),
            raw_qp_y.unsqueeze(-1),
            raw_qp_z.unsqueeze(-1),
            phi_x,                  # (B,T,P,20)
            phi_y,
            phi_qpx,
            phi_qpy,
            phi_qpz,
        ], dim=-1)                  # (B,T,P,105)

        return gamma_geo

    # ── A6: Angle embedding ───────────────────────────────────────────────────

    def _angle_feature(self, alpha: torch.Tensor, P: int) -> torch.Tensor:
        """
        Build γ_α ∈ ℝ^20 for every (b, i) tilt, then project → ℝ^256.

        γ_α = [sin(πα), cos(πα), ..., sin(10πα), cos(10πα)]

        Returns:
            t_ang : (B, T, P, D)  broadcast over patches
        """
        # alpha: (B, T) → Fourier encode → (B, T, 20)
        gamma_ang = fourier_encode(alpha, self._K)  # (B, T, 20)

        # Project → (B, T, 256)
        t_ang = self.f_ang(gamma_ang)               # (B, T, D)

        # Broadcast over patch dimension P
        t_ang = t_ang.unsqueeze(2).expand(-1, -1, P, -1)  # (B, T, P, D)
        return t_ang

    # ── A7: Forward / token assembly ──────────────────────────────────────────

    def forward(
        self,
        t_cont : torch.Tensor,   # (B, T, P, D)  from CNNFrontend
        alpha  : torch.Tensor,   # (B, T)         tilt angles in radians
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble geometry-aware token sequence.

        Returns:
            Z        : (B, M, D)   assembled tokens   M = T*P
            q_coords : (B, M, 3)   3D reciprocal coords (for Geometry-Aware Perceiver)
        """
        B, T, P, D = t_cont.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"
        assert D == self.D, f"Expected D={self.D}, got {D}"
        M = T * P

        # Build/rebuild spatial grid if patch count changed
        self._ensure_grid(P, t_cont.device)

        # A4: rotate to 3D
        q_prime = self._rotate_y(alpha, P)           # (B, T, P, 3)

        # A5: geometry tokens
        gamma_geo = self._geometry_feature(alpha, q_prime)  # (B, T, P, 105)
        t_geo = self.f_geo(gamma_geo)               # (B, T, P, D)

        # A6: angle tokens
        t_ang = self._angle_feature(alpha, P)       # (B, T, P, D)

        # A7: additive assembly
        t_tilde = t_cont + t_geo + t_ang            # (B, T, P, D)

        # Flatten T and P into a single token sequence
        Z        = t_tilde.view(B, M, D)            # (B, M, D)
        q_coords = q_prime.view(B, M, 3)            # (B, M, 3)

        return Z, q_coords

    def __repr__(self) -> str:
        d_geo = 5 + 5 * 2 * self._K
        d_ang = 2 * self._K
        n_params = sum(p.numel() for p in self.parameters())
        grid_str = f"{self._grid_h}×{self._grid_w}" if self._grid_h else "dynamic (built on first forward)"
        return (
            f"GeometryTokens(\n"
            f"  reciprocal_grid={grid_str}\n"
            f"  K={self._K} Fourier bands\n"
            f"  f_geo=Linear({d_geo}→{self.D})→GELU→Linear({self.D}→{self.D})\n"
            f"  f_ang=Linear({d_ang}→{self.D})\n"
            f"  output Z=(B, T*P, D={self.D})\n"
            f"  params={n_params:,}\n"
            f")"
        )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    import time

    B = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = GeometryTokens(d_model=256, K=10).to(device)
    print(model)
    print()

    # Simulate CNNFrontend output
    t_cont = torch.randn(B, 15, 169, 256, device=device)

    # Tilt angles: -7° to +7°, step 1° → radians
    tilt_deg = torch.arange(-7, 8, dtype=torch.float32)        # (15,)
    alpha = tilt_deg.mul(math.pi / 180).unsqueeze(0).expand(B, -1).to(device)

    t0 = time.time()
    with torch.no_grad():
        Z, q_coords = model(t_cont, alpha)
    elapsed = time.time() - t0

    print(f"t_cont   : {tuple(t_cont.shape)}")
    print(f"alpha    : {tuple(alpha.shape)}")
    print(f"Z        : {tuple(Z.shape)}      expected (B=2, M=2535, D=256)")
    print(f"q_coords : {tuple(q_coords.shape)}  expected (B=2, M=2535, 3)")

    assert Z.shape        == (B, 2535, 256), f"Z shape mismatch: {Z.shape}"
    assert q_coords.shape == (B, 2535, 3),   f"q_coords shape mismatch: {q_coords.shape}"

    # Check q_z range: at α = ±7° = ±0.1222 rad, |q'_z| = |q̃_x sin(α)| ≤ sin(7°) ≈ 0.122
    q_z_max = q_coords[..., 2].abs().max().item()
    print(f"\nq'_z max : {q_z_max:.4f}  (expected ≤ sin(7°) = 0.1219)")
    assert q_z_max <= 0.123, f"q_z out of range: {q_z_max}"

    # Check gamma_geo dimension would be 105
    d_geo = 5 + 5 * 2 * 10
    assert d_geo == 105, f"d_geo = {d_geo}"
    print(f"d_geo    : {d_geo}  (5 raw + 5×20 Fourier = 105) ✓")

    print(f"\nForward pass: {elapsed*1000:.1f} ms  (B={B})")
    print("\nAll assertions passed!")