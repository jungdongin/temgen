"""
aggregator.py

Method 1: Perceiver Latent aggregator for the TEMGen image encoder.

Takes the assembled token sequence Z from GeometryTokens and compresses
it into a single fixed-size latent vector z_TEM via cross-attention
from learned latent tokens, followed by self-attention refinement,
and a final global pooling step.

Architecture (Section A8, Method 1 of CuAu 101010 Encoding Manual):

    Latent tokens  : N_latent = 32  learnable queries  (B, 32, 256)
    Input tokens   : Z              (B, M=2535, 256)

    For each of L_cross=1 cross-attention blocks:
        1. Cross-attention: latents (Q) attend to Z (K, V)
        2. FFN on latents
        3. L_self=2 × Self-attention blocks on latents
            each: self-attn → FFN

    Global pooling:
        Learned global query q_glob (1, 256) cross-attends
        to final latent L^(L) (B, 32, 256) → z_TEM (B, 256)

    Projection head (A9):
        z_TEM → Linear(256,256) → GELU → Linear(256,128)
        → z_TEM_proj (B, 128)  used for InfoNCE only

Attention config:
    H_attn = 8 heads,  d_head = 256 / 8 = 32
    FFN hidden = 4 × D = 1024

NOTE: This is Method 1 (baseline Perceiver). Method 2 (Geometry-Aware
Perceiver) will extend this by adding anchor-based positional bias to
the cross-attention scores. Method 3 (Voxel Latent) replaces the
Perceiver entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Submodules ───────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Standard Transformer FFN: Linear → GELU → Linear, with pre-norm.

    Pre-LayerNorm is applied inside this module so callers don't need to.
    Residual connection is applied here too.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block with pre-norm and residual.

    Q comes from latents, K and V come from context (token sequence Z).
    Residual is applied to Q (latents) only.

    Args:
        d_model   : token dimension (256)
        n_heads   : number of attention heads (8)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q : torch.Tensor,          # (B, N_q, D)   latents
        kv: torch.Tensor,          # (B, N_kv, D)  context tokens
        attn_bias: torch.Tensor | None = None,  # (B, H, N_q, N_kv) optional
    ) -> torch.Tensor:
        """
        Returns updated q of shape (B, N_q, D).
        """
        B, N_q, D  = q.shape
        _,  N_kv, _ = kv.shape
        H  = self.n_heads
        dh = self.d_head

        # Pre-norm
        q_n  = self.norm_q(q)
        kv_n = self.norm_kv(kv)

        # Project and reshape to (B, H, N, dh)
        Q = self.q_proj(q_n).view(B, N_q,  H, dh).transpose(1, 2)   # (B,H,N_q,dh)
        K = self.k_proj(kv_n).view(B, N_kv, H, dh).transpose(1, 2)  # (B,H,N_kv,dh)
        V = self.v_proj(kv_n).view(B, N_kv, H, dh).transpose(1, 2)  # (B,H,N_kv,dh)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale    # (B,H,N_q,N_kv)

        # Optional positional bias (used by Method 2)
        if attn_bias is not None:
            scores = scores + attn_bias

        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, V)                                 # (B,H,N_q,dh)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)        # (B,N_q,D)
        out = self.out_proj(out)

        # Residual on Q
        return q + out


class SelfAttentionBlock(nn.Module):
    """
    Single self-attention block with pre-norm and residual.

    Args:
        d_model : token dimension (256)
        n_heads : number of attention heads (8)
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5

        self.norm    = nn.LayerNorm(d_model)
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D)"""
        B, N, D  = x.shape
        H  = self.n_heads
        dh = self.d_head

        x_n = self.norm(x)
        QKV = self.qkv(x_n).view(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV.unbind(0)                                        # (B,H,N,dh) each

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, V)                                 # (B,H,N,dh)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return x + out


# ─── Main aggregator ──────────────────────────────────────────────────────────

class PerceiverAggregator(nn.Module):
    """
    Method 1: Perceiver Latent aggregator.

    Compresses M=2535 token sequence Z → single vector z_TEM (B, 256),
    then projects to z_TEM_proj (B, 128) for InfoNCE loss.

    Args:
        d_model    : token dimension (default 256, fixed by spec)
        n_latents  : number of latent tokens (default 32, fixed by spec)
        n_heads    : attention heads (default 8, fixed by spec)
        L_cross    : number of cross-attention blocks (default 1)
        L_self     : number of self-attention blocks per cross block (default 2)
        d_proj_out : projection head output dim for InfoNCE (default 128)
    """

    # Fixed constants from spec
    D        : int = 256
    N_LATENT : int = 32
    N_HEADS  : int = 8
    D_FF     : int = 1024   # 4 × D
    L_CROSS  : int = 1
    L_SELF   : int = 2
    D_PROJ   : int = 128

    def __init__(
        self,
        d_model    : int = 256,
        n_latents  : int = 32,
        n_heads    : int = 8,
        L_cross    : int = 1,
        L_self     : int = 2,
        d_proj_out : int = 128,
    ):
        super().__init__()

        d_ff = 4 * d_model

        # ── Learned latent array ──────────────────────────────────────────────
        # Shape: (1, N_latent, D) — shared across the batch
        self.latents = nn.Parameter(
            torch.randn(1, n_latents, d_model) * 0.02
        )

        # ── Perceiver blocks ──────────────────────────────────────────────────
        # L_cross blocks, each containing:
        #   1 × CrossAttention + FFN
        #   L_self × SelfAttention + FFN
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
        # q_glob: (1, 1, D) — cross-attends to final latents → z_TEM
        self.q_glob     = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_cross = CrossAttentionBlock(d_model, n_heads)
        self.pool_norm  = nn.LayerNorm(d_model)

        # ── Projection head (A9) — used for InfoNCE only ──────────────────────
        self.img_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proj_out),
        )

    def forward(
        self,
        Z         : torch.Tensor,                    # (B, M, D)
        attn_bias : torch.Tensor | None = None,      # (B, H, N_latent, M)  Method 2 hook
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Z         : (B, M=2535, D=256)  assembled token sequence
            attn_bias : None for Method 1.
                        Method 2 passes (B, H, N_latent, M) positional bias here.

        Returns:
            z_TEM      : (B, D=256)   latent embedding
            z_TEM_proj : (B, 128)     projected embedding for InfoNCE loss
        """
        B = Z.shape[0]

        # Expand learned latents to batch size
        latents = self.latents.expand(B, -1, -1)    # (B, 32, 256)

        # ── Perceiver cross + self attention ──────────────────────────────────
        for block in self.cross_blocks:
            # Cross-attention: latents attend to Z
            latents = block["cross_attn"](latents, Z, attn_bias=attn_bias)
            latents = block["cross_ff"](latents)

            # Self-attention: latents attend to themselves
            for self_layer in block["self_layers"]:
                latents = self_layer["self_attn"](latents)
                latents = self_layer["self_ff"](latents)

        # ── Global pooling: q_glob cross-attends to final latents ─────────────
        q_glob = self.q_glob.expand(B, -1, -1)      # (B, 1, 256)
        z_TEM  = self.pool_cross(q_glob, latents)   # (B, 1, 256)
        z_TEM  = self.pool_norm(z_TEM)
        z_TEM  = z_TEM.squeeze(1)                   # (B, 256)

        # ── Projection head ───────────────────────────────────────────────────
        z_TEM_proj = self.img_proj(z_TEM)            # (B, 128)

        return z_TEM, z_TEM_proj

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"PerceiverAggregator [Method 1]\n"
            f"  latents      : ({self.N_LATENT}, {self.D})\n"
            f"  cross blocks : {self.L_CROSS} × (CrossAttn + FFN + {self.L_SELF}×SelfAttn+FFN)\n"
            f"  heads        : {self.N_HEADS},  d_head={self.D // self.N_HEADS}\n"
            f"  d_ff         : {self.D_FF}\n"
            f"  pool         : q_glob cross-attn → z_TEM (B, {self.D})\n"
            f"  img_proj     : Linear({self.D}→{self.D})→GELU→Linear({self.D}→{self.D_PROJ})\n"
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

    model = PerceiverAggregator().to(device)
    print(model)

    # Simulate GeometryTokens output
    Z = torch.randn(B, M, D, device=device)

    # Method 1: no attn_bias
    t0 = time.time()
    with torch.no_grad():
        z_TEM, z_TEM_proj = model(Z, attn_bias=None)
    elapsed = time.time() - t0

    print(f"Z          : {tuple(Z.shape)}")
    print(f"z_TEM      : {tuple(z_TEM.shape)}       expected (B={B}, 256)")
    print(f"z_TEM_proj : {tuple(z_TEM_proj.shape)}  expected (B={B}, 128)")

    assert z_TEM.shape      == (B, 256), f"z_TEM shape mismatch: {z_TEM.shape}"
    assert z_TEM_proj.shape == (B, 128), f"z_TEM_proj shape mismatch: {z_TEM_proj.shape}"

    # Verify attn_bias hook works (simulate Method 2 bias)
    H        = 8
    N_latent = 32
    fake_bias = torch.zeros(B, H, N_latent, M, device=device)
    with torch.no_grad():
        z2, z2p = model(Z, attn_bias=fake_bias)
    assert z2.shape == (B, 256)
    print(f"\nattn_bias hook (Method 2 prep): OK — shapes match with bias={tuple(fake_bias.shape)}")

    print(f"\nForward pass: {elapsed*1000:.1f} ms  (B={B})")
    print("All assertions passed!")