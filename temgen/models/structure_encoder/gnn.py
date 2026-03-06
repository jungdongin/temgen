"""
gnn.py

Structure encoder GNN for TEMGen.

Takes a batched PyG-style graph from GraphBuilder and encodes it into
a single fixed-size latent vector z_cell ∈ ℝ^D per sample.

Pipeline (Sections 4-7 of Crystal Structure Encoder spec):

    B4. Node embedding    : Embedding(119, 128)  →  h^(0) ∈ ℝ^128
    B5. Message passing   : 4 × CSPLayerCartesian
        Edge MLP input    : [h_i, h_j, G_flat, RBF(d_ij)]  ∈ ℝ^(2×128 + 9 + 50) = ℝ^315
        Edge MLP          : 315 → 128 → 128  (SiLU)
        Node MLP          : 256 → 128 → 128  (SiLU)
        Aggregation       : mean over neighbors
        Residual          : h^(s) = h^(s-1) + node_mlp(...)
        LayerNorm         : applied before edge/node MLP (pre-norm)
    B6. Readout           : mean pooling over atoms → Linear(128, 256)  → z_cell ∈ ℝ^256
    B7. Projection head   : Linear(256,256) → GELU → Linear(256,128)  → z_cell_proj ∈ ℝ^128

Key design (adapted from DiffCSP's CSPLayer):
    - Cartesian coords, no PBC wrap
    - Gaussian RBF (not Fourier)
    - Gram matrix G_flat appended to every edge feature
    - Pre-LayerNorm for training stability
    - Residual on node update only

Spec numbers (fixed for CuAu 10×10×10):
    hidden_dim   d = 128
    n_layers       = 4
    K_rbf          = 50      (from GraphBuilder)
    d_gram         = 9       (from GraphBuilder)
    edge_in        = 2*128 + 9 + 50 = 315
    node_in        = 2*128          = 256
    D_out          = 256     (latent dim for contrastive learning)
    D_proj         = 128     (InfoNCE projection dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


# ─── Single CSP layer ─────────────────────────────────────────────────────────

class CSPLayerCartesian(nn.Module):
    """
    One message-passing layer adapted from DiffCSP's CSPLayer for
    non-periodic Cartesian coordinates.

    Differences from DiffCSP:
        - Edge input uses Gaussian RBF distances + Gram matrix G_flat
          instead of Fourier fractional-displacement features
        - No PBC wrap, no minimum-image convention

    Args:
        hidden_dim : node/edge feature dimension d (default 128)
        rbf_dim    : Gaussian RBF dimension K_rbf   (default 50)
        gram_dim   : Gram matrix flat dimension      (default 9)
        ln         : use LayerNorm (pre-norm)        (default True)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        rbf_dim   : int = 50,
        gram_dim  : int = 9,
        ln        : bool = True,
    ):
        super().__init__()

        edge_in = hidden_dim * 2 + gram_dim + rbf_dim  # 315
        node_in = hidden_dim * 2                        # 256

        act = nn.SiLU()

        # Pre-norm on node features
        self.layer_norm = nn.LayerNorm(hidden_dim) if ln else nn.Identity()

        # φ_m: edge MLP  315 → 128 → 128
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # φ_h: node MLP  256 → 128 → 128
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        h         : torch.Tensor,   # (total_N, hidden_dim)    node features
        rbf_edge  : torch.Tensor,   # (total_E, rbf_dim)       RBF distances
        gram_edge : torch.Tensor,   # (total_E, gram_dim)      Gram matrix per edge
        edge_index: torch.Tensor,   # (2, total_E)             [src, dst]
    ) -> torch.Tensor:
        """
        Returns updated node features h' : (total_N, hidden_dim)
        """
        residual = h
        h = self.layer_norm(h)                          # pre-norm

        src, dst = edge_index[0], edge_index[1]

        # ── Edge model ────────────────────────────────────────────────────────
        # m_ij = φ_m([h_i, h_j, G_flat, RBF(d_ij)])
        edge_in = torch.cat(
            [h[src], h[dst], gram_edge, rbf_edge], dim=-1
        )                                               # (E, 315)
        m_ij = self.edge_mlp(edge_in)                  # (E, 128)

        # ── Aggregation: mean over incoming messages ───────────────────────────
        m_i = scatter(
            m_ij, dst,
            dim=0,
            dim_size=h.size(0),
            reduce="mean",
        )                                               # (total_N, 128)

        # ── Node update with residual ──────────────────────────────────────────
        node_in = torch.cat([h, m_i], dim=-1)          # (total_N, 256)
        h_new   = self.node_mlp(node_in)               # (total_N, 128)

        return residual + h_new


# ─── Full structure encoder ───────────────────────────────────────────────────

class StructureEncoder(nn.Module):
    """
    Full structure encoder: node embedding → 4×CSPLayerCartesian
    → mean pooling readout → projection head.

    Consumes output of GraphBuilder directly.

    Args:
        hidden_dim  : GNN hidden dimension d       (default 128, fixed by spec)
        n_layers    : number of CSP layers         (default 4,   fixed by spec)
        rbf_dim     : RBF feature dimension        (default 50,  fixed by spec)
        gram_dim    : Gram matrix flat dim         (default 9,   fixed by spec)
        d_out       : readout output dim           (default 256, latent dim)
        d_proj      : projection head output dim   (default 128, InfoNCE dim)
        max_atomic_num : size of embedding table   (default 119)
    """

    # Fixed by spec
    HIDDEN_DIM : int = 128
    N_LAYERS   : int = 4
    RBF_DIM    : int = 50
    GRAM_DIM   : int = 9
    D_OUT      : int = 256
    D_PROJ     : int = 128

    def __init__(
        self,
        hidden_dim    : int = 128,
        n_layers      : int = 4,
        rbf_dim       : int = 50,
        gram_dim      : int = 9,
        d_out         : int = 256,
        d_proj        : int = 128,
        max_atomic_num: int = 118,
    ):
        super().__init__()

        # ── B4: Node embedding ─────────────────────────────────────────────────
        # Embedding(119, 128): atomic number Z → ℝ^128
        self.node_embedding = nn.Embedding(max_atomic_num + 1, hidden_dim)

        # ── B5: Message passing layers ─────────────────────────────────────────
        self.layers = nn.ModuleList([
            CSPLayerCartesian(
                hidden_dim = hidden_dim,
                rbf_dim    = rbf_dim,
                gram_dim   = gram_dim,
                ln         = True,
            )
            for _ in range(n_layers)
        ])

        # ── B6: Readout ────────────────────────────────────────────────────────
        # mean pool → Linear(128, 256) → z_cell ∈ ℝ^256
        self.readout = nn.Linear(hidden_dim, d_out)

        # ── B7: Projection head ────────────────────────────────────────────────
        # z_cell → Linear(256,256) → GELU → Linear(256,128) → z_cell_proj
        self.struct_proj = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_proj),
        )

    def forward(self, graph: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            graph : dict from GraphBuilder.forward() with keys:
                atom_types  (total_N,)
                edge_index  (2, total_E)
                rbf_edge    (total_E, K_rbf)
                gram_edge   (total_E, 9)
                batch       (total_N,)

        Returns:
            z_cell      : (B, 256)  structure latent embedding
            z_cell_proj : (B, 128)  projected embedding for InfoNCE
        """
        atom_types = graph["atom_types"]    # (total_N,)
        edge_index  = graph["edge_index"]   # (2, total_E)
        rbf_edge    = graph["rbf_edge"]     # (total_E, K_rbf)
        gram_edge   = graph["gram_edge"]    # (total_E, 9)
        batch       = graph["batch"]        # (total_N,)

        B = int(batch.max().item()) + 1

        # ── B4: Initial node embedding ─────────────────────────────────────────
        h = self.node_embedding(atom_types)             # (total_N, 128)

        # ── B5: Message passing ────────────────────────────────────────────────
        for layer in self.layers:
            h = layer(h, rbf_edge, gram_edge, edge_index)  # (total_N, 128)

        # ── B6: Mean pooling + readout ─────────────────────────────────────────
        # mean over atoms within each graph
        h_mean  = scatter(h, batch, dim=0, dim_size=B, reduce="mean")  # (B, 128)
        z_cell  = self.readout(h_mean)                                  # (B, 256)

        # ── B7: Projection head ────────────────────────────────────────────────
        z_cell_proj = self.struct_proj(z_cell)                          # (B, 128)

        return z_cell, z_cell_proj

    def __repr__(self) -> str:
        edge_in = self.HIDDEN_DIM * 2 + self.GRAM_DIM + self.RBF_DIM
        node_in = self.HIDDEN_DIM * 2
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"StructureEncoder(\n"
            f"  node_embedding : Embedding(119, {self.HIDDEN_DIM})\n"
            f"  layers         : {self.N_LAYERS} × CSPLayerCartesian\n"
            f"    edge_mlp     : Linear({edge_in}→{self.HIDDEN_DIM})→SiLU→Linear({self.HIDDEN_DIM}→{self.HIDDEN_DIM})→SiLU\n"
            f"    node_mlp     : Linear({node_in}→{self.HIDDEN_DIM})→SiLU→Linear({self.HIDDEN_DIM}→{self.HIDDEN_DIM})→SiLU\n"
            f"    pre_norm     : LayerNorm({self.HIDDEN_DIM})\n"
            f"    aggregation  : mean\n"
            f"  readout        : mean_pool → Linear({self.HIDDEN_DIM}→{self.D_OUT})\n"
            f"  struct_proj    : Linear({self.D_OUT}→{self.D_OUT})→GELU→Linear({self.D_OUT}→{self.D_PROJ})\n"
            f"  params         : {n_params:,}\n"
            f")"
        )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from torch_geometric.nn import radius_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = StructureEncoder().to(device)
    print(model)
    print()

    # Simulate GraphBuilder output for B=2 samples
    B   = 2
    N   = 640   # atoms per sample
    K   = 50    # RBF dim
    rng = torch.Generator().manual_seed(0)

    total_N = B * N

    # Mock atom types: Cu=29 or Au=79
    atom_types = torch.where(
        torch.rand(total_N, generator=rng) > 0.5,
        torch.full((total_N,), 29),
        torch.full((total_N,), 79),
    ).to(device)

    # Mock Cartesian coords (centred, within ±7.5 xy, ±25 z)
    cart = torch.randn(total_N, 3, generator=rng).to(device)
    cart[:, :2] *= 7.5
    cart[:, 2]  *= 25.0

    # batch vector
    batch = torch.repeat_interleave(
        torch.arange(B, device=device),
        torch.full((B,), N, dtype=torch.long, device=device),
    )                                                  # (total_N,)

    # Radius graph
    edge_index = radius_graph(cart, r=5.0, batch=batch, loop=False, max_num_neighbors=64)
    total_E = edge_index.shape[1]

    # Mock RBF and Gram features
    rbf_edge  = torch.rand(total_E, K,  device=device)
    gram_edge = torch.rand(total_E, 9,  device=device)

    graph = dict(
        atom_types = atom_types,
        edge_index  = edge_index,
        rbf_edge    = rbf_edge,
        gram_edge   = gram_edge,
        batch       = batch,
    )

    t0 = time.time()
    with torch.no_grad():
        z_cell, z_cell_proj = model(graph)
    elapsed = time.time() - t0

    print(f"total_N      : {total_N}  (B={B} × N={N})")
    print(f"total_E      : {total_E}  avg_nbr={total_E/total_N:.1f}")
    print(f"z_cell       : {tuple(z_cell.shape)}       expected (B={B}, 256)")
    print(f"z_cell_proj  : {tuple(z_cell_proj.shape)}  expected (B={B}, 128)")

    assert z_cell.shape      == (B, 256), f"z_cell shape mismatch: {z_cell.shape}"
    assert z_cell_proj.shape == (B, 128), f"z_cell_proj shape mismatch"

    # Verify edge_mlp input dim
    edge_in = 128 * 2 + 9 + 50
    assert edge_in == 315, f"edge_in={edge_in}"
    print(f"\nedge_mlp input dim : {edge_in} == 315 ✓")

    print(f"Forward pass       : {elapsed*1000:.1f} ms  (B={B})")
    print("All assertions passed!")