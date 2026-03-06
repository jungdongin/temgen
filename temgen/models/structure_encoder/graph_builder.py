"""
graph_builder.py

Graph construction for the TEMGen structure encoder.

Takes per-sample atom data (frac_coords, atom_types, lengths, angles)
from the dataloader and converts it into a batched PyG-style graph
ready for message passing.

Pipeline (Sections 1-3 of Crystal Structure Encoder spec):

    1. Lattice matrix from lengths + angles
    2. Fractional → Cartesian coords, centered at mean
    3. Gram matrix G = LᵀL  (O(3)-invariant cell shape descriptor)
    4. radius_graph (non-periodic, no wrap)  →  edge_index, cart_diff
    5. Gaussian RBF on interatomic distances
    6. Edge features: [RBF(d_ij), G_flat]  ∈  ℝ^(K_rbf + 9)

Key design choices vs DiffCSP:
    - Cartesian coords (not fractional) — no PBC wrap
    - radius_graph (not fully-connected) — O(N·k) not O(N²)
    - Gaussian RBF (not Fourier basis) — no periodicity assumption

Output tensors (batched, compatible with PyG scatter ops):
    cart_coords  : (total_N, 3)        centred Cartesian coords
    edge_index   : (2, total_E)        source/target atom indices (global)
    rbf_edge     : (total_E, K_rbf)    RBF-encoded distances
    gram_edge    : (total_E, 9)        Gram matrix broadcast to each edge
    batch        : (total_N,)          graph index per atom
    node2graph   : alias for batch
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from torch_scatter import scatter


# ─── Lattice helper ───────────────────────────────────────────────────────────

def lattice_params_to_matrix(
    lengths: torch.Tensor,   # (..., 3)  a, b, c  in Å
    angles : torch.Tensor,   # (..., 3)  α, β, γ  in degrees
) -> torch.Tensor:
    """
    Convert lattice parameters to lattice matrix L  (..., 3, 3)
    where columns are the lattice vectors a, b, c.

    Convention: a along x-axis, b in xy-plane.

    For the CuAu ROI (orthorhombic, all angles = 90°):
        L = diag(15.0, 15.0, 50.0)
        G = LᵀL = diag(225, 225, 2500)

    Args:
        lengths : (..., 3)
        angles  : (..., 3)  in degrees

    Returns:
        L : (..., 3, 3)
    """
    a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    alpha = torch.deg2rad(angles[..., 0])
    beta  = torch.deg2rad(angles[..., 1])
    gamma = torch.deg2rad(angles[..., 2])

    cos_a = torch.cos(alpha)
    cos_b = torch.cos(beta)
    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma)

    # c vector components
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / (sin_g + 1e-8)
    cz = torch.sqrt(torch.clamp(c ** 2 - cx ** 2 - cy ** 2, min=1e-8))

    zeros = torch.zeros_like(a)

    # Build column-major lattice matrix
    # L[:, 0] = a_vec, L[:, 1] = b_vec, L[:, 2] = c_vec
    L = torch.stack([
        torch.stack([a,          zeros,  zeros], dim=-1),
        torch.stack([b * cos_g,  b * sin_g, zeros], dim=-1),
        torch.stack([cx,         cy,     cz   ], dim=-1),
    ], dim=-1)                                             # (..., 3, 3)

    return L


# ─── Gaussian RBF ─────────────────────────────────────────────────────────────

class GaussianRBF(nn.Module):
    """
    Gaussian radial basis function expansion.

    RBF(d) = [exp(-γ_k (d - μ_k)²)]_{k=1}^K  ∈ ℝ^K

    Centers μ_k uniformly spaced in [0, r_c].
    Width γ = 1 / (2 Δμ²), Δμ = r_c / K.

    Args:
        K   : number of RBF centers (default 50, from spec)
        r_c : cutoff radius in Å    (default 5.0, from spec)
    """

    def __init__(self, K: int = 50, r_c: float = 5.0):
        super().__init__()
        self.K   = K
        self.r_c = r_c

        mu    = torch.linspace(0.0, r_c, K)          # (K,)
        delta = mu[1] - mu[0]                         # Δμ = r_c / (K-1)
        gamma = 1.0 / (2.0 * delta ** 2)

        self.register_buffer("mu",    mu)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d : (E,)  interatomic distances in Å

        Returns:
            rbf : (E, K)
        """
        return torch.exp(
            -self.gamma * (d.unsqueeze(-1) - self.mu) ** 2
        )                                              # (E, K)


# ─── Graph builder ────────────────────────────────────────────────────────────

class GraphBuilder(nn.Module):
    """
    Converts a batch of raw atom data into a PyG-compatible batched graph.

    Takes the list-of-dicts output from cuau_dataset.py's collate_fn and
    produces flat batched tensors ready for CSPLayerCartesian.

    Args:
        r_c              : radius cutoff in Å (default 5.0, from spec)
        K_rbf            : number of Gaussian RBF centers (default 50)
        max_num_neighbors: max neighbours per atom for radius_graph (default 64)
    """

    # Fixed by spec
    R_C  : float = 5.0
    K_RBF: int   = 50

    def __init__(
        self,
        r_c              : float = 5.0,
        K_rbf            : int   = 50,
        max_num_neighbors: int   = 64,
    ):
        super().__init__()
        self.r_c               = r_c
        self.max_num_neighbors = max_num_neighbors
        self.rbf               = GaussianRBF(K=K_rbf, r_c=r_c)

    @torch.no_grad()
    def forward(
        self,
        frac_coords_list: List[torch.Tensor],   # list of (N_i, 3)
        atom_types_list : List[torch.Tensor],   # list of (N_i,)
        lengths_list    : List[torch.Tensor],   # list of (3,)
        angles_list     : List[torch.Tensor],   # list of (3,)
    ) -> dict:
        """
        Build batched graph from per-sample atom data.

        Args:
            frac_coords_list : B × (N_i, 3)  fractional coords
            atom_types_list  : B × (N_i,)    atomic numbers
            lengths_list     : B × (3,)      cell lengths in Å
            angles_list      : B × (3,)      cell angles in degrees

        Returns dict with keys:
            cart_coords  : (total_N, 3)        centred Cartesian coords
            atom_types   : (total_N,)          atomic numbers
            edge_index   : (2, total_E)        global atom indices
            rbf_edge     : (total_E, K_rbf)   RBF distance features
            gram_edge    : (total_E, 9)        Gram matrix per edge
            batch        : (total_N,)          graph index per atom
            num_atoms    : (B,)                atoms per graph
        """
        device = frac_coords_list[0].device
        B = len(frac_coords_list)

        # ── Per-sample: build L, Cartesian coords, Gram matrix ────────────────
        cart_coords_list = []
        gram_flat_list   = []

        for i in range(B):
            fc      = frac_coords_list[i]              # (N_i, 3)
            lengths = lengths_list[i].unsqueeze(0)     # (1, 3)
            angles  = angles_list[i].unsqueeze(0)      # (1, 3)

            # Lattice matrix L: (1, 3, 3)
            L = lattice_params_to_matrix(lengths, angles)  # (1, 3, 3)

            # Cartesian: x_i = f_i @ Lᵀ
            cart = fc @ L[0].T                         # (N_i, 3)

            # Centre at mean
            cart = cart - cart.mean(dim=0, keepdim=True)  # (N_i, 3)
            cart_coords_list.append(cart)

            # Gram matrix G = LᵀL → flatten to 9
            G      = L[0].T @ L[0]                     # (3, 3)
            G_flat = G.flatten()                        # (9,)
            gram_flat_list.append(G_flat)

        # ── Concatenate and build batch vector ────────────────────────────────
        cart_all    = torch.cat(cart_coords_list, dim=0)  # (total_N, 3)
        atom_all    = torch.cat(atom_types_list,  dim=0)  # (total_N,)
        num_atoms   = torch.tensor(
            [fc.shape[0] for fc in frac_coords_list],
            device=device, dtype=torch.long
        )                                                   # (B,)
        batch = torch.repeat_interleave(
            torch.arange(B, device=device), num_atoms
        )                                                   # (total_N,)

        # ── Radius graph (non-periodic, no wrap) ──────────────────────────────
        edge_index = radius_graph(
            cart_all,
            r            = self.r_c,
            batch        = batch,
            loop         = False,
            max_num_neighbors = self.max_num_neighbors,
        )                                                   # (2, total_E)

        # ── Edge features ─────────────────────────────────────────────────────
        # Interatomic distance
        diff   = cart_all[edge_index[1]] - cart_all[edge_index[0]]  # (E, 3)
        d_ij   = diff.norm(dim=-1)                                   # (E,)

        # Gaussian RBF
        rbf_edge = self.rbf(d_ij)                                    # (E, K_rbf)

        # Gram matrix broadcast: one G per sample → one per edge
        gram_per_atom = torch.stack(gram_flat_list)[batch]   # (total_N, 9)
        gram_edge     = gram_per_atom[edge_index[0]]         # (E, 9)  from src atom

        return dict(
            cart_coords = cart_all,     # (total_N, 3)
            atom_types  = atom_all,     # (total_N,)
            edge_index  = edge_index,   # (2, total_E)
            rbf_edge    = rbf_edge,     # (total_E, K_rbf)
            gram_edge   = gram_edge,    # (total_E, 9)
            batch       = batch,        # (total_N,)
            num_atoms   = num_atoms,    # (B,)
        )

    def __repr__(self) -> str:
        return (
            f"GraphBuilder(\n"
            f"  r_c={self.r_c} Å,  max_neighbors={self.max_num_neighbors}\n"
            f"  RBF: K={self.rbf.K}, μ∈[0,{self.r_c}], γ={self.rbf.gamma.item():.2f} Å⁻²\n"
            f"  edge_feat_dim={self.rbf.K + 9}  (K_rbf={self.rbf.K} + 9 Gram)\n"
            f")"
        )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    builder = GraphBuilder(r_c=5.0, K_rbf=50).to(device)
    print(builder)
    print()

    # Simulate a batch of B=2 CuAu ROI samples
    B = 2
    rng = torch.Generator().manual_seed(42)

    frac_list   = []
    types_list  = []
    lengths_list = []
    angles_list  = []

    for _ in range(B):
        N = torch.randint(620, 660, (1,), generator=rng).item()
        frac_list.append(torch.rand(N, 3, generator=rng, device=device))
        # Cu=29 or Au=79, random binary
        types_list.append(
            torch.where(
                torch.rand(N, generator=rng, device=device) > 0.5,
                torch.full((N,), 29, device=device),
                torch.full((N,), 79, device=device),
            )
        )
        # CuAu ROI: 15×15×50 Å, orthorhombic
        lengths_list.append(torch.tensor([15.0, 15.0, 50.0], device=device))
        angles_list.append(torch.tensor([90.0, 90.0, 90.0],  device=device))

    t0 = time.time()
    graph = builder(frac_list, types_list, lengths_list, angles_list)
    elapsed = time.time() - t0

    total_N = sum(f.shape[0] for f in frac_list)
    total_E = graph["edge_index"].shape[1]
    avg_nbr = total_E / total_N

    print(f"cart_coords : {tuple(graph['cart_coords'].shape)}")
    print(f"atom_types  : {tuple(graph['atom_types'].shape)}")
    print(f"edge_index  : {tuple(graph['edge_index'].shape)}")
    print(f"rbf_edge    : {tuple(graph['rbf_edge'].shape)}   expected (E, 50)")
    print(f"gram_edge   : {tuple(graph['gram_edge'].shape)}  expected (E, 9)")
    print(f"batch       : {tuple(graph['batch'].shape)}")
    print(f"num_atoms   : {graph['num_atoms'].tolist()}")
    print(f"\ntotal_N={total_N},  total_E={total_E},  avg_neighbors={avg_nbr:.1f}")
    print(f"  (spec expects ~42 at r_c=5.0 Å for bulk; boundary atoms have fewer)")

    # Shape assertions
    assert graph["rbf_edge"].shape[1]  == 50, "RBF dim mismatch"
    assert graph["gram_edge"].shape[1] == 9,  "Gram dim mismatch"
    assert graph["cart_coords"].shape[1] == 3

    # Gram matrix check: for orthorhombic 15×15×50, G = diag(225, 225, 2500)
    G0 = graph["gram_edge"][0]
    print(f"\nGram flat (first edge): {G0.tolist()}")
    print(f"  Expected ~[225, 0, 0, 0, 225, 0, 0, 0, 2500]")

    # Centring check: mean of cart_coords per graph should be ~0
    from torch_scatter import scatter
    means = scatter(graph["cart_coords"], graph["batch"], dim=0, reduce="mean")
    print(f"\nCart coord means per graph (should be ~0):")
    for i in range(B):
        print(f"  graph {i}: {means[i].tolist()}")

    print(f"\nGraph build time: {elapsed*1000:.1f} ms  (B={B})")
    print("All assertions passed!")