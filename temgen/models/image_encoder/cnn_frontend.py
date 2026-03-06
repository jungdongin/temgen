"""
cnn_frontend.py

CNN Frontend for the TEMGen image encoder.

Takes a batch of multi-tilt TEM diffraction patterns and extracts
per-patch content tokens for each tilt.

Architecture (from CuAu 101010 Encoding Manual, Section A2):
    - ResNet-18 backbone, conv1 modified for 1-channel grayscale input
    - Layers 1-4 only (no avgpool, no fc)
    - All 15 tilts processed together as (15B, 1, 409, 409)
    - Spatial features flattened to 169 patch tokens per tilt
    - Linear projection 512 → 256, followed by LayerNorm(256)

Shape flow:
    Input  : (B, 15, 1, 409, 409)
    Reshape: (15B, 1, 409, 409)
    conv1  : (15B, 64, 205, 205)
    maxpool: (15B, 64, 103, 103)
    layer1 : (15B, 64, 103, 103)
    layer2 : (15B, 128, 52, 52)
    layer3 : (15B, 256, 26, 26)
    layer4 : (15B, 512, 13, 13)
    flatten: (15B, 169, 512)
    proj   : (15B, 169, 256)
    norm   : (15B, 169, 256)
    reshape: (B, 15, 169, 256)  ← t_cont
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNNFrontend(nn.Module):
    """
    ResNet-18 CNN frontend producing content tokens t_cont.

    Args:
        d_model : output token dimension (default: 256, fixed by spec)
        pretrained : load ImageNet weights for conv/BN layers
                     (useful as initialisation even for 1-ch input;
                      conv1 weights are averaged over the 3 input channels)
    """

    # Fixed architectural constants from the spec
    T  : int = 15    # number of tilts
    H  : int = 409   # input height
    W  : int = 409   # input width
    h  : int = 13    # spatial grid height after layer4
    w  : int = 13    # spatial grid width  after layer4
    P  : int = 169   # h * w = number of patch tokens per tilt
    C  : int = 512   # ResNet-18 layer4 output channels
    D  : int = 256   # projected token dimension

    def __init__(
        self,
        d_model   : int  = 256,
        pretrained: bool = False,
    ):
        super().__init__()

        assert d_model == self.D, (
            f"d_model must be {self.D} to match the rest of the image encoder spec."
        )

        # ── ResNet-18 backbone ────────────────────────────────────────────────
        backbone = resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Replace conv1: 3-channel → 1-channel, same kernel/stride/padding
        original_conv1 = backbone.conv1               # (64, 3, 7, 7)
        new_conv1 = nn.Conv2d(
            in_channels  = 1,
            out_channels = 64,
            kernel_size  = 7,
            stride       = 2,
            padding      = 3,
            bias         = False,
        )
        if pretrained:
            # Average ImageNet weights over the 3 input channels → good init
            with torch.no_grad():
                new_conv1.weight.copy_(
                    original_conv1.weight.mean(dim=1, keepdim=True)
                )
        backbone.conv1 = new_conv1

        # Extract only the layers we need (no avgpool, no fc)
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4

        # ── Projection head ───────────────────────────────────────────────────
        # Linear(512 → 256) + LayerNorm(256)
        # LayerNorm is applied per-token (over the D=256 dimension)
        self.proj = nn.Linear(self.C, self.D)
        self.norm = nn.LayerNorm(self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 1, H, W)  float32  — multi-tilt DP batch

        Returns:
            t_cont : (B, T, P, D)  float32  — content tokens
                     where T=15, P=169, D=256
        """
        B, T, C_in, H, W = x.shape
        assert T    == self.T,  f"Expected T={self.T}, got {T}"
        assert C_in == 1,       f"Expected 1-channel input, got {C_in}"
        assert H    == self.H,  f"Expected H={self.H}, got {H}"
        assert W    == self.W,  f"Expected W={self.W}, got {W}"

        # ── Merge batch and tilt dims for parallel CNN processing ─────────────
        x = x.view(B * T, 1, H, W)                     # (15B, 1, 409, 409)

        # ── ResNet-18 feature extraction ──────────────────────────────────────
        x = self.conv1(x)                               # (15B, 64, 205, 205)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)                             # (15B, 64, 103, 103)

        x = self.layer1(x)                              # (15B, 64, 103, 103)
        x = self.layer2(x)                              # (15B, 128, 52, 52)
        x = self.layer3(x)                              # (15B, 256, 26, 26)
        x = self.layer4(x)                              # (15B, 512, 13, 13)

        # ── Flatten spatial dims → patch tokens ───────────────────────────────
        x = x.flatten(2)                                # (15B, 512, 169)
        x = x.permute(0, 2, 1)                         # (15B, 169, 512)

        # ── Project + normalise ───────────────────────────────────────────────
        x = self.proj(x)                                # (15B, 169, 256)
        x = self.norm(x)                                # (15B, 169, 256)

        # ── Restore tilt dimension ────────────────────────────────────────────
        t_cont = x.view(B, T, self.P, self.D)          # (B, 15, 169, 256)

        return t_cont

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"CNNFrontend(\n"
            f"  backbone=ResNet-18 (1-ch conv1)\n"
            f"  proj=Linear({self.C} → {self.D}) + LayerNorm({self.D})\n"
            f"  output=(B, T={self.T}, P={self.P}, D={self.D})\n"
            f"  params={n_params:,}\n"
            f")"
        )


# ─── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    B = 2   # small batch for quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = CNNFrontend(d_model=256, pretrained=False).to(device)
    print(model)
    print()

    # Random input mimicking normalised DP data
    x = torch.randn(B, 15, 1, 409, 409, device=device)

    # Warm-up
    with torch.no_grad():
        _ = model(x)

    # Timed forward pass
    t0 = time.time()
    with torch.no_grad():
        t_cont = model(x)
    elapsed = time.time() - t0

    print(f"Input  : {tuple(x.shape)}")
    print(f"Output : {tuple(t_cont.shape)}")
    print(f"Expected: (B={B}, T=15, P=169, D=256)")
    assert t_cont.shape == (B, 15, 169, 256), \
        f"Shape mismatch! Got {t_cont.shape}"
    print(f"\nForward pass: {elapsed*1000:.1f} ms  (B={B})")

    # Memory estimate at B=256
    bytes_per_elem = 4  # float32
    mem_gb = (256 * 15 * 169 * 256 * bytes_per_elem) / 1e9
    print(f"\nMemory estimate for t_cont at B=256: {mem_gb:.2f} GB")
    print("\nAll assertions passed!")