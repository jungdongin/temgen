"""
utils.py

Shared utilities for TEMGen.
"""

from __future__ import annotations

import math

import torch


def fourier_encode(x: torch.Tensor, K: int = 10) -> torch.Tensor:
    """
    Encode scalar(s) x with K sinusoidal Fourier bands.

    φ(x) = [sin(π x), cos(π x), sin(2π x), cos(2π x), ...,
             sin(Kπ x), cos(Kπ x)]  ∈ ℝ^(2K)

    Args:
        x   : (...,)  any shape
        K   : number of frequency bands

    Returns:
        enc : (..., 2K)
    """
    freqs = torch.arange(1, K + 1, device=x.device, dtype=x.dtype)  # (K,)
    angles = x.unsqueeze(-1) * math.pi * freqs                       # (..., K)
    enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., 2K)
    return enc
