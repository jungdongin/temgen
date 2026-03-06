"""
temgen/models/image_encoder/__init__.py

Exposes the three image encoder aggregation methods by name.

Usage
-----
    from temgen.models.image_encoder import (
        CNNFrontend,
        GeometryTokens,
        PerceiverAggregator,                  # Method 1
        GeometryAwarePerceiverAggregator,      # Method 2
        CrossViewVoxelAggregator,              # Method 3
    )
"""

from .cnn_frontend                  import CNNFrontend
from .geometry_tokens               import GeometryTokens
from .aggregator                    import PerceiverAggregator                 # Method 1
from .geometry_aware_perceiver      import GeometryAwarePerceiverAggregator    # Method 2
from .cross_view_voxel_aggregator   import CrossViewVoxelAggregator            # Method 3

__all__ = [
    "CNNFrontend",
    "GeometryTokens",
    "PerceiverAggregator",                # Method 1
    "GeometryAwarePerceiverAggregator",   # Method 2
    "CrossViewVoxelAggregator",           # Method 3
]