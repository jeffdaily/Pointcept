"""
SpConv Backend Selection

Sparse convolution models are written against the spconv 2.x API. spconv ships
prebuilt CUDA kernels and has no AMD GPU build, so on a ROCm PyTorch the same API
is taken from spconv-triton, a drop-in reimplementation of spconv 2.x in Triton
(`pip install spconv-triton`, refer to `README.md`).
"""

import torch

if torch.version.hip is None:
    import spconv.pytorch as spconv
else:
    import spconv_triton.pytorch as spconv

__all__ = ["spconv"]
