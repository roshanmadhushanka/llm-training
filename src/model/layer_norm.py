"""
LayerNorm - step 1 of building the model.

WHAT IT DOES
------------
For each token vector, LayerNorm re-centres it to mean 0 and re-scales it to
standard deviation 1, then applies a learnable scale (`weight`) and shift
(`bias`). In plain words: it keeps the numbers flowing through the network in a
stable, well-behaved range so training does not blow up or stall.

WHY A CUSTOM CLASS?
-------------------
PyTorch already ships ``nn.LayerNorm``, but it always creates a bias term.
GPT-2-style models sometimes want LayerNorm *without* a bias, and the built-in
module has no switch for that. So we write a tiny wrapper that makes the bias
optional. Everything else is delegated to the fast built-in ``F.layer_norm``.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        # `weight` starts as all 1s -> initially a no-op scaling.
        self.weight = nn.Parameter(torch.ones(ndim))
        # `bias` starts as all 0s -> initially no shift. None means "no bias".
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1e-5 is `eps`, added to the variance to avoid dividing by zero.
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

