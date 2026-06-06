"""
Transformer Block - step 4: assemble the pieces.

A GPT model is just N of these blocks stacked on top of each other. Each block
does two things in sequence:

    1. Self-attention  -> tokens share information with each other
    2. Feed-forward    -> each token processes that information

TWO KEY DESIGN PATTERNS
-----------------------
1) Residual ("skip") connections - the ``x + ...`` pattern.
   Instead of replacing x, each sub-layer computes a *change* that is ADDED to
   x. This creates a "residual highway" that lets gradients flow straight back
   through the whole network during training, which is what makes very deep
   networks trainable at all.

       x = x + attention(...)
       x = x + feed_forward(...)

2) Pre-normalization - LayerNorm is applied to the INPUT of each sub-layer
   (``attn(ln_1(x))``), not to its output. This "pre-norm" arrangement (as used
   in GPT-2) is more stable to train than the original "post-norm" Transformer.
"""

import torch
import torch.nn as nn

from ..config import GPTConfig
from .attention import CausalSelfAttention
from .feed_forward import MLP
from .layer_norm import LayerNorm


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)  # normalize before attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)  # normalize before MLP
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note the residual additions and the pre-norm placement.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

