"""
Feed-Forward network (MLP) - step 3.

WHAT IT DOES
------------
Attention lets tokens *exchange* information. The feed-forward network then lets
each token *think* about that information on its own - it is applied to every
token position independently and identically.

It is a simple two-layer MLP with a non-linearity in the middle:

    n_embd  ->  4 * n_embd  ->  GELU  ->  n_embd

WHY EXPAND TO 4x?
-----------------
Projecting up to a wider "hidden" dimension (4x is the GPT-2 convention) gives
the network room to compute richer, non-linear combinations of features before
squeezing back down to the embedding size. The 4x factor is empirical - it just
works well in practice.

GELU vs RELU
------------
GELU ("Gaussian Error Linear Unit") is a smooth version of ReLU. The smoothness
tends to train slightly better for Transformers, which is why GPT-2 uses it.
"""

import torch
import torch.nn as nn

from ..config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # "fc" = fully connected. Expand the representation 4x.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # The non-linearity - without it, two stacked Linears would collapse
        # into a single Linear and the network could not learn complex functions.
        self.gelu = nn.GELU()
        # Project back down to the embedding size so the result can be added
        # to the residual stream.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)     # expand
        x = self.gelu(x)     # non-linearity
        x = self.c_proj(x)   # project back
        x = self.dropout(x)  # regularize
        return x

