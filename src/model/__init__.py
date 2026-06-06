"""The model package: a GPT built up from small, single-concept modules.

Recommended reading order (simplest -> most complete):
    1. layer_norm.py        - normalization
    2. attention.py         - causal self-attention (the core idea)
    3. feed_forward.py      - the per-token MLP
    4. transformer_block.py - combine attention + MLP with residuals
    5. gpt.py               - the full model
"""

from .gpt import GPT
from .transformer_block import Block
from .attention import CausalSelfAttention
from .feed_forward import MLP
from .layer_norm import LayerNorm

__all__ = ["GPT", "Block", "CausalSelfAttention", "MLP", "LayerNorm"]

