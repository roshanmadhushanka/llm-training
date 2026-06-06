"""
Causal Self-Attention - step 2, and the heart of the Transformer.

THE BIG IDEA
------------
Each token needs to gather information from *other* tokens to understand its
context. Self-attention is the mechanism that lets every token decide, on its
own, which other tokens are worth listening to, and then blend their
information together.

For every token we build three vectors:
  * Query (Q): "what am I looking for?"
  * Key   (K): "what do I contain / advertise?"
  * Value (V): "what information will I hand over if attended to?"

A token's new representation is a weighted average of all the Values, where the
weights come from how well its Query matches each Key (a dot product). High
match -> high weight -> that token contributes more.

"CAUSAL" / MASKED
-----------------
This is a *language model*: it predicts the next token. So a token at position
t is only allowed to look at positions <= t (itself and the past), never the
future - otherwise it would be cheating by peeking at the answer. We enforce
this with a lower-triangular mask that sets all "future" attention weights to
-inf before the softmax (which turns them into 0 probability).

"MULTI-HEAD"
------------
Instead of one big attention, we split the embedding into `n_head` smaller
chunks and run attention independently on each. Each head can specialise in a
different kind of relationship (e.g. syntax vs. long-range topic), and we
concatenate their outputs at the end.

SHAPE LEGEND (used in the comments below)
  B  = batch size            (number of sequences processed at once)
  T  = time / sequence length (number of tokens)
  C  = channels = n_embd      (embedding dimension)
  nh = number of heads
  hs = head size = C // nh
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # The embedding must divide evenly into the heads.
        assert config.n_embd % config.n_head == 0

        # One Linear layer produces Q, K and V for ALL heads at once. It outputs
        # 3 * n_embd features which we later split into the three vectors. Doing
        # it in a single matmul is just an efficiency trick.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # After mixing information across tokens we project back to n_embd.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout for regularization (randomly zeroes some values during training).
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # "Flash Attention" is a fused, memory-efficient GPU kernel added in
        # PyTorch 2.0. If available we use it; otherwise we fall back to the
        # explicit, readable implementation below (great for understanding!).
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print("Using slow (but readable) attention. Flash Attention needs PyTorch >= 2.0")
            # The causal mask: a lower-triangular matrix of 1s. Stored as a
            # buffer so it moves to the GPU with the model but is not a trainable
            # parameter. Shape (1, 1, block_size, block_size) to broadcast over
            # the batch and head dimensions.
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # 1) Project the input into Q, K, V (each of size C), all in one matmul,
        #    then split the result back into three tensors along the last dim.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2) Reshape each into (B, nh, T, hs) so attention runs per-head.
        #    The transpose moves the head dimension next to the batch dimension,
        #    which lets the matmuls below treat (B, nh) as independent problems.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.flash:
            # The fast path does steps 3-6 below in a single fused kernel.
            # `is_causal=True` applies the future-masking for us.
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # ---- The explicit version, written out so you can see the math ----
            # 3) Score every Query against every Key with a dot product.
            #    Scaling by 1/sqrt(hs) keeps the numbers from getting too large,
            #    which would make the softmax too "peaky".
            #    (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # 4) Apply the causal mask: forbid attending to future positions by
            #    setting those scores to -inf (so they become 0 after softmax).
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            # 5) Softmax turns the raw scores into a probability distribution
            #    over the previous tokens (each row sums to 1).
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # 6) Use those weights to take a weighted average of the Values.
            #    (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v

        # 7) Put the heads back together: (B, nh, T, hs) -> (B, T, C).
        #    `contiguous()` is needed because transpose only changes the view.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8) Final output projection (+ dropout).
        y = self.resid_dropout(self.c_proj(y))
        return y

