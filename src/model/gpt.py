"""
The full GPT model - step 5: put the whole network together.

DATA FLOW (top to bottom)
-------------------------
    token ids (integers)
        |
        v
    [wte]  token embedding  -> turn each id into a learned vector
        +
    [wpe]  position embedding -> add "where am I in the sequence" info
        |
        v
    [Block] x n_layer        -> the stack of Transformer blocks
        |
        v
    [ln_f]  final LayerNorm
        |
        v
    [lm_head] linear         -> produce a score (logit) for every vocab token
        |
        v
    logits  -> softmax -> probability of the next token

WEIGHT TYING
------------
The input token-embedding matrix (`wte`) and the output projection (`lm_head`)
share the *same* weights. Intuitively both deal with the same "token <-> vector"
relationship, so sharing them saves parameters and tends to improve results.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..config import GPTConfig
from .layer_norm import LayerNorm
from .transformer_block import Block


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # ``ModuleDict`` just groups the sub-modules under readable names.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe=nn.Embedding(config.block_size, config.n_embd),   # position embeddings
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # the blocks
            ln_f=LayerNorm(config.n_embd, bias=config.bias),      # final layer norm
        ))
        # Maps the final hidden vectors to a score per vocabulary token.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output layer.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise all weights to sensible small random values.
        self.apply(self._init_weights)
        # Special scaled init for the residual projections (from the GPT-2 paper):
        # because outputs of many layers are summed into the residual stream, we
        # shrink these weights so the signal does not grow with depth.
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    # ------------------------------------------------------------------ utils
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters. By convention we exclude the position embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        # Standard GPT-2 initialisation: small normal noise, zero biases.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------- forward
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx:     (B, T) tensor of token ids - the input context.
        targets: (B, T) tensor of token ids - the correct next tokens (training).

        Returns (logits, loss). ``loss`` is None when no targets are provided.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        # Positions 0, 1, 2, ... t-1 - used to look up position embeddings.
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Look up token and position embeddings and add them together. The model
        # has no inherent sense of order, so the position embedding is what tells
        # it "this token came 3rd, that one came 7th".
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd) - broadcasts over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pass through every Transformer block in turn.
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # TRAINING: score every position and compare to the true next token.
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # cross_entropy expects (N, vocab) logits and (N,) targets, so flatten
            # the batch and time dimensions together. ignore_index=-1 lets us mark
            # padding positions that should not contribute to the loss.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # INFERENCE: we only need the prediction for the LAST position, so
            # we save compute by running lm_head on just that one. The list
            # index [-1] keeps the time dimension (shape (B, 1, vocab_size)).
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # ------------------------------------------------------------- optimizer
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Build an AdamW optimizer with sensible weight-decay groups.

        Rule of thumb: apply weight decay to the big 2D matrices (the matmul and
        embedding weights) but NOT to 1D tensors (biases and LayerNorm scales),
        because decaying those tends to hurt.
        """
        import inspect

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")

        # Use the faster "fused" AdamW kernel when training on CUDA.
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively generate new tokens given a starting context.

        idx: (B, T) starting tokens. We append one new token at a time, each
        time feeding the growing sequence back into the model. This loop is what
        "language model generation" actually is under the hood.
        """
        for _ in range(max_new_tokens):
            # If the context is longer than the model can handle, keep only the
            # most recent block_size tokens.
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size:]
            )

            # Forward pass -> logits for the next token (we take the last step).
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # temperature: <1 sharpens, >1 flattens

            # Optionally restrict sampling to the top_k most likely tokens. This
            # avoids occasionally picking a very unlikely (often nonsensical) token.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Convert logits to probabilities and sample one token from them.
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the new token and repeat.
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

