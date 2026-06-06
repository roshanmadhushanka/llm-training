"""
Configuration objects for the model and training.

We use ``@dataclass`` instead of a big pile of loose variables because:
  * every hyper-parameter has an explicit name, type and default value;
  * the IDE can autocomplete and type-check them;
  * it is trivial to print / save / load the whole config for reproducibility.

Read this file first - almost every other module takes one of these configs
as an argument, so knowing the knobs here makes the rest easier to follow.
"""

from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    """Everything needed to *describe the shape* of a GPT model.

    These values decide how big the network is. They do NOT control how it is
    trained (that lives in ``TrainConfig`` below).
    """

    # Maximum context length: how many tokens the model can look back at.
    # GPT-2 used 1024. For our tiny Shakespeare model 256 is plenty.
    block_size: int = 256

    # Size of the vocabulary (number of distinct tokens the model knows).
    # For character-level Shakespeare this ends up being ~65. It is filled in
    # automatically from the dataset's meta.pkl, so we leave it None here.
    vocab_size: int = 65

    # Depth: how many Transformer blocks are stacked on top of each other.
    n_layer: int = 6

    # Number of attention heads in each block. n_embd must be divisible by this.
    n_head: int = 6

    # Width: the dimensionality of every token's vector representation.
    # Bigger = more capacity (and more compute / memory).
    n_embd: int = 384

    # Dropout probability used for regularization. 0.0 disables it.
    # Use 0 for large pre-training, 0.1-0.2 when overfitting a small dataset.
    dropout: float = 0.2

    # Whether Linear and LayerNorm layers include a learnable bias term.
    # GPT-2 used bias=True; disabling it is slightly faster and often a touch
    # better, but True is a fine default while learning.
    bias: bool = True


@dataclass
class TrainConfig:
    """Everything that controls the *training process* (not the model shape)."""

    # ------------------------------------------------------------------ I/O
    out_dir: str = "out-shakespeare-char"   # where checkpoints are written
    dataset: str = "shakespeare_char"        # subfolder under data/
    eval_interval: int = 250                  # run a full eval every N steps
    eval_iters: int = 200                     # how many batches to average over
    log_interval: int = 10                    # print a one-line update every N steps
    always_save_checkpoint: bool = False      # save only when val loss improves

    # ------------------------------------------------------- batching / data
    # We process `batch_size` sequences at once. If we cannot fit a big batch in
    # memory we can split it across `gradient_accumulation_steps` micro-batches
    # and only update the weights once - this *simulates* a larger batch.
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # ----------------------------------------------------- AdamW optimizer
    learning_rate: float = 1e-3   # peak learning rate
    max_iters: int = 5000         # total number of optimizer steps
    weight_decay: float = 1e-1    # L2-style regularization on the big matrices
    beta1: float = 0.9            # AdamW momentum term
    beta2: float = 0.99           # AdamW variance term (0.99 suits small data)
    grad_clip: float = 1.0        # clip gradient norm to this; 0 disables

    # ------------------------------------------- learning-rate schedule
    decay_lr: bool = True         # use the warmup + cosine-decay schedule?
    warmup_iters: int = 100       # linearly ramp LR up over this many steps
    lr_decay_iters: int = 5000    # step at which LR reaches its minimum
    min_lr: float = 1e-4          # the floor LR (usually ~ learning_rate / 10)

    # ----------------------------------------------------------- system
    # 'cuda' if you have an NVIDIA GPU, 'mps' on Apple silicon, else 'cpu'.
    device: str = "cpu"
    # Reduced precision speeds training up a lot on modern GPUs.
    # 'float32' is the safe default for CPU.
    dtype: str = "float32"        # 'float32' | 'bfloat16' | 'float16'
    # torch.compile (PyTorch 2.x) makes the model much faster, but adds a slow
    # first-step compile and needs a recent PyTorch. Off by default for clarity.
    compile: bool = False
    seed: int = 1337

