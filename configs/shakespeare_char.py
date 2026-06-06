"""
A small character-level Shakespeare configuration.

This is the "hello world" of training a GPT - small enough to train on a CPU in
a reasonable time, or in a couple of minutes on a GPU. Import these two configs
from the entry-point scripts (train.py / sample.py).

Feel free to tweak the numbers and watch how training behaves - that hands-on
experimentation is the best way to build intuition.
"""

from src.config import GPTConfig, TrainConfig

# ----- model shape (a "baby" GPT) -----
model_config = GPTConfig(
    block_size=256,   # context of up to 256 previous characters
    vocab_size=65,    # overwritten from data/meta.pkl at train time
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,      # we expect to overfit this tiny dataset, so use dropout
    bias=False,
)

# ----- training process -----
train_config = TrainConfig(
    out_dir="out-shakespeare-char",
    dataset="shakespeare_char",
    eval_interval=250,
    eval_iters=200,
    log_interval=10,
    always_save_checkpoint=False,  # only save when validation loss improves

    batch_size=64,
    gradient_accumulation_steps=1,

    learning_rate=1e-3,   # small networks can take a higher LR
    max_iters=5000,
    lr_decay_iters=5000,  # usually == max_iters
    min_lr=1e-4,
    beta2=0.99,           # a bit higher because tokens-per-iter is small
    warmup_iters=100,

    # Auto-pick the best available device. Change to 'cpu' to force CPU.
    device="cpu",
    dtype="float32",
    compile=False,
)

