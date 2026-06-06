"""
A *much* smaller config tuned for training on a CPU in a few minutes.

The default `shakespeare_char.py` config (10.6M params) is meant for a GPU and
would take many hours on a CPU. This config shrinks the model and context so it
trains end-to-end on a laptop CPU quickly, while still learning to produce
recognizable Shakespeare-like character patterns.

These are the same kind of settings Karpathy recommends for the CPU/MacBook
case in the original nanoGPT.

Use it with:
    python train.py shakespeare_char_cpu
"""

from src.config import GPTConfig, TrainConfig

# ----- a tiny GPT (~0.2M params) -----
model_config = GPTConfig(
    block_size=64,    # short context = far less compute per step
    vocab_size=65,    # overwritten from data/meta.pkl at train time
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,      # small/short run, regularization less important
    bias=False,
)

# ----- training process (short + frequent feedback) -----
train_config = TrainConfig(
    out_dir="out-shakespeare-char-cpu",
    dataset="shakespeare_char",
    eval_interval=250,
    eval_iters=20,     # fewer eval batches = faster evals on CPU
    log_interval=10,

    always_save_checkpoint=False,

    batch_size=12,
    gradient_accumulation_steps=1,

    learning_rate=1e-3,
    max_iters=2000,
    lr_decay_iters=2000,
    min_lr=1e-4,
    beta2=0.99,
    warmup_iters=100,

    device="cpu",
    dtype="float32",
    compile=False,
)

