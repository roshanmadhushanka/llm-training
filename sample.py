"""
Entry point: SAMPLE (generate text) from a trained checkpoint.

Usage (from the project root, after training):
    python sample.py                      # uses configs/shakespeare_char.py
    python sample.py shakespeare_char_cpu # match the config you trained with

It loads the best checkpoint written during training, then generates a few
samples of Shakespeare-like text, decoding token ids back into characters using
the meta.pkl produced during data preparation.
"""

import importlib
import os
import pickle
import sys

import torch

from src.config import GPTConfig
from src.model import GPT

# ---- generation settings (tweak these and see what changes!) ----
NUM_SAMPLES = 3          # how many independent samples to generate
MAX_NEW_TOKENS = 500     # length of each sample, in characters
TEMPERATURE = 0.8        # <1.0 = more confident/repetitive, >1.0 = more random
TOP_K = 200              # only sample from the K most likely next chars
START = "\n"             # the prompt the model continues from
SEED = 1337


def main():
    torch.manual_seed(SEED)

    # Match the config used for training so we read the right out_dir/dataset.
    config_name = sys.argv[1] if len(sys.argv) > 1 else "shakespeare_char"
    train_config = importlib.import_module(f"configs.{config_name}").train_config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load the checkpoint saved by train.py.
    ckpt_path = os.path.join(train_config.out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first: python train.py")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 2) Rebuild the model with the exact same shape it was trained with.
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()           # disable dropout for clean generation
    model.to(device)

    # 3) Load the char<->int mappings so we can encode the prompt and decode output.
    meta_path = os.path.join("data", train_config.dataset, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # 4) Encode the starting prompt into a (1, T) tensor of token ids.
    start_ids = encode(START)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # 5) Generate and print.
    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
            print(decode(y[0].tolist()))
            print("---------------")


if __name__ == "__main__":
    main()



