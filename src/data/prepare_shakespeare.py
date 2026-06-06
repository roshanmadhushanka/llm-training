"""
Prepare the tiny-Shakespeare dataset for CHARACTER-LEVEL language modeling.

Big models (like GPT-2) tokenize text into sub-word pieces using BPE. To keep
things simple and dependency-free, here we use the simplest possible tokenizer:
every unique CHARACTER becomes one integer id. The vocabulary is therefore tiny
(~65 symbols), which is perfect for learning and for training on a laptop.

This script produces three files in the same folder:
  * train.bin  - 90% of the text, as uint16 token ids
  * val.bin    - the remaining 10%, for measuring generalisation
  * meta.pkl   - the char<->int mappings + vocab_size, so we can decode later

Run it with:
    python -m src.data.prepare_shakespeare
"""

import os
import pickle

import numpy as np
import requests

# Save the data next to this file, inside data/shakespeare_char/.
HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "..", "..", "data", "shakespeare_char")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Download the raw text if we don't already have it.
    input_file_path = os.path.join(OUT_DIR, "input.txt")
    if not os.path.exists(input_file_path):
        data_url = (
            "https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt"
        )
        print("downloading tiny shakespeare...")
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # 2) Build the vocabulary: every distinct character, sorted for determinism.
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # 3) The tokenizer is just two lookup tables:
    #    stoi = string/char -> int,  itos = int -> char.
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # text -> list of ids

    # 4) Split into train/val and encode each to integers.
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # 5) Write the ids out as raw uint16 binary (the format DataLoader expects).
    np.array(train_ids, dtype=np.uint16).tofile(os.path.join(OUT_DIR, "train.bin"))
    np.array(val_ids, dtype=np.uint16).tofile(os.path.join(OUT_DIR, "val.bin"))

    # 6) Save the mappings so sample.py can turn generated ids back into text.
    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"done. files written to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()

