"""
Entry point: TRAIN a GPT on the character-level Shakespeare dataset.

Usage (from the project root):
    # 1. one-time: build the dataset
    python -m src.data.prepare_shakespeare

    # 2. train (pick a config by name; defaults to shakespeare_char)
    python train.py                      # full GPU-sized model
    python train.py shakespeare_char_cpu # small model, trains fast on CPU

    # Optionally force a device after the config name:
    python train.py shakespeare_char_cpu cpu   # CPU
    python train.py shakespeare_char_cpu cuda  # NVIDIA GPU
    python train.py shakespeare_char_cpu dml   # AMD/Intel GPU via DirectML

The actual logic lives in src/ - this file just wires the config to the Trainer
and fills in a couple of values that depend on the prepared dataset.
"""

import importlib
import os
import pickle
import sys

import torch

from src.training import Trainer


def load_config(name: str):
    """Import configs/<name>.py and return its model_config + train_config."""
    module = importlib.import_module(f"configs.{name}")
    return module.model_config, module.train_config


def auto_select_device(requested: str) -> str:
    """Use the requested device if it makes sense, otherwise fall back nicely."""
    # "dml" (DirectML, for AMD/Intel GPUs) is honoured as-is; the Trainer will
    # give a clear error if torch-directml isn't installed.
    if requested == "dml":
        return "dml"
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available -> using CPU")
        return "cpu"
    # If the config says cpu but an NVIDIA GPU exists, opportunistically use it.
    # (We do NOT auto-switch to DirectML, since it isn't always faster - opt in
    #  explicitly with `python train.py <config> dml`.)
    if requested == "cpu" and torch.cuda.is_available():
        print("CUDA detected -> using GPU")
        return "cuda"
    return requested


def main():
    # Allow choosing a config by name, e.g. `python train.py shakespeare_char_cpu`.
    config_name = sys.argv[1] if len(sys.argv) > 1 else "shakespeare_char"
    print(f"using config: configs/{config_name}.py")
    model_config, train_config = load_config(config_name)

    # An optional 2nd argument overrides the device from the config.
    if len(sys.argv) > 2:
        train_config.device = sys.argv[2]

    # The vocabulary size depends on the dataset, so read it from meta.pkl
    # (written by prepare_shakespeare.py) instead of hard-coding it.
    meta_path = os.path.join("data", train_config.dataset, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        model_config.vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {model_config.vocab_size} (from {meta_path})")
    else:
        raise FileNotFoundError(
            f"{meta_path} not found. Run: python -m src.data.prepare_shakespeare"
        )

    train_config.device = auto_select_device(train_config.device)

    trainer = Trainer(model_config, train_config)
    trainer.train()


if __name__ == "__main__":
    main()



