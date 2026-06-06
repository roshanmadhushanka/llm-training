"""
Device selection - pick where the tensors and model live.

This project can run on three kinds of device:

  * "cpu"  - always available, the safe default.
  * "cuda" - NVIDIA GPUs (needs a CUDA build of PyTorch).
  * "dml"  - DirectML: runs on ANY DirectX 12 GPU on Windows, including AMD
             Radeon (e.g. the 890M iGPU) and Intel Arc. Requires the optional
             `torch-directml` package (`pip install torch-directml`).

A note on AMD "Ryzen AI" laptops
--------------------------------
These chips have BOTH a Radeon iGPU and an XDNA "NPU".
  * The iGPU is usable here via DirectML (device="dml").
  * The NPU is an INFERENCE-ONLY accelerator. It is reached through ONNX Runtime
    + the Ryzen AI / Vitis AI stack with quantized (INT8) models - PyTorch
    cannot train on it. So there is no "device=npu" option for training.

Reality check: for a tiny model the CPU is often as fast or faster than an iGPU,
because DirectML has noticeable per-operation overhead and the iGPU shares
memory/power with the CPU. DirectML tends to pay off on the larger model.
"""

import torch


def resolve_device(name: str):
    """Turn a device *name* into (torch_device, is_cuda, pin_memory).

    Returns:
        device:     a torch.device (or DirectML device object) to use.
        is_cuda:    True only for real CUDA devices (controls autocast/scaler).
        pin_memory: whether the data loader should pin host memory (CUDA only).
    """
    if name == "dml":
        try:
            import torch_directml  # optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "device='dml' needs the torch-directml package.\n"
                "Install it with:  pip install torch-directml\n"
                "(it provides DirectML GPU support for AMD/Intel GPUs on Windows)."
            ) from exc
        if not torch_directml.is_available():
            raise RuntimeError("DirectML reports no available GPU device.")
        # DirectML exposes its device through this call rather than torch.device.
        return torch_directml.device(), False, False

    device = torch.device(name)
    is_cuda = device.type == "cuda"
    return device, is_cuda, is_cuda


def describe(name: str) -> str:
    """Human-readable one-liner about the chosen device, for logging."""
    if name == "dml":
        try:
            import torch_directml
            return f"DirectML GPU: {torch_directml.device_name(0)}"
        except Exception:
            return "DirectML GPU"
    if name.startswith("cuda"):
        try:
            return f"CUDA GPU: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA GPU"
    return "CPU"

