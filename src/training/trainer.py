"""
The Trainer - ties the model, data and optimizer together into a training loop.

This is a deliberately simplified version of nanoGPT's train.py: it drops the
multi-GPU (DistributedDataParallel) machinery so you can focus on the core
training concepts, which are all still here:

  * mixed-precision training (autocast + GradScaler) for speed on GPUs
  * gradient accumulation to simulate a larger batch than fits in memory
  * gradient clipping for stability
  * a warmup + cosine learning-rate schedule
  * periodic evaluation on a held-out split
  * checkpointing the best model

THE CORE LOOP (one iteration)
-----------------------------
  1. Set the learning rate for this step.
  2. (occasionally) Evaluate on train/val and maybe save a checkpoint.
  3. Forward pass  -> compute the loss.
  4. Backward pass -> compute gradients.
  5. Clip gradients, then take an optimizer step.
  6. Zero the gradients, ready for the next iteration.
"""

import os
from contextlib import nullcontext

import torch
from tqdm import tqdm

from ..config import GPTConfig, TrainConfig
from ..model import GPT
from .data_loader import DataLoader
from .device import resolve_device, describe
from .lr_schedule import get_lr


class Trainer:
    def __init__(self, model_config: GPTConfig, train_config: TrainConfig):
        self.mcfg = model_config
        self.tcfg = train_config

        # Reproducibility: fixing the seed makes runs repeatable.
        torch.manual_seed(train_config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Work out where everything will live (cpu / cuda / dml-DirectML).
        self.device, self.is_cuda, pin_memory = resolve_device(train_config.device)
        print(f"using device: {describe(train_config.device)}")

        # Mixed precision picks a faster lower-precision type for the heavy
        # matmuls while keeping things numerically safe.
        #   * CUDA:  full autocast (bf16/fp16) + GradScaler for fp16.
        #   * CPU:   bfloat16 autocast - modern AMD Zen / Intel cores have
        #            hardware bf16 (AVX512-BF16), so this can speed up training
        #            with no extra dependencies. Set dtype="bfloat16" to use it.
        #   * DML:   run plain float32 (autocast isn't reliable there).
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[train_config.dtype]
        if self.is_cuda:
            self.ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        elif self.device.type == "cpu" and train_config.dtype == "bfloat16":
            self.ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            self.ctx = nullcontext()

        # Data.
        data_dir = os.path.join("data", train_config.dataset)
        self.loader = DataLoader(
            data_dir=data_dir,
            block_size=model_config.block_size,
            batch_size=train_config.batch_size,
            device=self.device,
            pin_memory=pin_memory,
        )

        # Model.
        self.model = GPT(model_config).to(self.device)

        # The GradScaler is only needed for CUDA float16 training; it scales the
        # loss up before backprop to avoid tiny gradients underflowing to zero.
        # A no-op everywhere else.
        self.scaler = torch.amp.GradScaler(
            enabled=(self.is_cuda and train_config.dtype == "float16")
        )

        # Optimizer. The "fused" AdamW path only applies to CUDA; "cpu" is a safe
        # label for both CPU and DirectML here.
        self.optimizer = self.model.configure_optimizers(
            train_config.weight_decay,
            train_config.learning_rate,
            (train_config.beta1, train_config.beta2),
            "cuda" if self.is_cuda else "cpu",
        )

        if train_config.compile:
            print("compiling the model... (first step will be slow)")
            self.model = torch.compile(self.model)

        self.iter_num = 0
        self.best_val_loss = 1e9

    @torch.no_grad()
    def estimate_loss(self):
        """Average the loss over several batches of train and val data.

        A single batch's loss is noisy; averaging many gives a much more
        reliable estimate of how the model is really doing.
        """
        out = {}
        self.model.eval()  # switch off dropout etc. for evaluation
        for split in ["train", "val"]:
            losses = torch.zeros(self.tcfg.eval_iters)
            for k in range(self.tcfg.eval_iters):
                x, y = self.loader.get_batch(split)
                with self.ctx:
                    _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()  # back to training mode
        return out

    def _save_checkpoint(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_args": vars(self.mcfg),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": vars(self.tcfg),
        }
        os.makedirs(self.tcfg.out_dir, exist_ok=True)
        path = os.path.join(self.tcfg.out_dir, "ckpt.pt")
        torch.save(checkpoint, path)

    def train(self):
        cfg = self.tcfg
        # Fetch the first batch; we prefetch the next one inside the loop.
        x, y = self.loader.get_batch("train")

        # A tqdm progress bar gives us a live view of training: how far along we
        # are, iterations/second, an automatic ETA, and (via set_postfix below)
        # the current loss and learning rate. It's purely for visibility - it
        # does not change the training maths at all.
        progress = tqdm(
            total=cfg.max_iters + 1,  # +1: the loop runs one final eval step at max_iters
            desc="training",
            unit="it",
            dynamic_ncols=True,
        )

        while True:
            # 1) Set the learning rate for this iteration.
            lr = (
                get_lr(self.iter_num, cfg.learning_rate, cfg.warmup_iters,
                       cfg.lr_decay_iters, cfg.min_lr)
                if cfg.decay_lr
                else cfg.learning_rate
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # 2) Periodically evaluate and checkpoint. We use tqdm.write so these
            # multi-line messages print cleanly *above* the live progress bar
            # instead of corrupting it.
            if self.iter_num % cfg.eval_interval == 0:
                losses = self.estimate_loss()
                tqdm.write(
                    f"step {self.iter_num}: "
                    f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if losses["val"] < self.best_val_loss or cfg.always_save_checkpoint:
                    self.best_val_loss = min(losses["val"], self.best_val_loss)
                    if self.iter_num > 0:
                        self._save_checkpoint()
                        tqdm.write(f"  saved checkpoint to {self.tcfg.out_dir}")

            # 3+4) Forward/backward, with gradient accumulation. We split the
            # effective batch into `gradient_accumulation_steps` micro-batches,
            # accumulating gradients before a single optimizer step. This lets us
            # train with a large effective batch even on limited memory.
            for micro_step in range(cfg.gradient_accumulation_steps):
                with self.ctx:
                    _, loss = self.model(x, y)
                    # Divide so the accumulated gradient equals the average over
                    # the whole effective batch (not the sum).
                    loss = loss / cfg.gradient_accumulation_steps
                # Prefetch the next batch while the GPU finishes the forward pass.
                x, y = self.loader.get_batch("train")
                # Backward pass (scaler is a no-op unless dtype == float16).
                self.scaler.scale(loss).backward()

            # 5) Clip gradients to a maximum norm to prevent the occasional huge
            # update from destabilising training.
            if cfg.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            # Optimizer step (+ update the scaler for next time).
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 6) Clear gradients. set_to_none=True is a small memory/speed win.
            self.optimizer.zero_grad(set_to_none=True)

            # Update the progress bar with the latest loss and learning rate.
            # loss.item() syncs GPU->CPU; multiply back to undo the division.
            lossf = loss.item() * cfg.gradient_accumulation_steps
            progress.set_postfix(loss=f"{lossf:.4f}", lr=f"{lr:.2e}")
            progress.update(1)  # advance the bar by one iteration

            self.iter_num += 1
            if self.iter_num > cfg.max_iters:
                break

        progress.close()
        print("training finished.")

