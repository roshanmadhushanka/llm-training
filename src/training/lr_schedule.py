"""
Learning-rate schedule: warmup + cosine decay.

A constant learning rate is rarely optimal. A common, effective recipe is:

  1. WARMUP - start near 0 and ramp the LR up linearly for the first few
     hundred steps. Early on the weights are random and big steps can destabilise
     training, so we ease in gently.

  2. COSINE DECAY - after warmup, smoothly lower the LR following the shape of a
     cosine curve, all the way down to a small `min_lr`. Big steps early (to make
     fast progress) and small steps late (to settle into a good minimum).

  3. FLOOR - once we pass `lr_decay_iters`, just hold at `min_lr`.

    LR
     |        ___
     |       /   \\___
     |      /        \\____
     |     /              \\______ min_lr
     |    /
     |___/______________________________ step
        warmup      cosine decay      floor
"""

import math


def get_lr(step: int, learning_rate: float, warmup_iters: int,
           lr_decay_iters: int, min_lr: float) -> float:
    # 1) Linear warmup.
    if step < warmup_iters:
        return learning_rate * (step + 1) / (warmup_iters + 1)

    # 3) Past the decay horizon -> constant minimum.
    if step > lr_decay_iters:
        return min_lr

    # 2) Cosine decay between warmup_iters and lr_decay_iters.
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    # coeff goes smoothly from 1 (start of decay) to 0 (end of decay).
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

