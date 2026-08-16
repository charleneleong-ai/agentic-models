"""One training run, printed as a single number — for testing cross-process reproducibility.

Three runs of one config inside a single process agreed to 0.038, while runs of the *same*
config from separate invocations spanned 0.42 (0.5439 / ~0.78 / 0.9638). That pattern points at
something that varies between processes but is stable within one: kernel autotuning, TF32 or
cuDNN algorithm selection, or library state that depends on machine conditions at import.

This module exists so the same config can be launched as N independent processes and compared
against the within-process spread already measured. `--deterministic` additionally pins
PyTorch's deterministic algorithms, which distinguishes "nondeterministic kernels" from
"different kernels chosen".
"""

from __future__ import annotations

import argparse

import torch

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import DyckSpec
from archlab.model import ModelSpec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--nesting", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    corpus = DyckSpec(vocab_size=64, seq_len=256, n_types=8, depth=args.nesting, n_groups=2)
    metrics = train_arm(
        ModelSpec(
            vocab_size=64,
            d_model=128,
            n_layers=args.layers,
            n_heads=4,
            d_head=32,
            d_hidden=512,
            depth_mixing="residual",
            chunk_size=64,
        ),
        TrainSpec(steps=args.steps, batch_size=16, lr=3e-4, seed=0, device="cuda"),
        corpus,
    )
    print(
        f"RESULT tag={args.tag} deterministic={args.deterministic} "
        f"close={metrics['val_recall_loss']:.4f} local={metrics['val_markov_loss']:.4f} "
        f"tf32={torch.backends.cuda.matmul.allow_tf32}",
        flush=True,
    )


if __name__ == "__main__":
    main()
