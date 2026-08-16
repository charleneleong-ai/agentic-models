"""The muP verification the literature prescribes: does the optimal learning rate hold still?

Tensor Programs V's operational definition — a hyperparameter is muTransferable if its optimal
value is the same across model sizes — makes the diagnostic a width sweep of LR-vs-loss curves,
checking whether the minimum stays put. Run this before any width ladder: if the optimum drifts
under muP, the implementation is wrong and the ladder would measure that instead of the
mechanism under test.

SP is run alongside as the control. Its optimum is expected to drift, which is precisely why a
ladder built on SP cannot distinguish "this mechanism does not transfer" from "the learning rate
stopped being right".
"""

from __future__ import annotations

import math

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import CorpusSpec
from archlab.model import ModelSpec
from archlab.mup import BASE_WIDTH

# Extended past 1e-2 deliberately. The first run put muP's optimum at 1e-2 for all three
# widths, which is the grid's own edge — an optimum pinned to a boundary agrees across widths
# for free, and proves nothing. The claim needs an interior minimum at every width.
LRS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
WIDTHS = (128, 256, 512)


def run(width: int, lr: float, mup: bool, steps: int) -> float:
    corpus = CorpusSpec(vocab_size=64, seq_len=128, n_pairs=3, key_vocab=16)
    spec = ModelSpec(
        vocab_size=64,
        d_model=width,
        n_layers=4,
        n_heads=max(1, width // 32),
        d_head=32,
        d_hidden=4 * width,
        chunk_size=64,
        mup=mup,
        base_width=BASE_WIDTH,
    )
    return train_arm(
        spec, TrainSpec(steps=steps, batch_size=16, lr=lr, seed=0, device="cuda"), corpus
    )["val_markov_loss"]


def main(steps: int = 300) -> None:
    for mup in (False, True):
        label = "muP" if mup else "SP "
        print(f"\n=== {label} — LR vs loss by width ({steps} steps) ===")
        print(f"{'width':>7} " + "".join(f"{lr:>10.0e}" for lr in LRS) + f"{'argmin':>10}")
        optima = {}
        for width in WIDTHS:
            losses = [run(width, lr, mup, steps) for lr in LRS]
            best = LRS[losses.index(min(losses))]
            optima[width] = best
            row = "".join(f"{v:>10.4f}" if math.isfinite(v) else f"{'nan':>10}" for v in losses)
            print(f"{width:>7} {row}{best:>10.0e}", flush=True)
        stable = len(set(optima.values())) == 1
        print(
            f"{label}: optimum {'HOLDS at ' + f'{next(iter(optima.values())):.0e}' if stable else 'DRIFTS ' + str([f'{v:.0e}' for v in optima.values()])}"
        )


if __name__ == "__main__":
    main()
