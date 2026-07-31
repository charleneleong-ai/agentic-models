"""Diagnostics for the compositional corpus, before any sweep is run.

`attn-res-depth` burned two GPU-hours to discover its corpus was depth-saturated. The first
version of this corpus failed its gate too: depth did not help, and the model landed at 2.20
against a marginal-only baseline of 2.70 — learning something, nowhere near solving.

That leaves two candidate causes, and they need different fixes:

  difficulty  the chain is too long to learn in the step budget, so no depth can do it and
              depth cannot differentiate. Fix: shorten the chain or lengthen training.
  structure   the task is degenerate, so it does not require composition at all.
              Fix: change the corpus.

`chain_length_envelope` separates them by holding depth fixed and sweeping chain length. If
short chains are learned and long ones are not, difficulty is the binding constraint and there
is a usable middle. If even a length-1 chain is not learned, the corpus is broken.

Only once a chain length is known to be learnable does sweeping depth mean anything.
"""

from __future__ import annotations

import math

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import ChainSpec
from archlab.model import ModelSpec

D_MODEL, N_HEADS, D_HEAD, D_HIDDEN = 128, 4, 32, 512


def spec_for(n_layers: int, vocab_size: int) -> ModelSpec:
    return ModelSpec(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=n_layers,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_hidden=D_HIDDEN,
        depth_mixing="residual",
        chunk_size=64,
    )


def run_one(n_layers: int, corpus: ChainSpec, steps: int) -> dict[str, float]:
    return train_arm(
        spec_for(n_layers, corpus.vocab_size),
        TrainSpec(steps=steps, batch_size=16, lr=3e-4, seed=0, device="cuda"),
        corpus,
    )


def chain_length_envelope() -> None:
    """Hold depth fixed, sweep chain length. Finds what is learnable at all."""
    print("=== chain-length envelope (residual, 12 layers) ===")
    print(f"chance = ln(16) = {math.log(16):.3f}; a solved chain should approach 0\n")
    print(f"{'chain_len':>10} {'steps':>7} {'compose':>9} {'local':>9}")
    for chain_len, steps in ((1, 600), (2, 600), (4, 600), (8, 600), (2, 2400), (4, 2400)):
        corpus = ChainSpec(
            vocab_size=64,
            seq_len=256,
            n_states=16,
            n_funcs=8,
            chain_len=chain_len,
            n_chains=3,
        )
        m = run_one(12, corpus, steps)
        print(
            f"{chain_len:>10} {steps:>7} {m['val_recall_loss']:>9.4f} {m['val_local_loss']:>9.4f}",
            flush=True,
        )


def depth_at_the_cliff(chain_len: int, steps: int) -> None:
    """Sweep depth at the chain length where learning breaks down.

    This is the gate the sweep actually needs. The envelope showed a sharp cliff — length 2 is
    solved (0.08), length 4 is not (2.18) — so the interesting question is whether depth moves
    that boundary. Shallow failing while deep succeeds is a depth-sensitive task, and the only
    condition under which a depth ablation measures depth.
    """
    corpus = ChainSpec(
        vocab_size=64, seq_len=256, n_states=16, n_funcs=8, chain_len=chain_len, n_chains=3
    )
    print(f"=== depth at chain_len={chain_len}, {steps} steps ===")
    print(f"chance = {math.log(corpus.n_states):.3f}; solved approaches 0\n")
    print(f"{'depth':>6} {'compose':>9} {'local':>9}")

    scores = {}
    for n_layers in (6, 12, 24, 48):
        m = run_one(n_layers, corpus, steps)
        scores[n_layers] = m["val_recall_loss"]
        print(f"{n_layers:>6} {scores[n_layers]:>9.4f} {m['val_local_loss']:>9.4f}", flush=True)

    shallow, deep = scores[6], scores[48]
    print(f"\n6 -> 48 layers: compose {deep - shallow:+.4f}")
    print(
        "GATE PASSES — depth crosses the cliff"
        if deep < shallow - 0.20
        else "GATE FAILS — depth does not move the boundary"
    )
