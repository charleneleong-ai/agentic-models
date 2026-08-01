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
from archlab.data import ChainSpec, DyckSpec
from archlab.model import ModelSpec

D_MODEL, N_HEADS, D_HEAD, D_HIDDEN = 128, 4, 32, 512
CHUNK_SIZE = 64  # KDA constraint: every corpus seq_len must be a multiple of this


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


def dyck_depth_gate(depth: int = 8, steps: int = 600) -> None:
    """The gate for the Dyck corpus: does adding layers help predict nested closes?

    Same bar as before — a corpus is only usable for a depth ablation if plain depth buys
    something on it. Two prior designs failed here, so this runs before any sweep.
    """
    corpus = DyckSpec(vocab_size=64, seq_len=256, n_types=8, depth=depth, n_groups=2)
    print(f"=== Dyck depth gate (nesting depth={depth}, {steps} steps) ===")
    print(f"chance = ln({corpus.n_types}) = {math.log(corpus.n_types):.3f}; solved approaches 0\n")
    print(f"{'layers':>7} {'close':>9} {'local':>9}")

    scores = {}
    for n_layers in (6, 12, 24, 48):
        m = run_one(n_layers, corpus, steps)
        scores[n_layers] = m["val_recall_loss"]
        print(f"{n_layers:>7} {scores[n_layers]:>9.4f} {m['val_local_loss']:>9.4f}", flush=True)

    shallow, deep = scores[6], scores[48]
    # Relative, not absolute. An absolute bar is wrong for a task whose whole range is small:
    # the first Dyck run cut loss 0.0686 -> 0.0197, a 71% reduction, and still missed a -0.05
    # absolute threshold by 0.001. What matters is whether depth buys a meaningful *fraction*
    # of the available headroom, and separately whether enough headroom exists to measure with.
    reduction = (shallow - deep) / max(shallow, 1e-9)
    print(f"\n6 -> 48 layers: close {deep - shallow:+.4f}  ({reduction:.0%} reduction)")
    if reduction < 0.25:
        print("GATE FAILS — depth does not meaningfully help")
    elif shallow < 0.2:
        print(
            f"GATE PARTIAL — depth helps ({reduction:.0%}) but the shallow baseline is already "
            f"at {shallow:.3f}; too little headroom for a clean ablation. Raise nesting depth."
        )
    else:
        print("GATE PASSES — depth helps and there is headroom to measure it")


def dyck_nesting_envelope(steps: int = 600, n_layers: int = 6) -> None:
    """Sweep *nesting* depth at fixed model depth to locate the hard-but-learnable band.

    The first Dyck gate showed nesting depth 8 is nearly solved by 6 layers (close loss 0.07
    against chance 2.08). That is the opposite failure to the composition corpus — learnable,
    but too easy for depth to matter. A depth ablation needs a nesting depth that a shallow
    model cannot handle, so find where accuracy starts to fall before sweeping model depth.

    Unlike the composition corpus this should degrade *smoothly*: partial credit is available,
    since getting the inner brackets right is worth something even when the outer ones are lost.
    """
    print(f"=== Dyck nesting envelope ({n_layers} layers, {steps} steps) ===")
    print(f"chance = ln(8) = {math.log(8):.3f}; solved approaches 0\n")
    print(f"{'nesting':>8} {'seq_len':>8} {'close':>9} {'local':>9}")
    for depth in (8, 16, 32, 64):
        # KDA requires seq_len % chunk_size == 0, so round up rather than take the raw width
        # multiple — 4 * (2*32+1) = 260 is not divisible by 64 and crashes the kernel.
        raw = max(256, 4 * (2 * depth + 1))
        seq_len = -(-raw // CHUNK_SIZE) * CHUNK_SIZE
        corpus = DyckSpec(vocab_size=64, seq_len=seq_len, n_types=8, depth=depth, n_groups=2)
        m = run_one(n_layers, corpus, steps)
        print(
            f"{depth:>8} {seq_len:>8} {m['val_recall_loss']:>9.4f} {m['val_local_loss']:>9.4f}",
            flush=True,
        )


def dyck_stability(depth: int = 16, repeats: int = 3) -> None:
    """Is this operating point reproducible at all? Run the *same* config repeatedly.

    The first Dyck sweep found a 0.42 swing between two runs of identical config and seed —
    pure CUDA nondeterminism, not seed variance. That is the signature of an operating point
    sitting on the learnability edge, where tiny numerical differences flip whether the model
    cracks the task. No arm comparison survives that.

    Sweeping the step budget asks whether the instability is a convergence artifact (longer
    training settles it) or intrinsic to the difficulty (it does not). Fixed seed throughout,
    so any spread here is nondeterminism alone — a floor under every gap this corpus can
    resolve.
    """
    corpus = DyckSpec(vocab_size=64, seq_len=256, n_types=8, depth=depth, n_groups=2)
    print(f"=== Dyck stability at nesting {depth} (residual, 12 layers, seed 0 throughout) ===")
    print(f"chance = {math.log(corpus.n_types):.3f}\n")
    print(f"{'steps':>7} {'runs':>28} {'spread':>8}")

    for steps in (600, 1200, 2400):
        vals = [run_one(12, corpus, steps)["val_recall_loss"] for _ in range(repeats)]
        spread = max(vals) - min(vals)
        print(f"{steps:>7} {' '.join(f'{v:.4f}' for v in vals):>28} {spread:>8.4f}", flush=True)

    print("\nSpread here is the noise floor: no arm gap smaller than this is measurable.")


def dyck_converged_envelope(steps: int = 2400, repeats: int = 2) -> None:
    """Find a nesting depth that is hard *at convergence*, not hard because training stopped.

    The stability check showed nesting 16 solves to 0.003 with a 0.002 noise floor at 2400
    steps, while at 600 steps it sits near 0.78 — so the difficulty at 600 steps was
    undertraining, and an undertrained model is precisely what cannot be measured reliably.

    Difficulty has to come from the task. This sweeps nesting depth at a converged budget and
    repeats each point, so both the loss and its noise floor are known before any arm is
    compared against it. A usable operating point needs loss well above the floor *and* a floor
    well below the effects worth detecting.
    """
    print(f"=== Dyck converged envelope ({steps} steps, 12 layers, {repeats}x each) ===")
    print(f"chance = ln(8) = {math.log(8):.3f}\n")
    print(f"{'nesting':>8} {'seq_len':>8} {'runs':>20} {'mean':>8} {'floor':>8}")
    for depth in (16, 32, 48, 64):
        raw = max(256, 4 * (2 * depth + 1))
        seq_len = -(-raw // CHUNK_SIZE) * CHUNK_SIZE
        corpus = DyckSpec(vocab_size=64, seq_len=seq_len, n_types=8, depth=depth, n_groups=2)
        vals = [run_one(12, corpus, steps)["val_recall_loss"] for _ in range(repeats)]
        mean = sum(vals) / len(vals)
        print(
            f"{depth:>8} {seq_len:>8} {' '.join(f'{v:.4f}' for v in vals):>20} "
            f"{mean:>8.4f} {max(vals) - min(vals):>8.4f}",
            flush=True,
        )
    print("\nWant: mean well above the floor, floor well below the gaps worth detecting.")
