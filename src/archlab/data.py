"""A synthetic corpus with both local statistics and long-range dependencies.

Why not TinyStories or Shakespeare: this repo promises to run anywhere on CPU with no network,
and a downloaded corpus breaks that. More importantly, a *known* generative process makes an
architecture ablation interpretable — when one arm wins, you can say which structure it
captured, instead of reporting a loss delta with no mechanism attached.

The process has two components, deliberately:

Local — a Zipfian filler vocabulary emitted from a fixed random Markov chain. Learnable from a
short window, so every architecture drives loss down on it. This is what makes the loss curve
smooth and comparable; without it, differences are dominated by noise on rare events.

Long-range — key/value pairs planted early, then queried late, with the answer required at the
query position. The distance between plant and query is controlled and can exceed any local
window, so this is the part a model can only get right by carrying information forward. For a
fixed-size recurrent state it is also the part that should degrade as the state fills up, which
is what `kda-state-capacity` is built to measure.

Reporting the two losses *separately* is the point. Aggregate loss hides which capability an
architecture bought, and the whole reason to run an ablation is to know that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

N_SPECIAL = 2
QUERY_TOKEN, PAD_TOKEN = 0, 1

PLANT_WIDTH = 2  # key, value
QUERY_WIDTH = 3  # query marker, key, answer


@dataclass(frozen=True)
class CorpusSpec:
    """A reproducible synthetic corpus. `seed` fixes both the chain and the samples."""

    vocab_size: int = 64
    seq_len: int = 256
    n_pairs: int = 4
    key_vocab: int = 16
    zipf_alpha: float = 1.2
    seed: int = 0

    @property
    def n_filler(self) -> int:
        return self.vocab_size - N_SPECIAL - self.key_vocab


def markov_chain(spec: CorpusSpec) -> Tensor:
    """Row-stochastic transition matrix over filler tokens, Zipf-weighted so a few dominate."""
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.n_filler
    zipf = torch.arange(1, n + 1, dtype=torch.float) ** -spec.zipf_alpha
    logits = torch.rand(n, n, generator=g) * 2.0 + zipf.log().unsqueeze(0)
    return torch.softmax(logits, dim=-1)


def generate(spec: CorpusSpec, n_seqs: int, seed: int | None = None) -> tuple[Tensor, Tensor]:
    """Return (tokens, answer_mask), both (n_seqs, seq_len).

    `answer_mask` marks the positions that can only be predicted by recalling a planted pair —
    the long-range subset. Everything unmasked is local structure.
    """
    g = torch.Generator().manual_seed(spec.seed if seed is None else seed)
    trans = markov_chain(spec)
    key_base = N_SPECIAL + spec.n_filler

    tokens = torch.empty(n_seqs, spec.seq_len, dtype=torch.long)
    state = torch.randint(spec.n_filler, (n_seqs,), generator=g)
    for t in range(spec.seq_len):
        tokens[:, t] = state + N_SPECIAL
        state = torch.multinomial(trans[state], 1, generator=g).squeeze(-1)

    answer_mask = torch.zeros(n_seqs, spec.seq_len, dtype=torch.bool)
    # Plant pairs in the first half, query them in the last quarter — the gap is the
    # dependency length the architecture has to span.
    plant_end, query_start = spec.seq_len // 2, 3 * spec.seq_len // 4

    # Slots come from a strided grid, not randperm over raw positions. A plant writes 2 tokens
    # and a query writes 3, so freely-chosen slots can overlap and silently clobber each other
    # — which corrupts the answer the model is scored on, and makes the recall metric measure
    # noise. The grid guarantees disjointness by construction.
    plant_cells = (plant_end - 1) // PLANT_WIDTH
    query_cells = (spec.seq_len - query_start) // QUERY_WIDTH
    if min(plant_cells, query_cells) < spec.n_pairs:
        raise ValueError(
            f"n_pairs={spec.n_pairs} exceeds capacity: {plant_cells} plant and {query_cells} "
            f"query slots fit in seq_len={spec.seq_len}. Shorten n_pairs or lengthen the sequence."
        )

    for i in range(n_seqs):
        keys = torch.randperm(spec.key_vocab, generator=g)[: spec.n_pairs]
        values = torch.randint(spec.n_filler, (spec.n_pairs,), generator=g) + N_SPECIAL

        slots = torch.randperm(plant_cells, generator=g)[: spec.n_pairs] * PLANT_WIDTH + 1
        for k, v, s in zip(keys, values, slots, strict=True):
            tokens[i, s] = key_base + int(k)
            tokens[i, s + 1] = v

        cells = torch.randperm(query_cells, generator=g)[: spec.n_pairs]
        for k, v, q in zip(keys, values, cells * QUERY_WIDTH + query_start, strict=True):
            tokens[i, q] = QUERY_TOKEN
            tokens[i, q + 1] = key_base + int(k)
            tokens[i, q + 2] = v
            answer_mask[i, q + 2] = True  # only this position needs recall

    return tokens, answer_mask


def batches(
    spec: CorpusSpec, batch_size: int, n_batches: int, seed: int
) -> list[tuple[Tensor, Tensor]]:
    """Pre-generate the whole run so every arm sees byte-identical data."""
    return [generate(spec, batch_size, seed=seed * 100_000 + i) for i in range(n_batches)]


# ---------------------------------------------------------------------------
# Compositional corpus
# ---------------------------------------------------------------------------
#
# The recall corpus above is depth-saturated: `attn-res-depth` measured an 8x depth increase
# changing the residual baseline by +0.006, in the wrong direction. Markov filler is learnable
# in a couple of layers and the planted-recall task was never learned at all, so nothing in
# between rewards composition — which makes any depth ablation on it meaningless.
#
# This corpus is built so that depth is *required*, not merely permitted. A chain
#
#     [CHAIN] x0 f_a f_b ... f_z [ANSWER] y        with  y = f_z(...f_b(f_a(x0)))
#
# gives the answer only after `chain_len` sequential function applications. Each application
# depends on the previous result, so the computation cannot be flattened: a model gets roughly
# one composition step per layer, and a network shallower than the chain has to guess.
#
# The functions are arbitrary maps rather than permutations, which blocks the shortcut of
# learning a group structure and composing analytically. They are fixed per corpus seed, so
# they are part of the language rather than per-sequence noise — learnable, but only by
# actually composing.
#
# Intermediate results are never emitted. Emitting them would let a 1-layer model chain
# stepwise across positions and would destroy the depth requirement entirely.

CHAIN_TOKEN, ANSWER_TOKEN = 0, 1


@dataclass(frozen=True)
class ChainSpec:
    """A corpus whose targets require `chain_len` sequential compositions to predict."""

    vocab_size: int = 64
    seq_len: int = 256
    n_states: int = 16
    n_funcs: int = 8
    chain_len: int = 8
    n_chains: int = 3
    zipf_alpha: float = 1.2
    seed: int = 0

    @property
    def width(self) -> int:
        """Tokens one chain occupies: [CHAIN] x0 f... [ANSWER] y."""
        return self.chain_len + 4

    @property
    def n_filler(self) -> int:
        return self.vocab_size - N_SPECIAL - self.n_states - self.n_funcs

    @property
    def state_base(self) -> int:
        return N_SPECIAL

    @property
    def func_base(self) -> int:
        return N_SPECIAL + self.n_states

    @property
    def filler_base(self) -> int:
        return N_SPECIAL + self.n_states + self.n_funcs


def function_table(spec: ChainSpec) -> Tensor:
    """(n_funcs, n_states) arbitrary maps — fixed per corpus, so they are learnable."""
    g = torch.Generator().manual_seed(spec.seed + 7919)
    return torch.randint(spec.n_states, (spec.n_funcs, spec.n_states), generator=g)


def filler_chain(spec: ChainSpec) -> Tensor:
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.n_filler
    zipf = torch.arange(1, n + 1, dtype=torch.float) ** -spec.zipf_alpha
    return torch.softmax(torch.rand(n, n, generator=g) * 2.0 + zipf.log().unsqueeze(0), dim=-1)


def generate_chains(spec: ChainSpec, n_seqs: int, seed: int | None = None) -> tuple[Tensor, Tensor]:
    """Return (tokens, target_mask). The mask marks answer positions only."""
    g = torch.Generator().manual_seed(spec.seed if seed is None else seed)
    funcs = function_table(spec)
    trans = filler_chain(spec)

    cells = spec.seq_len // spec.width
    if cells < spec.n_chains:
        raise ValueError(
            f"n_chains={spec.n_chains} of width {spec.width} needs seq_len >= "
            f"{spec.n_chains * spec.width}, got {spec.seq_len}"
        )

    tokens = torch.empty(n_seqs, spec.seq_len, dtype=torch.long)
    state = torch.randint(spec.n_filler, (n_seqs,), generator=g)
    for t in range(spec.seq_len):
        tokens[:, t] = state + spec.filler_base
        state = torch.multinomial(trans[state], 1, generator=g).squeeze(-1)

    mask = torch.zeros(n_seqs, spec.seq_len, dtype=torch.bool)
    for i in range(n_seqs):
        # Strided cells, so chains cannot overlap and clobber one another's answers.
        for cell in torch.randperm(cells, generator=g)[: spec.n_chains]:
            start = int(cell) * spec.width
            x = int(torch.randint(spec.n_states, (1,), generator=g))
            picks = torch.randint(spec.n_funcs, (spec.chain_len,), generator=g)

            tokens[i, start] = CHAIN_TOKEN
            tokens[i, start + 1] = spec.state_base + x
            for j, f in enumerate(picks):
                tokens[i, start + 2 + j] = spec.func_base + int(f)
                x = int(funcs[int(f), x])  # compose; intermediates are never emitted
            tokens[i, start + 2 + spec.chain_len] = ANSWER_TOKEN
            tokens[i, start + 3 + spec.chain_len] = spec.state_base + x
            mask[i, start + 3 + spec.chain_len] = True

    return tokens, mask


def chain_batches(
    spec: ChainSpec, batch_size: int, n_batches: int, seed: int
) -> list[tuple[Tensor, Tensor]]:
    return [generate_chains(spec, batch_size, seed=seed * 100_000 + i) for i in range(n_batches)]
