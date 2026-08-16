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


# ---------------------------------------------------------------------------
# Dyck corpus — nested brackets
# ---------------------------------------------------------------------------
#
# Third corpus design. The first two failed the same way (`corpus-gate.md`): the recall corpus
# was trivial-or-impossible, the composition corpus memorizable-or-impossible. Neither had a
# regime that was hard, learnable and depth-sensitive at once, because difficulty jumped in a
# cliff rather than rising smoothly.
#
# Dyck is chosen for that specific reason: nesting depth is a *continuous* difficulty dial, and
# depth separation for bracket languages is the best-established result in the area.
#
#     [ ( { < ... > } ) ]
#
# Opening brackets are random; the closing sequence is then fully *determined* — close j must
# match open (d-j). So the closes are perfectly predictable in principle, and predicting them
# requires reading the stack in reverse order, which is the operation that costs depth. A model
# that only tracks recent context can close the innermost pairs and must guess the outer ones,
# so accuracy should degrade with nesting position in a way that is directly observable.
#
# Difficulty knob: `depth`. Unlike `chain_len` in the composition corpus, partial credit is
# available — getting the inner half right is worth something — so the loss should move
# smoothly instead of sitting at chance until it collapses.

BRACKET_MARK = 0  # signals "start of a bracket group"


@dataclass(frozen=True)
class DyckSpec:
    """Nested-bracket corpus. `depth` is the difficulty dial and the depth requirement."""

    vocab_size: int = 64
    seq_len: int = 256
    n_types: int = 8
    depth: int = 8
    n_groups: int = 2
    zipf_alpha: float = 1.2
    seed: int = 0

    @property
    def width(self) -> int:
        """One group: marker + `depth` opens + `depth` closes."""
        return 2 * self.depth + 1

    @property
    def n_filler(self) -> int:
        return self.vocab_size - 1 - 2 * self.n_types

    @property
    def open_base(self) -> int:
        return 1

    @property
    def close_base(self) -> int:
        return 1 + self.n_types

    @property
    def filler_base(self) -> int:
        return 1 + 2 * self.n_types


def dyck_filler_chain(spec: DyckSpec) -> Tensor:
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.n_filler
    zipf = torch.arange(1, n + 1, dtype=torch.float) ** -spec.zipf_alpha
    return torch.softmax(torch.rand(n, n, generator=g) * 2.0 + zipf.log().unsqueeze(0), dim=-1)


def generate_dyck(spec: DyckSpec, n_seqs: int, seed: int | None = None) -> tuple[Tensor, Tensor]:
    """Return (tokens, target_mask). The mask marks closing brackets only.

    Also returns nothing about nesting position — see `dyck_close_positions` for the per-depth
    breakdown, which is what shows whether a model is tracking the whole stack or just the top.
    """
    g = torch.Generator().manual_seed(spec.seed if seed is None else seed)
    trans = dyck_filler_chain(spec)

    cells = spec.seq_len // spec.width
    if cells < spec.n_groups:
        raise ValueError(
            f"n_groups={spec.n_groups} of width {spec.width} needs seq_len >= "
            f"{spec.n_groups * spec.width}, got {spec.seq_len}"
        )

    tokens = torch.empty(n_seqs, spec.seq_len, dtype=torch.long)
    state = torch.randint(spec.n_filler, (n_seqs,), generator=g)
    for t in range(spec.seq_len):
        tokens[:, t] = state + spec.filler_base
        state = torch.multinomial(trans[state], 1, generator=g).squeeze(-1)

    mask = torch.zeros(n_seqs, spec.seq_len, dtype=torch.bool)
    for i in range(n_seqs):
        for cell in torch.randperm(cells, generator=g)[: spec.n_groups]:
            start = int(cell) * spec.width
            opens = torch.randint(spec.n_types, (spec.depth,), generator=g)

            tokens[i, start] = BRACKET_MARK
            for j, o in enumerate(opens):
                tokens[i, start + 1 + j] = spec.open_base + int(o)
            # Closes are forced: close j matches open (depth-1-j). Reading the stack in reverse
            # is the operation that requires depth.
            for j, o in enumerate(reversed(opens.tolist())):
                pos = start + 1 + spec.depth + j
                tokens[i, pos] = spec.close_base + o
                mask[i, pos] = True

    return tokens, mask


def dyck_close_positions(spec: DyckSpec, mask: Tensor) -> Tensor:
    """Nesting index of each masked close: 0 = innermost, depth-1 = outermost.

    The diagnostic that matters. A model tracking only recent context gets the innermost closes
    and guesses the outermost, so accuracy falling with nesting index is the signature of a
    depth-limited model — and a flat profile means the stack is genuinely being carried.
    """
    idx = torch.zeros_like(mask, dtype=torch.long)
    for i in range(mask.shape[0]):
        pos = mask[i].nonzero().flatten()
        for start in range(0, len(pos), spec.depth):
            for j in range(spec.depth):
                if start + j < len(pos):
                    idx[i, pos[start + j]] = j
    return idx


def dyck_batches(
    spec: DyckSpec, batch_size: int, n_batches: int, seed: int
) -> list[tuple[Tensor, Tensor]]:
    return [generate_dyck(spec, batch_size, seed=seed * 100_000 + i) for i in range(n_batches)]


# ---------------------------------------------------------------------------
# Permutation composition corpus
# ---------------------------------------------------------------------------
#
# Fourth corpus design. The first three failed the same way (`corpus-gate.md`): difficulty
# jumped in a cliff rather than rising smoothly. Dyck was closer — partial credit exists —
# but the closing brackets are deterministic given the openings, so the model doesn't need
# to *use* nesting depth, just pattern-match.
#
# Permutation composition has no shortcut: you must apply each sigma in order. The answer
# at position k depends on everything before it. Intermediates are never emitted, so the
# model MUST compose — a 1-layer model cannot chain stepwise across positions.
#
#     [PERM] x0 σ1 σ2 ... σk [ANSWER] y     with  y = σk(...σ2(σ1(x0)))
#
# Key property: permutations are non-contracting (bijections), so the answer depends on the
# full chain — unlike arbitrary maps which can collapse. This is the property the doc
# identified as potentially escaping memorization.
#
# Difficulty knob: `chain_len`. Partial credit is available (getting the first few
# compositions right is worth something), so the loss should move smoothly.

PERM_TOKEN, PERM_ANSWER = 0, 1


@dataclass(frozen=True)
class PermSpec:
    """Permutation composition corpus. `chain_len` is the difficulty dial."""

    vocab_size: int = 64
    seq_len: int = 256
    set_size: int = 4           # elements: 0..set_size-1
    n_perms: int = 8            # permutations to use (subset of S_set_size)
    chain_len: int = 4          # permutations to compose per chain
    n_chains: int = 3           # chains per sequence
    zipf_alpha: float = 1.2
    seed: int = 0

    @property
    def width(self) -> int:
        """Tokens one chain occupies: [PERM] x0 σ... [ANSWER] y."""
        return self.chain_len + 4

    @property
    def n_filler(self) -> int:
        return self.vocab_size - 2 - self.set_size - self.n_perms

    @property
    def elem_base(self) -> int:
        return 2  # after PERM_TOKEN, PERM_ANSWER

    @property
    def perm_base(self) -> int:
        return 2 + self.set_size

    @property
    def filler_base(self) -> int:
        return 2 + self.set_size + self.n_perms


def permutation_table(spec: PermSpec) -> Tensor:
    """(n_perms, set_size) — each row is a permutation of 0..set_size-1, fixed per seed."""
    g = torch.Generator().manual_seed(spec.seed + 7919)
    table = torch.empty(spec.n_perms, spec.set_size, dtype=torch.long)
    for p in range(spec.n_perms):
        table[p] = torch.randperm(spec.set_size, generator=g)
    return table


def perm_filler_chain(spec: PermSpec) -> Tensor:
    g = torch.Generator().manual_seed(spec.seed)
    n = spec.n_filler
    zipf = torch.arange(1, n + 1, dtype=torch.float) ** -spec.zipf_alpha
    return torch.softmax(torch.rand(n, n, generator=g) * 2.0 + zipf.log().unsqueeze(0), dim=-1)


def generate_permutations(
    spec: PermSpec, n_seqs: int, seed: int | None = None
) -> tuple[Tensor, Tensor]:
    """Return (tokens, target_mask). The mask marks answer positions only."""
    g = torch.Generator().manual_seed(spec.seed if seed is None else seed)
    perms = permutation_table(spec)
    trans = perm_filler_chain(spec)

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
        for cell in torch.randperm(cells, generator=g)[: spec.n_chains]:
            start = int(cell) * spec.width
            x = int(torch.randint(spec.set_size, (1,), generator=g))
            picks = torch.randint(spec.n_perms, (spec.chain_len,), generator=g)

            tokens[i, start] = PERM_TOKEN
            tokens[i, start + 1] = spec.elem_base + x
            for j, p in enumerate(picks):
                tokens[i, start + 2 + j] = spec.perm_base + int(p)
                x = int(perms[int(p), x])  # apply permutation; intermediate never emitted
            tokens[i, start + 2 + spec.chain_len] = PERM_ANSWER
            tokens[i, start + 3 + spec.chain_len] = spec.elem_base + x
            mask[i, start + 3 + spec.chain_len] = True

    return tokens, mask


def perm_batches(
    spec: PermSpec, batch_size: int, n_batches: int, seed: int
) -> list[tuple[Tensor, Tensor]]:
    return [generate_permutations(spec, batch_size, seed=seed * 100_000 + i) for i in range(n_batches)]


def build_corpus(cfg: dict[str, object]) -> CorpusSpec | ChainSpec | DyckSpec | PermSpec:
    """Turn an ablation config's `corpus:` block into a spec, dispatching on `type`.

    Defaults to the recall corpus so existing configs keep working unchanged. Without this the
    runners each hard-coded CorpusSpec, so a new corpus could not reach an existing ablation.
    """
    fields = {k: v for k, v in cfg.items() if k != "type"}
    kind = cfg.get("type", "recall")
    if kind == "dyck":
        return DyckSpec(**fields)
    if kind == "chain":
        return ChainSpec(**fields)
    if kind == "perm":
        return PermSpec(**fields)
    if kind == "recall":
        return CorpusSpec(**fields)
    raise ValueError(f"unknown corpus type: {kind!r} (expected 'recall', 'chain', 'dyck' or 'perm')")
