"""The synthetic corpus. If the recall task is not solvable from context, the whole
depth/capacity ablation measures nothing — so that is the first thing asserted here."""

from __future__ import annotations

import math

import pytest
import torch

from archlab.data import (
    ANSWER_TOKEN,
    CHAIN_TOKEN,
    N_SPECIAL,
    QUERY_TOKEN,
    ChainSpec,
    CorpusSpec,
    function_table,
    generate,
    generate_chains,
    markov_chain,
)

SPEC = CorpusSpec(vocab_size=64, seq_len=128, n_pairs=3, key_vocab=16, seed=0)


@pytest.fixture(scope="module")
def sample() -> tuple[torch.Tensor, torch.Tensor]:
    return generate(SPEC, 32)


class TestTaskIsSolvable:
    """Every marked answer must be recoverable from earlier context, or the metric is noise."""

    def test_every_answer_has_its_key_value_planted_earlier(
        self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        tokens, mask = sample
        key_base = N_SPECIAL + SPEC.n_filler
        checked = 0

        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            key, answer = tokens[i, j - 1], tokens[i, j]
            assert tokens[i, j - 2] == QUERY_TOKEN  # the query marker precedes the key
            assert key >= key_base  # and the queried token really is a key

            earlier = tokens[i, : j - 2]
            plant = (earlier == key).nonzero()
            assert plant.numel() > 0, "queried key was never planted"
            assert tokens[i, plant[0, 0] + 1] == answer, "planted value disagrees with the answer"
            checked += 1

        assert checked == 32 * SPEC.n_pairs

    def test_answers_are_not_guessable_from_the_key_alone(
        self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """If a key always maps to the same value, recall collapses to memorization."""
        tokens, mask = sample
        by_key: dict[int, set[int]] = {}
        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            by_key.setdefault(int(tokens[i, j - 1]), set()).add(int(tokens[i, j]))
        assert max(len(v) for v in by_key.values()) > 1


class TestStructure:
    def test_mask_marks_exactly_n_pairs_per_sequence(
        self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        _, mask = sample
        assert (mask.sum(dim=1) == SPEC.n_pairs).all()

    def test_dependency_spans_more_than_a_local_window(
        self, sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Plant sits in the first half, query in the last quarter — the gap is the point."""
        tokens, mask = sample
        key_base = N_SPECIAL + SPEC.n_filler
        gaps = []
        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            plant = (tokens[i, : j - 2] == tokens[i, j - 1]).nonzero()[0, 0]
            gaps.append(int(j) - int(plant))
        assert min(gaps) > SPEC.seq_len // 8

    def test_tokens_stay_in_vocab(self, sample: tuple[torch.Tensor, torch.Tensor]) -> None:
        tokens, _ = sample
        assert tokens.min() >= 0
        assert tokens.max() < SPEC.vocab_size

    def test_local_stream_is_learnable_not_uniform(self) -> None:
        """The Markov chain must carry real signal, else 'local loss' measures nothing."""
        trans = markov_chain(SPEC)
        entropy = -(trans * trans.clamp_min(1e-12).log()).sum(-1).mean()
        assert entropy < math.log(SPEC.n_filler) * 0.95  # meaningfully below uniform


class TestReproducibility:
    def test_same_seed_same_data(self) -> None:
        a, ma = generate(SPEC, 8, seed=7)
        b, mb = generate(SPEC, 8, seed=7)
        assert torch.equal(a, b) and torch.equal(ma, mb)

    def test_different_seed_different_data(self) -> None:
        a, _ = generate(SPEC, 8, seed=7)
        b, _ = generate(SPEC, 8, seed=8)
        assert not torch.equal(a, b)


CHAIN = ChainSpec(vocab_size=64, seq_len=256, n_states=16, n_funcs=8, chain_len=6, n_chains=3)
N_CHAIN_SEQS = 24


@pytest.fixture(scope="module")
def chain_sample() -> tuple[torch.Tensor, torch.Tensor]:
    return generate_chains(CHAIN, N_CHAIN_SEQS)


class TestCompositionalCorpus:
    """The corpus built to make depth *necessary*. If composition can be shortcut, the whole
    depth ablation is measuring something else."""

    def test_every_answer_is_the_true_composition(
        self, chain_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The target must actually equal f_z(...f_a(x0)) — otherwise it is unlearnable noise."""
        tokens, mask = chain_sample
        s, funcs = CHAIN, function_table(CHAIN)
        checked = 0

        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            start = int(j) - 3 - s.chain_len
            assert tokens[i, start] == CHAIN_TOKEN
            assert tokens[i, j - 1] == ANSWER_TOKEN

            x = int(tokens[i, start + 1]) - s.state_base
            for t in range(s.chain_len):
                x = int(funcs[int(tokens[i, start + 2 + t]) - s.func_base, x])
            assert x + s.state_base == int(tokens[i, j])
            checked += 1

        assert checked == 24 * s.n_chains

    def test_intermediates_are_never_emitted(
        self, chain_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Emitting them would let a 1-layer model chain stepwise and erase the depth
        requirement — the single most important property of this corpus."""
        tokens, mask = chain_sample
        s = CHAIN
        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            start = int(j) - 3 - s.chain_len
            body = tokens[i, start + 2 : start + 2 + s.chain_len]
            assert (body >= s.func_base).all()  # function tokens only
            assert (body < s.filler_base).all()

    def test_answer_depends_on_the_whole_chain(
        self, chain_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perturbing the *first* function must change the answer, or later steps dominate and
        the effective chain is shorter than advertised."""
        s, funcs = CHAIN, function_table(CHAIN)
        tokens, mask = chain_sample
        changed = 0
        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            start = int(j) - 3 - s.chain_len
            picks = [int(tokens[i, start + 2 + t]) - s.func_base for t in range(s.chain_len)]
            x0 = int(tokens[i, start + 1]) - s.state_base

            def compose(first: int) -> int:
                x = x0
                for t, f in enumerate([first, *picks[1:]]):
                    x = int(funcs[f, x])
                return x

            changed += any(compose(alt) != compose(picks[0]) for alt in range(s.n_funcs))
        assert changed > 0.5 * int(mask.sum())

    def test_chains_do_not_overlap(self, chain_sample: tuple[torch.Tensor, torch.Tensor]) -> None:
        _, mask = chain_sample
        s = CHAIN
        for i in range(mask.shape[0]):
            positions = mask[i].nonzero().flatten().tolist()
            assert len(positions) == s.n_chains
            assert all(b - a >= s.width for a, b in zip(positions, positions[1:]))

    def test_capacity_is_validated(self) -> None:
        with pytest.raises(ValueError, match="needs seq_len"):
            generate_chains(ChainSpec(seq_len=16, chain_len=8, n_chains=3), 2)

    def test_reproducible_by_seed(self) -> None:
        a, ma = generate_chains(CHAIN, 4, seed=3)
        b, mb = generate_chains(CHAIN, 4, seed=3)
        assert torch.equal(a, b) and torch.equal(ma, mb)
        assert not torch.equal(generate_chains(CHAIN, 4, seed=4)[0], a)
