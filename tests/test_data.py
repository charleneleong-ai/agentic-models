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
    PERM_TOKEN,
    QUERY_TOKEN,
    ChainSpec,
    CorpusSpec,
    DyckSpec,
    PermSpec,
    function_table,
    generate,
    dyck_close_positions,
    generate_chains,
    generate_dyck,
    generate_permutations,
    markov_chain,
    permutation_table,
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


DYCK = DyckSpec(vocab_size=64, seq_len=256, n_types=8, depth=6, n_groups=2)
N_DYCK_SEQS = 24


@pytest.fixture(scope="module")
def dyck_sample() -> tuple[torch.Tensor, torch.Tensor]:
    return generate_dyck(DYCK, N_DYCK_SEQS)


class TestDyckCorpus:
    """Third corpus design. Chosen because nesting depth is a *continuous* difficulty dial —
    the previous two failed by jumping from trivial to impossible with nothing between."""

    def test_closes_are_the_reverse_of_opens(
        self, dyck_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The target is fully determined by the prefix. If this breaks, the task is noise."""
        tokens, mask = dyck_sample
        checked = 0
        for i in range(N_DYCK_SEQS):
            pos = mask[i].nonzero().flatten().tolist()
            for start in range(0, len(pos), DYCK.depth):
                group = pos[start : start + DYCK.depth]
                open_start = group[0] - DYCK.depth
                opens = [int(tokens[i, open_start + j]) - DYCK.open_base for j in range(DYCK.depth)]
                closes = [int(tokens[i, p]) - DYCK.close_base for p in group]
                assert closes == list(reversed(opens))
                checked += 1
        assert checked == N_DYCK_SEQS * DYCK.n_groups

    def test_group_is_marked_and_well_formed(
        self, dyck_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        tokens, mask = dyck_sample
        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            assert DYCK.close_base <= tokens[i, j] < DYCK.filler_base

    def test_every_nesting_level_is_represented(
        self, dyck_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The per-depth breakdown is the diagnostic; it needs all levels present."""
        _, mask = dyck_sample
        idx = dyck_close_positions(DYCK, mask)
        assert sorted(set(idx[mask].tolist())) == list(range(DYCK.depth))

    def test_outer_brackets_are_not_guessable_from_recent_context(
        self, dyck_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The outermost close is `depth` tokens from its open. If the type distribution at the
        outermost position were skewed, a model could guess it without tracking the stack."""
        tokens, mask = dyck_sample
        idx = dyck_close_positions(DYCK, mask)
        outer = (idx == DYCK.depth - 1) & mask
        types = tokens[outer] - DYCK.close_base
        assert len(set(types.tolist())) > DYCK.n_types // 2  # spread, not concentrated

    def test_capacity_is_validated(self) -> None:
        with pytest.raises(ValueError, match="needs seq_len"):
            generate_dyck(DyckSpec(seq_len=16, depth=8, n_groups=2), 2)

    def test_reproducible_by_seed(self) -> None:
        a, ma = generate_dyck(DYCK, 4, seed=5)
        b, mb = generate_dyck(DYCK, 4, seed=5)
        assert torch.equal(a, b) and torch.equal(ma, mb)
        assert not torch.equal(generate_dyck(DYCK, 4, seed=6)[0], a)


PERM = PermSpec(vocab_size=64, seq_len=256, set_size=4, n_perms=8, chain_len=4, n_chains=3)
N_PERM_SEQS = 24


@pytest.fixture(scope="module")
def perm_sample() -> tuple[torch.Tensor, torch.Tensor]:
    return generate_permutations(PERM, N_PERM_SEQS)


class TestPermCorpus:
    """Permutation composition corpus. The model must compose a chain of permutations
    without seeing intermediates — depth is genuinely required."""

    def test_every_answer_is_the_true_composition(
        self, perm_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The target must actually equal σ_k(...σ_1(x0)) — otherwise it is noise."""
        tokens, mask = perm_sample
        s, perms = PERM, permutation_table(PERM)
        checked = 0

        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            start = int(j) - 3 - s.chain_len
            assert tokens[i, start] == PERM_TOKEN
            assert tokens[i, j - 1] == 1  # PERM_ANSWER

            x = int(tokens[i, start + 1]) - s.elem_base
            for t in range(s.chain_len):
                x = int(perms[int(tokens[i, start + 2 + t]) - s.perm_base, x])
            assert x + s.elem_base == int(tokens[i, j])
            checked += 1

        assert checked == N_PERM_SEQS * s.n_chains

    def test_intermediates_are_never_emitted(
        self, perm_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Emitting intermediates would let a 1-layer model chain stepwise — the single most
        important property of this corpus."""
        tokens, mask = perm_sample
        s = PERM
        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            start = int(j) - 3 - s.chain_len
            body = tokens[i, start + 2 : start + 2 + s.chain_len]
            assert (body >= s.perm_base).all()  # permutation tokens only
            assert (body < s.filler_base).all()

    def test_answer_depends_on_the_whole_chain(
        self, perm_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perturbing the *first* permutation must change the answer — otherwise later
        steps dominate and the effective chain is shorter than advertised."""
        s, perms = PERM, permutation_table(PERM)
        tokens, mask = perm_sample
        changed = 0
        for i, j in zip(*mask.nonzero(as_tuple=True), strict=True):
            start = int(j) - 3 - s.chain_len
            picks = [int(tokens[i, start + 2 + t]) - s.perm_base for t in range(s.chain_len)]
            x0 = int(tokens[i, start + 1]) - s.elem_base

            def compose(first: int) -> int:
                x = x0
                for p in [first, *picks[1:]]:
                    x = int(perms[p, x])
                return x

            changed += any(compose(alt) != compose(picks[0]) for alt in range(s.n_perms))
        assert changed > 0.5 * int(mask.sum())

    def test_permutations_are_bijections(
        self, perm_sample: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Each permutation in the table must be a bijection — otherwise composition can
        collapse and the task degenerates."""
        perms = permutation_table(PERM)
        for p in range(PERM.n_perms):
            assert set(perms[p].tolist()) == set(range(PERM.set_size))

    def test_chains_do_not_overlap(self, perm_sample: tuple[torch.Tensor, torch.Tensor]) -> None:
        _, mask = perm_sample
        s = PERM
        for i in range(mask.shape[0]):
            positions = mask[i].nonzero().flatten().tolist()
            assert len(positions) == s.n_chains
            assert all(b - a >= s.width for a, b in zip(positions, positions[1:]))

    def test_capacity_is_validated(self) -> None:
        with pytest.raises(ValueError, match="needs seq_len"):
            generate_permutations(PermSpec(seq_len=16, chain_len=8, n_chains=3), 2)

    def test_reproducible_by_seed(self) -> None:
        a, ma = generate_permutations(PERM, 4, seed=5)
        b, mb = generate_permutations(PERM, 4, seed=5)
        assert torch.equal(a, b) and torch.equal(ma, mb)
        assert not torch.equal(generate_permutations(PERM, 4, seed=6)[0], a)
