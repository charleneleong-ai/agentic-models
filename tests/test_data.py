"""The synthetic corpus. If the recall task is not solvable from context, the whole
depth/capacity ablation measures nothing — so that is the first thing asserted here."""

from __future__ import annotations

import math

import pytest
import torch

from archlab.data import N_SPECIAL, QUERY_TOKEN, CorpusSpec, generate, markov_chain

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
