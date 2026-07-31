"""The nano LM assembler. Its job is to make arms differ in exactly one object — so these
tests are mostly about what stays *equal* across a swap."""

from __future__ import annotations

import pytest
import torch

from archlab.data import CorpusSpec, generate
from archlab.model import ModelSpec, NanoLM, losses

VOCAB, D_MODEL, N_LAYERS = 64, 32, 6


def spec(**kw: object) -> ModelSpec:
    base = dict(
        vocab_size=VOCAB,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=2,
        d_head=16,
        d_hidden=64,
        chunk_size=8,  # test sequences are short; KDA requires seq_len % chunk_size == 0
    )
    return ModelSpec(**{**base, **kw})


class TestDepthMixingIsTheOnlyDifference:
    """The ablation's core assumption: swapping the mixer changes nothing else structurally."""

    def test_attnres_adds_exactly_one_pseudo_query_per_layer(self) -> None:
        torch.manual_seed(0)
        residual = NanoLM(spec(depth_mixing="residual")).n_params()
        torch.manual_seed(0)
        full = NanoLM(spec(depth_mixing="full")).n_params()

        # One pseudo-query per layer, plus the RMSNorm DepthAttention applies to keys (Eq. 9).
        assert full - residual == N_LAYERS * D_MODEL + D_MODEL
        assert (full - residual) / residual < 0.02  # not buying capacity

    @pytest.mark.parametrize("mixing,n_blocks", [("residual", 0), ("full", 0), ("block", 3)])
    def test_all_mixers_produce_the_same_output_shape(self, mixing: str, n_blocks: int) -> None:
        model = NanoLM(spec(depth_mixing=mixing, n_blocks=n_blocks or 3))
        tokens = torch.randint(VOCAB, (2, 32))
        assert model(tokens).shape == (2, 32, VOCAB)

    def test_mixers_actually_differ(self) -> None:
        """Same seed, same weights where shared — outputs must still diverge."""
        tokens = torch.randint(VOCAB, (2, 32))
        outs = []
        for mixing in ("residual", "full", "block"):
            torch.manual_seed(0)
            with torch.no_grad():
                outs.append(NanoLM(spec(depth_mixing=mixing, n_blocks=3))(tokens))
        assert not torch.allclose(outs[0], outs[1])
        assert not torch.allclose(outs[1], outs[2])

    def test_live_sources_orders_as_designed(self) -> None:
        """1 vs O(N) vs O(L) — the memory axis the ablation trades quality against."""
        tokens = torch.randint(VOCAB, (1, 16))
        counts = {}
        for mixing, blocks in (("residual", 3), ("block", 3), ("full", 3)):
            model = NanoLM(spec(depth_mixing=mixing, n_blocks=blocks))
            model(tokens)
            counts[mixing] = model.peak_live_sources()
        assert counts["residual"] == 1
        assert counts["residual"] < counts["block"] < counts["full"]


class TestBlockReconstructsAStandardTransformer:
    def test_residual_stack_is_plain_addition(self) -> None:
        """With ResidualStack the skeleton must be an ordinary pre-norm transformer."""
        torch.manual_seed(0)
        model = NanoLM(spec(depth_mixing="residual"))
        tokens = torch.randint(VOCAB, (1, 16))

        with torch.no_grad():
            h = model.embed(tokens)
            for layer in model.layers:
                h = h + layer(h)
            expected = model.head(model.out_norm(h))
            torch.testing.assert_close(model(tokens), expected)


class TestCausality:
    @pytest.mark.parametrize("mixing", ["residual", "full", "block"])
    def test_later_tokens_cannot_change_earlier_logits(self, mixing: str) -> None:
        torch.manual_seed(0)
        model = NanoLM(spec(depth_mixing=mixing, n_blocks=3, chunk_size=8)).eval()
        tokens = torch.randint(VOCAB, (1, 32))
        perturbed = tokens.clone()
        perturbed[:, 24:] = (perturbed[:, 24:] + 7) % VOCAB

        with torch.no_grad():
            a, b = model(tokens), model(perturbed)
        torch.testing.assert_close(a[:, :24], b[:, :24], rtol=1e-4, atol=1e-4)


class TestLossSplit:
    def test_local_and_recall_partition_the_positions(self) -> None:
        """Aggregate loss must be the mask-weighted mix of the two reported halves."""
        torch.manual_seed(0)
        corpus = CorpusSpec(vocab_size=VOCAB, seq_len=64, n_pairs=3, key_vocab=16)
        tokens, mask = generate(corpus, 4)
        model = NanoLM(spec())

        with torch.no_grad():
            logits = model(tokens)
            local, recall = losses(logits, tokens, mask)
            flat = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), tokens[:, 1:].reshape(-1), reduction="none"
            )
        n_recall = int(mask[:, 1:].sum())
        n_local = flat.numel() - n_recall
        mixed = (local * n_local + recall * n_recall) / flat.numel()
        torch.testing.assert_close(mixed, flat.mean(), rtol=1e-4, atol=1e-4)

    def test_recall_loss_is_zero_when_nothing_is_marked(self) -> None:
        model = NanoLM(spec())
        tokens = torch.randint(VOCAB, (2, 16))
        _, recall = losses(model(tokens), tokens, torch.zeros_like(tokens, dtype=torch.bool))
        assert recall == 0.0


class TestConfigErrors:
    def test_unknown_attention_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown attention"):
            NanoLM(spec(attention_pattern=["nope"]))

    def test_unknown_depth_mixing_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown depth_mixing"):
            NanoLM(spec(depth_mixing="nope"))

    def test_blocks_must_divide_layers(self) -> None:
        with pytest.raises(ValueError, match="must divide"):
            NanoLM(spec(depth_mixing="block", n_blocks=5))  # 6 layers / 5 blocks
