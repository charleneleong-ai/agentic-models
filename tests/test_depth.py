"""Attention Residuals: retrieval across depth, and what blocking buys (§2.2, Eq. 8-10).

Claims are asserted against `attention_weights` and `live_sources` — real module API — so a
change to the shipped kernel cannot leave these green.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch import nn

from archlab.model import ModelSpec, NanoLM
from archlab.depth.attn_res import BlockAttnRes, FullAttnRes

D_MODEL, B, T = 16, 2, 4


def layer_stack(n: int) -> list[nn.Module]:
    torch.manual_seed(0)
    return [nn.Linear(D_MODEL, D_MODEL) for _ in range(n)]


class TestFullAttnRes:
    def test_retrieves_selectively_not_uniformly(self) -> None:
        """The whole claim: weights are data-dependent, not the uniform sum of a residual."""
        torch.manual_seed(0)
        res = FullAttnRes(D_MODEL, n_layers=8)
        sources = [torch.randn(B, T, D_MODEL) for _ in range(5)]

        weights = res.attention_weights(sources, layer_idx=3)
        torch.testing.assert_close(weights.sum(-1), torch.ones(B, T))
        assert weights.std() > 1e-3  # not a uniform average

    def test_norm_stops_large_layers_dominating(self) -> None:
        """RMSNorm on keys means magnitude alone cannot buy attention mass (Eq. 9)."""
        torch.manual_seed(0)
        res = FullAttnRes(D_MODEL, n_layers=4)
        sources = [torch.randn(B, T, D_MODEL) for _ in range(3)]
        blown_up = [*sources[:2], sources[2] * 1000]

        torch.testing.assert_close(
            res.attention_weights(sources, 1),
            res.attention_weights(blown_up, 1),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_retains_one_source_per_layer(self) -> None:
        """The O(Ld) memory cost Block AttnRes exists to remove."""
        n_layers = 8
        res = FullAttnRes(D_MODEL, n_layers)
        out = res(torch.randn(B, T, D_MODEL), layer_stack(n_layers))
        assert out.shape == (B, T, D_MODEL)
        assert res.live_sources() == n_layers + 1  # every layer output, plus the embedding


class TestBlockAttnRes:
    @pytest.mark.parametrize("n_blocks,block_size", [(4, 3), (8, 12), (2, 5)])
    def test_live_sources_bounded_by_block_count(self, n_blocks: int, block_size: int) -> None:
        """Eq. 10's payoff, measured: retained state is O(N), not O(L), at every depth."""
        n_layers = n_blocks * block_size
        res = BlockAttnRes(D_MODEL, n_blocks, block_size)
        res.reset(torch.randn(B, T, D_MODEL))

        peak = 0
        for idx in range(n_layers):
            res.step(torch.randn(B, T, D_MODEL), idx)
            peak = max(peak, res.live_sources())

        assert peak <= n_blocks + 2  # N block states + embedding + at most one partial
        assert peak < n_layers  # ...and strictly better than holding every layer

    def test_block_boundary_rolls_partial_into_block_state(self) -> None:
        block_size = 3
        res = BlockAttnRes(D_MODEL, n_blocks=4, block_size=block_size)
        res.reset(torch.randn(B, T, D_MODEL))

        for idx in range(block_size - 1):
            res.step(torch.randn(B, T, D_MODEL), idx)
            assert res.partial is not None  # still accumulating inside the block

        res.step(torch.randn(B, T, D_MODEL), block_size - 1)
        assert res.partial is None  # boundary: partial became a block representation
        assert len(res.block_states) == 2  # embedding + first completed block

    def test_intra_block_reduction_is_summation(self) -> None:
        """Inside a block it is an ordinary uniform residual — that is what makes it cheap."""
        res = BlockAttnRes(D_MODEL, n_blocks=4, block_size=3)
        res.reset(torch.zeros(B, T, D_MODEL))
        a, b = torch.randn(B, T, D_MODEL), torch.randn(B, T, D_MODEL)

        res.step(a, 0)
        res.step(b, 1)
        torch.testing.assert_close(res.partial, a + b)

    def test_forward_runs_the_whole_depth_loop(self) -> None:
        res = BlockAttnRes(D_MODEL, n_blocks=2, block_size=3)
        out = res(torch.randn(B, T, D_MODEL), layer_stack(6))
        assert out.shape == (B, T, D_MODEL)


class TestBlockedDegeneratesToFull:
    """`block_size == 1` makes BlockAttnRes identical to FullAttnRes, arms and all.

    Found by a smoke run whose two AttnRes arms reported identical losses and identical
    parameter counts. That was a degenerate config rather than a bug — but a ladder configured
    that way silently compares an arm against itself and reports perfect rank agreement, which
    is the most flattering possible wrong answer.
    """

    def outputs_for(self, n_layers: int, n_blocks: int) -> tuple[Tensor, Tensor]:
        outs = []
        for mixing in ("full", "block"):
            torch.manual_seed(0)
            model = NanoLM(
                ModelSpec(
                    vocab_size=32,
                    d_model=64,
                    n_layers=n_layers,
                    n_heads=2,
                    d_head=32,
                    d_hidden=128,
                    chunk_size=64,
                    depth_mixing=mixing,
                    n_blocks=n_blocks,
                )
            )
            with torch.no_grad():
                outs.append(model(torch.randint(32, (2, 64))))
        return outs[0], outs[1]

    def test_block_size_one_is_indistinguishable_from_full(self) -> None:
        full, blocked = self.outputs_for(n_layers=4, n_blocks=4)
        torch.testing.assert_close(full, blocked)

    def test_the_shipped_config_shape_keeps_them_distinct(self) -> None:
        """12 layers over 4 blocks — what configs/ablations/attn-res-scale.yaml actually runs."""
        full, blocked = self.outputs_for(n_layers=12, n_blocks=4)
        assert not torch.allclose(full, blocked)
