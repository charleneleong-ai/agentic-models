"""Regression tests for bugs found by probing, plus the contracts that make misuse loud.

Every test here corresponds to something that was actually broken or silently cryptic, not to
a hypothetical. Grouped separately from the mechanism tests because they assert *plumbing*
behaviour — gradient hygiene, error messages, shape freedom — rather than paper claims.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch import nn

from archlab.attention.kda import kda_chunkwise, kda_recurrent, lower_bounded_decay
from archlab.depth.attn_res import BlockAttnRes, FullAttnRes
from archlab.moe.latent_moe import LatentMoE
from archlab.moe.quantile_balance import topk_route


def tiny_moe() -> LatentMoE:
    torch.manual_seed(0)
    return LatentMoE(d_model=16, d_latent=8, d_expert_hidden=12, n_routed=8, n_active=2)


class TestBiasStaysOutsideAutograd:
    """`balance_step` used to copy a grad-carrying estimate into the `bias` buffer.

    The buffer then held a `grad_fn`, pinning that step's entire graph alive for the rest of
    training and making non-learnable state look learnable. Eq. 13 defines the bias as sitting
    outside gradient flow; these tests hold it there.
    """

    def test_balance_step_leaves_no_graph_on_the_buffer(self) -> None:
        moe = tiny_moe()
        x = torch.randn(2, 4, 16)

        out, routing = moe(x)
        out.sum().backward()
        moe.balance_step(routing)

        assert moe.bias.grad_fn is None
        assert not moe.bias.requires_grad

    def test_graph_is_released_between_steps(self) -> None:
        """The leak's actual cost: without detaching, the graph outlives the step."""
        moe = tiny_moe()
        x = torch.randn(2, 4, 16)

        out, routing = moe(x)
        out.sum().backward()
        moe.balance_step(routing)
        del out, routing
        gc.collect()

        assert moe.bias.grad_fn is None

    def test_bias_still_changes(self) -> None:
        """Detaching must not neuter the update — it should still steer routing."""
        moe = tiny_moe()
        x = torch.randn(8, 16, 16)

        before = moe.bias.clone()
        _, routing = moe(x)
        moe.balance_step(routing)

        assert not torch.allclose(moe.bias, before)
        assert torch.isfinite(moe.bias).all()

    def test_repeated_train_steps_do_not_accumulate_graph(self) -> None:
        moe = tiny_moe()
        x = torch.randn(2, 4, 16)
        for _ in range(3):
            moe.zero_grad()
            out, routing = moe(x)
            out.sum().backward()
            moe.balance_step(routing)
            assert moe.bias.grad_fn is None


class TestMisuseIsLoud:
    """Each of these used to surface as a cryptic error from deep inside torch."""

    def test_k_equal_to_n_experts_is_rejected(self) -> None:
        """Top-(k+1) needs one unselected expert to supply the cutoff."""
        with pytest.raises(ValueError, match="must be <"):
            topk_route(torch.rand(4, 3), torch.zeros(3), k=3)

    def test_more_layers_than_pseudo_queries_is_rejected(self) -> None:
        """Previously an IndexError from the einsum, three frames deep."""
        res = FullAttnRes(8, n_layers=2)
        with pytest.raises(IndexError, match="pseudo-quer"):
            res(torch.randn(1, 2, 8), [nn.Linear(8, 8) for _ in range(5)])


class TestShapeFreedom:
    def test_kda_supports_d_v_different_from_d_k(self) -> None:
        """The signature allows it and the chunkwise algebra must honour it — S is d_k x d_v,
        not square. Nothing else in the suite exercises the asymmetric case."""
        torch.manual_seed(0)
        b, h, t, dk, dv = 1, 2, 16, 8, 12
        q = torch.randn(b, h, t, dk, dtype=torch.float64)
        k = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=torch.float64), dim=-1)
        v = torch.randn(b, h, t, dv, dtype=torch.float64)
        alpha = lower_bounded_decay(
            torch.randn(b, h, t, dk, dtype=torch.float64),
            torch.zeros(h, 1, 1, dtype=torch.float64),
        )
        beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float64))

        out_ref, state_ref = kda_recurrent(q, k, v, alpha, beta)
        out, state = kda_chunkwise(q, k, v, alpha, beta, chunk_size=4)

        assert state.shape == (b, h, dk, dv)
        torch.testing.assert_close(out, out_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(state, state_ref, rtol=1e-10, atol=1e-10)


class TestNoStateLeaksBetweenForwards:
    """Both depth modules cache sources on `self`; a second forward must not inherit the first."""

    @pytest.mark.parametrize(
        "module", [FullAttnRes(16, 6), BlockAttnRes(16, n_blocks=2, block_size=3)]
    )
    def test_live_sources_is_identical_across_forwards(self, module: nn.Module) -> None:
        layers = [nn.Linear(16, 16) for _ in range(6)]
        x = torch.randn(2, 4, 16)

        module(x, layers)
        first = module.live_sources()
        module(x, layers)

        assert module.live_sources() == first
