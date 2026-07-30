"""KDA: the chunkwise parallel form must equal the serial recurrence it replaces."""

from __future__ import annotations

import pytest
import torch

from archlab.attention.kda import (
    KDA,
    G_MIN,
    kda_chunkwise,
    kda_recurrent,
    lower_bounded_decay,
)

B, H, T, DK, DV = 2, 3, 64, 16, 16


@pytest.fixture
def inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    q = torch.randn(B, H, T, DK, dtype=torch.float64)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, DK, dtype=torch.float64), dim=-1)
    v = torch.randn(B, H, T, DV, dtype=torch.float64)
    z = torch.randn(B, H, T, DK, dtype=torch.float64)
    alpha = lower_bounded_decay(z, torch.zeros(H, 1, 1, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))
    return q, k, v, alpha, beta


class TestChunkwiseEquivalence:
    """The UT transform is only worth trusting if it reproduces Eq. 1 exactly."""

    @pytest.mark.parametrize("chunk_size", [1, 2, 8, 16, 32, 64])
    def test_matches_recurrence(self, inputs: tuple[torch.Tensor, ...], chunk_size: int) -> None:
        out_ref, state_ref = kda_recurrent(*inputs)
        out, state = kda_chunkwise(*inputs, chunk_size=chunk_size)
        torch.testing.assert_close(out, out_ref, rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(state, state_ref, rtol=1e-8, atol=1e-8)

    def test_carries_state_across_calls(self, inputs: tuple[torch.Tensor, ...]) -> None:
        """Splitting a sequence and threading the state must equal one pass over the whole."""
        q, k, v, alpha, beta = inputs
        out_full, state_full = kda_chunkwise(q, k, v, alpha, beta, chunk_size=16)

        half = T // 2
        first = (x[:, :, :half] for x in (q, k, v, alpha, beta))
        out_a, state_a = kda_chunkwise(*first, chunk_size=16)
        second = (x[:, :, half:] for x in (q, k, v, alpha, beta))
        out_b, state_b = kda_chunkwise(*second, state=state_a, chunk_size=16)

        torch.testing.assert_close(torch.cat([out_a, out_b], dim=2), out_full)
        torch.testing.assert_close(state_b, state_full)

    def test_output_reads_state_after_own_update(self, inputs: tuple[torch.Tensor, ...]) -> None:
        """The intra-chunk mask retains its diagonal — token t sees its own write."""
        q, k, v, alpha, beta = (x[:, :, :1] for x in inputs)
        out, _ = kda_chunkwise(q, k, v, alpha, beta, chunk_size=1)
        # With one token from a zero state, S_1 = b k v^T, so o_1 = b <k, q> v.
        expected = (
            beta[..., 0].unsqueeze(-1)
            * (k[:, :, 0] * q[:, :, 0]).sum(-1, keepdim=True)
            * v[:, :, 0]
        )
        torch.testing.assert_close(out[:, :, 0], expected)


class TestLowerBoundedDecay:
    """Eq. 5 exists to keep 1/G inside finite precision — check the bound actually holds."""

    def test_stays_within_bounds(self) -> None:
        # Mathematically alpha lies in the open interval (exp(g_min), 1); under extreme logits
        # float32 sigmoid saturates to exactly 0 or 1, so the finite-precision bound is closed.
        z = torch.randn(1000, 64) * 50
        alpha = lower_bounded_decay(z, torch.zeros(1))
        assert (alpha >= torch.exp(torch.tensor(G_MIN))).all()
        assert (alpha <= 1.0).all()

    def test_cumulative_decay_over_tile_fits_bf16(self) -> None:
        """16-token tile: log-decay in (-80, 0), so the reciprocal stays under e^80."""
        alpha = lower_bounded_decay(torch.full((16, 64), -1e4), torch.zeros(1))
        log_cumulative = torch.log(alpha).sum(dim=0)
        assert (log_cumulative > 16 * G_MIN - 1e-3).all()
        assert (log_cumulative <= 0).all()

    def test_unbounded_softplus_baseline_overflows(self) -> None:
        """Contrast: Kimi Linear's g = -exp(A)*Softplus(z) has no floor, so 1/G can explode."""
        z = torch.full((16, 8), 50.0, dtype=torch.float32)
        log_decay = -torch.nn.functional.softplus(z)  # ~ -50 per step
        assert torch.isinf(torch.exp(-log_decay.sum(dim=0))).all()


class TestKDAModule:
    def test_shape_and_finiteness(self) -> None:
        torch.manual_seed(0)
        layer = KDA(d_model=32, n_heads=4, d_head=8)
        x = torch.randn(2, 32, 32)
        out = layer(x, chunk_size=8)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_is_causal(self) -> None:
        """Perturbing a later token must not change an earlier output."""
        torch.manual_seed(0)
        layer = KDA(d_model=32, n_heads=4, d_head=8).eval()
        x = torch.randn(1, 16, 32)
        perturbed = x.clone()
        perturbed[:, 12:] += 5.0
        with torch.no_grad():
            a, b = layer(x, chunk_size=4), layer(perturbed, chunk_size=4)
        torch.testing.assert_close(a[:, :12], b[:, :12])
        assert not torch.allclose(a[:, 12:], b[:, 12:])
