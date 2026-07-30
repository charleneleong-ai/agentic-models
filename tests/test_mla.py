"""MLA: latent KV compression, NoPE extrapolation, and the cache arithmetic that motivates
the KDA/MLA hybrid (§2.1.2)."""

from __future__ import annotations

import pytest
import torch

from archlab.attention.mla import MLA
from archlab.memory import (
    kda_state_bytes,
    latent_kv_cache_bytes,
    mha_kv_cache_bytes,
)

K3_HEADS, K3_DHEAD, K3_LATENT = 96, 128, 512


class TestCacheGrowth:
    """Why 3-in-4 layers are KDA: at 1M tokens, even a compressed cache is the whole budget."""

    @pytest.mark.parametrize("seq_len", [128_000, 1_000_000])
    def test_latent_cache_beats_full_mha(self, seq_len: int) -> None:
        latent = latent_kv_cache_bytes(seq_len, K3_LATENT)
        full = mha_kv_cache_bytes(seq_len, K3_HEADS, K3_DHEAD)
        assert latent < full / 40

    def test_kda_state_undercuts_even_the_latent_cache(self) -> None:
        """`kda_state_bytes` takes no seq_len — constancy is in the signature, so measure the
        gap instead: at 1M tokens the fixed state is orders of magnitude below MLA's cache."""
        state = kda_state_bytes(K3_HEADS, K3_DHEAD, K3_DHEAD)
        assert state < latent_kv_cache_bytes(1_000_000, K3_LATENT) / 100


class TestMLA:
    def test_output_shape_and_cache(self) -> None:
        torch.manual_seed(0)
        layer = MLA(d_model=32, n_heads=4, d_head=8, d_latent=12)
        x = torch.randn(2, 10, 32)
        out, cache = layer(x)
        assert out.shape == x.shape
        assert cache.shape == (2, 10, 12)  # one latent per token, shared across heads

    def test_is_causal(self) -> None:
        torch.manual_seed(0)
        layer = MLA(d_model=32, n_heads=4, d_head=8, d_latent=12).eval()
        x = torch.randn(1, 12, 32)
        perturbed = x.clone()
        perturbed[:, 8:] += 5.0
        with torch.no_grad():
            a, _ = layer(x)
            b, _ = layer(perturbed)
        torch.testing.assert_close(a[:, :8], b[:, :8])

    def test_incremental_decode_matches_full_pass(self) -> None:
        """Cached decoding must reproduce the single-shot result — the serving invariant."""
        torch.manual_seed(0)
        layer = MLA(d_model=32, n_heads=4, d_head=8, d_latent=12).eval()
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            full, _ = layer(x)
            prefix, cache = layer(x[:, :4])
            step, _ = layer(x[:, 4:], latent_cache=cache)
        torch.testing.assert_close(full[:, :4], prefix, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(full[:, 4:], step, rtol=1e-5, atol=1e-5)

    def test_has_no_positional_parameters(self) -> None:
        """NoPE: nothing to retune when the context window grows 8x (§3.4)."""
        layer = MLA(d_model=32, n_heads=4, d_head=8, d_latent=12)
        names = [n for n, _ in layer.named_parameters()]
        assert not any(tag in n.lower() for n in names for tag in ("rope", "pos", "freq", "alibi"))
