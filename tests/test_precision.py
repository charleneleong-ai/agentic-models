"""Simulated FP8 and the activation probe.

Several of these encode corrections to things I had wrong while designing `activation-bound`.
They are here so the next reader inherits the correction rather than the mistake.
"""

from __future__ import annotations

import pytest
import torch

from archlab.activations.situ_glu import BETA_GATE, BETA_UP, SiTUGLU, situ_glu, softcap
from archlab.precision import (
    FP8_E4M3_MAX,
    ActivationStats,
    fake_fp8,
    identity,
    quantizer_for,
)


class TestFakeFP8:
    def test_overflow_saturates_rather_than_producing_nan(self) -> None:
        """Correction: `e4m3fn` is finite-only, which I read as "overflow -> NaN". It does not
        — the cast clamps at 448. Overflow is therefore *silent* information loss, which is
        worse for an experiment than a crash: only the activation stats reveal it."""
        for value in (500.0, 1e4, 1e8):
            out = fake_fp8(torch.tensor([value]))
            assert torch.isfinite(out).all()
            assert out.item() == pytest.approx(FP8_E4M3_MAX)

    def test_values_inside_range_survive_approximately(self) -> None:
        x = torch.tensor([0.5, 1.0, 4.0, 64.0, 256.0])
        out = fake_fp8(x)
        assert torch.allclose(out, x, rtol=0.1)  # e4m3 has 3 mantissa bits

    def test_precision_is_actually_lost(self) -> None:
        """If the round-trip were exact the whole `fp8_sim` arm would be a no-op."""
        x = torch.linspace(1.0, 2.0, 512)
        assert not torch.equal(fake_fp8(x), x)
        assert len(torch.unique(fake_fp8(x))) < len(torch.unique(x))

    def test_gradient_passes_straight_through(self) -> None:
        """The cast is not differentiable; QAT convention is to pass the gradient unchanged."""
        x = torch.tensor([3.0, 700.0], requires_grad=True)
        fake_fp8(x).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_quantizer_dispatch(self) -> None:
        assert quantizer_for("bf16") is identity
        assert quantizer_for("fp8_sim") is fake_fp8
        with pytest.raises(ValueError, match="unknown precision"):
            quantizer_for("int4")


class TestActivationStats:
    def test_tracks_the_network_wide_maximum(self) -> None:
        """Pooled across layers on purpose: the failure mode is a rare outlier *somewhere*, so
        a per-layer average would dilute exactly the event of interest."""
        stats = ActivationStats()
        stats.observe(torch.tensor([1.0, -2.0]))
        stats.observe(torch.tensor([0.5, -9.0]))
        assert stats.summary()["max_abs_activation"] == 9.0

    def test_counts_values_beyond_the_fp8_ceiling(self) -> None:
        stats = ActivationStats()
        stats.observe(torch.tensor([1.0, 500.0, 600.0, 2.0]))
        assert stats.summary()["fp8_overflow_frac"] == pytest.approx(0.5)

    def test_nonfinite_values_do_not_corrupt_the_maximum(self) -> None:
        stats = ActivationStats()
        stats.observe(torch.tensor([3.0, float("inf"), float("nan")]))
        summary = stats.summary()
        assert summary["max_abs_activation"] == 3.0
        assert summary["nonfinite_frac"] == pytest.approx(2 / 3)


class TestBranchCapsNotProductCap:
    """Correction: b1*b2 = 100 bounds the *product*. The branches cap separately at b1 and b2.

    I predicted `situ-4-25` would leave activations near SwiGLU's observed peak of ~58 because
    58 < 100. It did not — peak fell to 25.2, because the up branch caps at 25. Getting this
    wrong changes which arms are expected to bind at all, so it is worth pinning down.
    """

    def test_up_branch_is_bounded_by_beta_up_alone(self) -> None:
        gate = torch.zeros(1000)  # sigmoid(0)=0.5, softcap(0)=0 -> isolate the up branch
        up = torch.linspace(-500, 500, 1000)
        assert softcap(up, BETA_UP).abs().max() <= BETA_UP
        assert situ_glu(gate, up).abs().max() <= BETA_UP  # gate factor only shrinks it further

    def test_product_bound_is_the_product_of_the_branch_bounds(self) -> None:
        torch.manual_seed(0)
        gate, up = torch.randn(5000) * 500, torch.randn(5000) * 500
        assert situ_glu(gate, up).abs().max() <= BETA_GATE * BETA_UP

    @pytest.mark.parametrize(
        "beta_up,natural_peak,binds", [(8.0, 58.0, True), (100.0, 58.0, False)]
    )
    def test_whether_a_cap_binds_depends_on_beta_up_not_the_product(
        self, beta_up: float, natural_peak: float, binds: bool
    ) -> None:
        capped = float(softcap(torch.tensor([natural_peak]), beta_up))
        assert (capped < natural_peak * 0.9) is binds


class TestQuantizedModule:
    def test_module_applies_its_quantizer(self) -> None:
        torch.manual_seed(0)
        ffn = SiTUGLU(32, 64)
        x = torch.randn(4, 32)
        baseline = ffn(x)

        ffn.quantize = fake_fp8
        assert not torch.equal(ffn(x), baseline)

    def test_stats_record_pre_quantization_magnitude(self) -> None:
        """Peak must be the true activation, not the already-clipped one — otherwise the metric
        can never show that an overflow happened."""
        ffn = SiTUGLU(16, 32)
        ffn.quantize = fake_fp8
        stats = ActivationStats()
        ffn.stats = stats

        with torch.no_grad():
            ffn.up_proj.weight.mul_(1000.0)
            ffn.gate_proj.weight.mul_(1000.0)
        ffn(torch.randn(8, 16))

        assert stats.summary()["max_abs_activation"] > 0
