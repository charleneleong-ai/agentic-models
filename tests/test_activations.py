"""SiTU-GLU: bounded output without losing SwiGLU's shape (Eq. 12, Appendix B)."""

from __future__ import annotations

import pytest
import torch

from archlab.activations.situ_glu import BETA_GATE, BETA_UP, SiTUGLU, situ_glu, softcap, swiglu


class TestSoftcap:
    @pytest.mark.parametrize("beta", [4.0, 25.0])
    def test_is_identity_to_first_order(self, beta: float) -> None:
        """b*tanh(z/b) = z + O(z^3/b^2) — indistinguishable from identity near the origin."""
        z = torch.linspace(-0.05, 0.05, 101)
        torch.testing.assert_close(softcap(z, beta), z, rtol=0, atol=1e-5)

    @pytest.mark.parametrize("beta", [4.0, 25.0])
    def test_saturates_at_beta(self, beta: float) -> None:
        assert softcap(torch.tensor([1e6]), beta).item() == pytest.approx(beta)

    def test_gradient_survives_past_the_cap(self) -> None:
        """A smooth cap has no hard dead zone: past beta, clamp is flat, softcap is not."""
        for value in (5.0, 8.0, 12.0):  # 1.25x to 3x the gate cap
            z = torch.tensor([value], requires_grad=True)
            softcap(z, BETA_GATE).backward()

            z_clamped = torch.tensor([value], requires_grad=True)
            z_clamped.clamp(max=BETA_GATE).backward()

            assert z.grad.item() > 0, f"softcap gradient vanished at {value}"
            assert z_clamped.grad.item() == 0

    def test_gradient_still_underflows_deep_in_saturation(self) -> None:
        """Honest limit: tanh reaches exactly 1.0 in float32, so its gradient hits exactly 0.

        The smooth cap buys a wide non-vanishing band, not an unconditional one — worth
        knowing before assuming softcap alone keeps a saturated unit trainable.
        """
        z = torch.tensor([50.0], requires_grad=True)
        softcap(z, BETA_GATE).backward()
        assert z.grad.item() == 0
        assert torch.tanh(torch.tensor(50.0 / BETA_GATE)).item() == 1.0


class TestSiTUGLU:
    def test_output_is_bounded(self) -> None:
        """||SiTU-GLU(x)||_inf <= b1*b2 = 100 (Eq. 19) — the property MXFP4/8 depends on."""
        torch.manual_seed(0)
        gate, up = torch.randn(2000, 64) * 500, torch.randn(2000, 64) * 500
        assert situ_glu(gate, up).abs().max().item() <= BETA_GATE * BETA_UP

    def test_swiglu_baseline_is_unbounded(self) -> None:
        """Contrast: coincident large coordinates compound without limit."""
        gate, up = torch.full((1, 1), 500.0), torch.full((1, 1), 500.0)
        assert swiglu(gate, up).item() > 1e5

    def test_matches_swiglu_near_origin(self) -> None:
        torch.manual_seed(0)
        gate, up = torch.randn(500, 32) * 0.01, torch.randn(500, 32) * 0.01
        torch.testing.assert_close(situ_glu(gate, up), swiglu(gate, up), rtol=0, atol=1e-6)

    def test_recovers_swiglu_as_beta_grows(self) -> None:
        """Eq. 12 reduces to SwiGLU pointwise as b1, b2 -> inf."""
        torch.manual_seed(0)
        gate, up = torch.randn(200, 16), torch.randn(200, 16)
        torch.testing.assert_close(
            situ_glu(gate, up, 1e6, 1e6), swiglu(gate, up), rtol=1e-5, atol=1e-5
        )

    def test_module_shape(self) -> None:
        x = torch.randn(2, 8, 32)
        assert SiTUGLU(32, 64)(x).shape == x.shape
