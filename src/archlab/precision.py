"""Simulated low-precision activations, for testing what SiTU-GLU's bound actually buys.

K3 justifies SiTU-GLU on precision grounds (§2.3.2): SwiGLU multiplies two *unbounded* factors,
so coincident large coordinates compound into outliers, and at MXFP4 weights / MXFP8 activations
an outlier is an overflow. SiTU-GLU bounds output at b1*b2 = 100. The report shows no quality
comparison, and no measurement of whether the unbounded baseline actually overflows.

This module makes that testable on hardware without FP8 tensor cores. An A100 is sm_80: the
float8 *dtypes* exist and casting works, but `torch._scaled_mm` requires sm_90, so real FP8
matmuls are unavailable. Fake quantization — cast to float8 and back — reproduces the format's
**range and precision** exactly while doing the arithmetic in bf16.

That is the right tool for this question and the wrong one for a different question. The claim
under test is numerical (does the activation leave the representable range?), and fake quant
answers it faithfully. It says nothing about FP8 throughput, and no speed claim should be drawn
from it.

The property that matters: `float8_e4m3fn` represents at most **448.0**, and SiTU-GLU's bound
of b1*b2 = 100 sits comfortably inside it. Measured, PyTorch's cast *saturates* at 448 rather
than producing NaN — so overflow is silent information loss, not a crash. That is worse for an
experiment than a crash would be: nothing announces it, and only the activation statistics show
it happened.

One structural caveat this ablation has to respect. The FFN sits behind an RMSNorm, so its
*input* is unit-scale no matter what the residual stream does. Activation outliers therefore
cannot come from input magnitude; they can only come from learned weight magnitudes, which
require real training to develop. Any test that inflates inputs to "stress" the FFN is
measuring nothing — verified directly: scaling the embedding 25x leaves peak activations at
~2.0, unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

FP8_E4M3_MAX = 448.0
Quantizer = Callable[[Tensor], Tensor]


def identity(x: Tensor) -> Tensor:
    return x


def fake_fp8(x: Tensor, dtype: torch.dtype = torch.float8_e4m3fn) -> Tensor:
    """Round-trip through float8: exact range and precision, arithmetic still in bf16/fp32.

    Straight-through on the backward pass — the cast is not differentiable, and quantization
    aware training conventionally passes the gradient unchanged.
    """
    return x + (x.to(dtype).to(x.dtype) - x).detach()


def quantizer_for(precision: str) -> Quantizer:
    if precision == "bf16":
        return identity
    if precision == "fp8_sim":
        return fake_fp8
    raise ValueError(f"unknown precision: {precision!r} (expected 'bf16' or 'fp8_sim')")


class ActivationStats:
    """Tracks what the precision claim is actually about: magnitude and overflow.

    Accumulated across a run rather than sampled, because the failure mode is a *rare* coincident
    outlier — an average would hide exactly the event that matters.
    """

    def __init__(self) -> None:
        self.max_abs = 0.0
        self.n_over_fp8 = 0
        self.n_values = 0
        self.n_nonfinite = 0

    def observe(self, x: Tensor) -> None:
        with torch.no_grad():
            finite = torch.isfinite(x)
            self.n_nonfinite += int((~finite).sum())
            self.n_values += x.numel()
            if finite.any():
                self.max_abs = max(self.max_abs, float(x[finite].abs().max()))
            self.n_over_fp8 += int((x.abs() > FP8_E4M3_MAX).sum())

    def summary(self) -> dict[str, float]:
        return {
            "max_abs_activation": round(self.max_abs, 2),
            "fp8_overflow_frac": self.n_over_fp8 / max(1, self.n_values),
            "nonfinite_frac": self.n_nonfinite / max(1, self.n_values),
        }
