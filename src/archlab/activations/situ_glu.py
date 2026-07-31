"""SiTU-GLU — SwiGLU with both branches smoothly capped.

Reference: K3 tech report §2.3.2 (Eq. 12) and Appendix B.

SwiGLU multiplies two *unbounded* factors, so coincident large coordinates compound into
activation outliers. At 2.8T parameters with MXFP4 weights and MXFP8 activations, an outlier
is not a curiosity — it is an overflow. The original GLU's sigmoid gate is bounded, but
loses Swish's approximately-linear positive regime, which is the part that trains well.

SiTU-GLU applies a smooth cap softcap(x, b) = b * tanh(x / b) to the linear factor of the
Swish gate and, independently, to the up branch:

    SiTU-GLU(x) = [b1 * tanh(W_g x / b1) * Sigmoid(W_g x)] * [b2 * tanh(W_u x / b2)]

K3 uses b1 = 4 (gate), b2 = 25 (up). Two properties make this work (Appendix B):

  Local:    b * tanh(z / b) = z + O(z^3 / b^2), so SiTU-GLU matches SwiGLU to first order
            near the origin, and recovers it pointwise as b1, b2 -> inf.
  Bounded:  |tanh| < 1 and 0 < Sigmoid < 1, so every output coordinate obeys
            ||SiTU-GLU(x)||_inf <= b1 * b2 = 100.

Unlike hard clamping, the smooth cap keeps gradients nonzero away from the saturation
boundary — you get the bound without a dead zone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from archlab.precision import ActivationStats, Quantizer, identity

BETA_GATE = 4.0
BETA_UP = 25.0


def softcap(x: Tensor, beta: float) -> Tensor:
    """b * tanh(x / b): identity to first order near 0, asymptote at +/- b."""
    return beta * torch.tanh(x / beta)


def situ_glu(
    gate: Tensor, up: Tensor, beta_gate: float = BETA_GATE, beta_up: float = BETA_UP
) -> Tensor:
    """Eq. 12, on pre-activations. Output is bounded by beta_gate * beta_up."""
    return softcap(gate, beta_gate) * torch.sigmoid(gate) * softcap(up, beta_up)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """The unbounded baseline, for comparison: (x * Sigmoid(x)) * up."""
    return F.silu(gate) * up


class SiTUGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        beta_gate: float = BETA_GATE,
        beta_up: float = BETA_UP,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.beta_gate, self.beta_up = beta_gate, beta_up
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=bias)
        self.quantize: Quantizer = identity
        self.stats: ActivationStats | None = None

    def hidden(self, x: Tensor) -> Tensor:
        return situ_glu(self.gate_proj(x), self.up_proj(x), self.beta_gate, self.beta_up)

    def forward(self, x: Tensor) -> Tensor:
        h = self.hidden(x)
        if self.stats is not None:
            self.stats.observe(h)  # measured *before* quantization — the true magnitude
        return self.down_proj(self.quantize(h))
