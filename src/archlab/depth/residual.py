"""The ordinary residual stream — the baseline Attention Residuals is measured against.

Deliberately given the same `forward(embedding, layers)` signature as `FullAttnRes` and
`BlockAttnRes`. That shared interface is what makes the depth-axis ablation a one-word config
change rather than a second model implementation, and it keeps the comparison honest: the arms
differ in exactly one object.

Read alongside `attn_res.py`: this accumulates uniformly (h <- h + f(h), every layer weighted
identically and irrevocably), where AttnRes retrieves with learned, data-dependent weights.
"""

from __future__ import annotations

from torch import Tensor, nn


class ResidualStack(nn.Module):
    """h_l = h_{l-1} + f_l(h_{l-1}). One live tensor, regardless of depth."""

    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers

    @staticmethod
    def live_sources() -> int:
        """Always 1 — the whole point of the comparison against O(L) and O(N)."""
        return 1

    def forward(self, embedding: Tensor, layers: list[nn.Module]) -> Tensor:
        h = embedding
        for layer in layers:
            h = h + layer(h)
        return h
