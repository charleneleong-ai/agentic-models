"""Head plumbing and the output gate shared by every attention layer here.

The attention *cores* differ genuinely — KDA is a recurrence, MLA is a softmax — and keeping
them separate is the point. What surrounds them does not differ: both split into heads, merge
back, and apply the same input-dependent full-rank gate, which the report presents as one
mechanism in two places (Eq. 6 for KDA, Eq. 7 for MLA). It lives here once so it is testable
as an object rather than as incidental code in two layers.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def split_heads(x: Tensor, n_heads: int, d_head: int) -> Tensor:
    """(B, T, n_heads*d_head) -> (B, n_heads, T, d_head)."""
    b, t, _ = x.shape
    return x.view(b, t, n_heads, d_head).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """(B, n_heads, T, d_head) -> (B, T, n_heads*d_head)."""
    b, _, t, _ = x.shape
    return x.transpose(1, 2).reshape(b, t, -1)


def causal_mask(t_q: int, t_kv: int, device: torch.device, diagonal: int = 0) -> Tensor:
    """Boolean mask, right-aligned so queries are the last `t_q` positions of `t_kv`.

    `diagonal=0` retains self-interaction; `diagonal=-1` excludes it. That choice is the
    subtlest decision in the chunkwise KDA algebra, so it is a named argument, not a literal.
    """
    offset = t_kv - t_q
    rows = torch.arange(t_q, device=device).unsqueeze(-1) + offset + diagonal
    return rows >= torch.arange(t_kv, device=device).unsqueeze(0)


class GatedOutput(nn.Module):
    """y = W_o[Sigmoid(W_g x) * o] — Eq. 6 / Eq. 7.

    The gate is full-rank and input-dependent, so each token modulates which channels it reads
    out of attention. K3 replaced KDA's low-rank Kimi Linear gate with exactly this.
    """

    def __init__(self, d_model: int, d_inner: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, attn_out: Tensor, x: Tensor) -> Tensor:
        return self.out_proj(torch.sigmoid(self.gate_proj(x)) * merge_heads(attn_out))
