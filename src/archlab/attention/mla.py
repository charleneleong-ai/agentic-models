"""Multi-head Latent Attention — global attention with a compressed KV cache.

Reference: K3 tech report §2.1.2; MLA originates in DeepSeek-V2.

MLA caches a low-rank latent c_t = W_c x_t instead of full per-head keys and values, then
reconstructs K and V through learned up-projections at attention time. The cache shrinks by
the compression ratio while attention stays *global* — every token still sees every other.
See `archlab.memory` for the byte accounting that motivates it.

In K3 this is the 1-in-4 layer. The other three are KDA (fixed-size recurrent state, cheap
but lossy mixing); MLA periodically restores unrestricted token-to-token interaction. The
hybrid is the whole design: KDA carries the sequence cheaply, MLA supplies exactness.

Two K3-specific departures from DeepSeek-V2 MLA:

NoPE. No positional encoding on queries or keys, anywhere. Position information reaches the
model only through KDA's decay and gating. This is why K3 extrapolates to 1M tokens with no
RoPE rescaling, no YaRN, no retuning of a frequency base — there is no frequency base to
retune. Long-context extension becomes a data-and-curriculum problem, not a surgery problem.

Full-rank output gate — Eq. 7, the same `GatedOutput` mechanism KDA applies in Eq. 6.

Note this is the naive formulation: it up-projects the whole cached prefix on every call, so
incremental decode is O(T) per step and transiently materializes full K and V. Production
kernels absorb W_k into W_q and score against the latents directly, realizing the saving at
inference as well as in the cache. That trick reads far worse, so it is not here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from archlab.attention.heads import GatedOutput, causal_mask, split_heads


class MLA(nn.Module):
    """Latent-KV global attention, NoPE, with a full-rank output gate."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, d_latent: int) -> None:
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_head
        inner = n_heads * d_head

        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.kv_down = nn.Linear(d_model, d_latent, bias=False)  # the cached quantity
        self.k_up = nn.Linear(d_latent, inner, bias=False)
        self.v_up = nn.Linear(d_latent, inner, bias=False)

        self.gated_out = GatedOutput(d_model, inner)  # Eq. 7

    def split_heads(self, x: Tensor) -> Tensor:
        return split_heads(x, self.n_heads, self.d_head)

    def forward(self, x: Tensor, latent_cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        c = self.kv_down(x)
        if latent_cache is not None:
            c = torch.cat([latent_cache, c], dim=1)

        q = self.split_heads(self.q_proj(x))
        k, v = self.split_heads(self.k_up(c)), self.split_heads(self.v_up(c))

        mask = causal_mask(q.shape[-2], k.shape[-2], x.device)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        out = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ v

        return self.gated_out(out, x), c
