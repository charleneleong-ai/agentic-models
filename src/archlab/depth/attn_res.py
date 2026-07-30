"""Attention Residuals — attention over *depth* instead of a single accumulated stream.

Reference: K3 tech report §2.2 (Eq. 8-10).

The argument is a neat historical rhyme. A standard residual stream compresses everything
prior into one state h_l and adds to it — which is exactly what an RNN does over *time*, and
exactly the bottleneck attention was invented to remove. Sequence modelling already made
that move: attention lets each position reach any earlier position with data-dependent
weights. AttnRes makes the same move over depth: each layer *retrieves* from all preceding
layers rather than reading a uniform running sum.

Full AttnRes (Eq. 8-9). Layer l holds a learnable pseudo-query q_l = w_l; keys and values are
the layer outputs themselves (with the token embedding h_1 as source 0). Weights use an
exponential kernel phi(q, k) = exp(q^T RMSNorm(k)); the RMSNorm is load-bearing, stopping
layers with large-magnitude outputs from dominating purely on scale.

Cost. Depth is modest (L < 100), so the O(L^2 d) arithmetic is free. The real price is O(Ld)
*memory* — every layer output must stay live — plus cross-stage traffic under pipeline
parallelism. Hence:

Block AttnRes (Eq. 10). Partition L layers into N blocks of S. Inside a block, collapse
outputs to a running partial sum (cheap, uniform — an ordinary residual). Across blocks,
apply full attention over only the N block representations. Memory drops O(Ld) -> O(Nd).
Empirically N ~ 8 recovers most of the benefit; K3 uses 8 blocks of 12 layers.

The block structure also bounds the *inference-time* state, and lets parallel inter-block
results merge with the sequential intra-block partial sum via online softmax — which is what
makes the serving cost tolerable (§5.4.2).

Both classes hold every source they attend over, so `FullAttnRes` retains O(L) tensors where
`BlockAttnRes` retains O(N). That contrast is the point, and `live_sources` makes it
observable rather than asserted.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DepthAttention(nn.Module):
    """Shared Eq. 8-9 kernel. Subclasses differ only in *what the source list is*."""

    def __init__(self, d_model: int, n_queries: int) -> None:
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.randn(n_queries, d_model) * d_model**-0.5)
        self.norm = nn.RMSNorm(d_model)

    def attention_weights(self, sources: list[Tensor], layer_idx: int) -> Tensor:
        """Eq. 9. Exposed because the paper's claims are about the *weights*, not the output."""
        n_queries = self.pseudo_query.shape[0]
        if not 0 <= layer_idx < n_queries:
            raise IndexError(
                f"layer_idx={layer_idx} outside the {n_queries} pseudo-queries this module "
                "was built for — each layer needs its own learnable query (Eq. 8)"
            )
        stacked = torch.stack(sources, dim=-2)
        logits = torch.einsum("d,btsd->bts", self.pseudo_query[layer_idx], self.norm(stacked))
        return torch.softmax(logits, dim=-1)

    def retrieve(self, sources: list[Tensor], layer_idx: int) -> Tensor:
        stacked = torch.stack(sources, dim=-2)
        return torch.einsum("bts,btsd->btd", self.attention_weights(sources, layer_idx), stacked)


class FullAttnRes(DepthAttention):
    """Eq. 8-9: every layer attends over all preceding layer outputs.

    Sources are ordered [embedding, layer_1_out, ..., layer_{l-1}_out] and grow with depth.
    """

    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__(d_model, n_layers)
        self.sources: list[Tensor] = []

    def live_sources(self) -> int:
        return len(self.sources)

    def forward(self, embedding: Tensor, layers: list[nn.Module]) -> Tensor:
        self.sources = [embedding]  # b_0 — the token embedding is always a source
        h = embedding
        for idx, layer in enumerate(layers):
            self.sources.append(layer(h))
            h = self.retrieve(self.sources, idx)
        return h


class BlockAttnRes(DepthAttention):
    """Eq. 10: summation inside a block, attention across block representations.

    The module owns the blocking. Feed it one layer output at a time via `step`; it derives
    block boundaries from `block_size`, folds outputs into the running partial sum, and rolls
    a completed partial into the block states.
    """

    def __init__(self, d_model: int, n_blocks: int, block_size: int) -> None:
        super().__init__(d_model, n_blocks * block_size)
        self.n_blocks, self.block_size = n_blocks, block_size
        self.block_states: list[Tensor] = []
        self.partial: Tensor | None = None

    def reset(self, embedding: Tensor) -> None:
        """b_0 = the token embedding, so it is always among the sources (Eq. 10)."""
        self.block_states, self.partial = [embedding], None

    def live_sources(self) -> int:
        """Tensors currently retained — the quantity Block AttnRes bounds at O(N)."""
        return len(self.block_states) + (self.partial is not None)

    def sources(self) -> list[Tensor]:
        return self.block_states if self.partial is None else [*self.block_states, self.partial]

    def step(self, layer_out: Tensor, layer_idx: int) -> Tensor:
        """Absorb one layer's output, then retrieve for the next layer.

        Intra-block reduction is plain summation — a uniform residual, by design. On a block
        boundary the completed partial becomes a block representation and the sum restarts,
        which is why a block's first layer sees only [b_0..b_{n-1}] while later layers also
        see the partial.
        """
        self.partial = layer_out if self.partial is None else self.partial + layer_out
        if (layer_idx + 1) % self.block_size == 0:
            self.block_states.append(self.partial)
            self.partial = None
        return self.retrieve(self.sources(), layer_idx)

    def forward(self, embedding: Tensor, layers: list[nn.Module]) -> Tensor:
        self.reset(embedding)
        h = embedding
        for idx, layer in enumerate(layers):
            h = self.step(layer(h), idx)
        return h
