"""Stable LatentMoE — 896 experts made affordable by routing in a compressed latent space.

Reference: K3 tech report §2.3 (Eq. 11), §2.3.1.

Widening the expert pool buys more specialization, but in a conventional MoE every selected
expert receives the full d-dimensional token, so communication and expert-weight traffic grow
with routing multiplicity. LatentMoE separates the widths: *shared* experts keep a full-width
path for transformations every token needs, while *routed* experts work in a compact latent
space of width l. K3 sets l = d/2 and reaches 896 routed experts with 16 active — sparsity 56.

    u = sum_{i in Topk(x)} p_i * E_i^routed(W_down x)
    y = sum_j E_j^shared(x) + W_up * RMSNorm(u)                            (Eq. 11)

Two failure modes appear at this sparsity, and "Stable" names the fixes for both.

1. Exploding activations. The routed path chains W_down, a gated multi-branch expert FFN, and
   W_up into nearly four consecutive matmuls. That is ill-conditioned, and at 2.8T parameters
   it blows up. Fixes: an RMSNorm between expert aggregation and the up-projection (§2.3.1),
   which decouples the routed branch's scale — it varies with *which* experts fired and with
   their routing weights — from the full-width shared branch it is about to be added to; and
   SiTU-GLU inside the experts to cap the activation itself.

2. Load balancing at ~10^3 experts. Fixed-step sign updates equilibrate too slowly; see
   `quantile_balance` for what replaces them.

The RMSNorm was not merely a stability patch — the report notes it consistently improves
validation loss and downstream benchmarks on its own.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor, nn

from archlab.activations.situ_glu import SiTUGLU
from archlab.moe.quantile_balance import (
    histogram_quantile_bias,
    quantile_balance_update,
    routing_scores,
    topk_route,
)

BiasEstimator = Callable[[Tensor, Tensor, int], Tensor]


class Routing(NamedTuple):
    """What the router decided, in the form `balance_step` consumes.

    `indices` keeps the token layout of the input; `scores` and `alpha` stay flat because the
    balancing update is a reduction over tokens and does not care about batch structure.
    """

    indices: Tensor
    scores: Tensor
    alpha: Tensor


class LatentMoE(nn.Module):
    """Eq. 11. Routed experts operate at `d_latent`; shared experts at full `d_model`."""

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        d_expert_hidden: int,
        n_routed: int,
        n_active: int,
        n_shared: int = 2,
        bias_estimator: BiasEstimator = quantile_balance_update,
    ) -> None:
        super().__init__()
        self.n_routed, self.n_active = n_routed, n_active
        # Swappable so Appendix D's histogram estimator is reachable from the module, not
        # only from tests — it is the form that actually scales.
        self.bias_estimator = bias_estimator

        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.register_buffer("bias", torch.zeros(n_routed))  # frozen at inference

        self.down = nn.Linear(d_model, d_latent, bias=False)
        self.experts = nn.ModuleList(SiTUGLU(d_latent, d_expert_hidden) for _ in range(n_routed))
        self.norm = nn.RMSNorm(d_latent)  # §2.3.1 — the "Stable" in Stable LatentMoE
        self.up = nn.Linear(d_latent, d_model, bias=False)

        self.shared = nn.ModuleList(SiTUGLU(d_model, d_expert_hidden) for _ in range(n_shared))

    def sparsity(self) -> float:
        return self.n_routed / self.n_active

    @torch.no_grad()
    def balance_step(self, routing: Routing) -> None:
        """Apply one balancing update. Kept off `forward` because it is a training-loop
        concern: the real update reduces over microbatches and ranks, and takes effect on the
        *next* step — a batch is never routed with a bias derived from itself (§2.3.3).

        `no_grad` is load-bearing, not defensive. `routing.scores` carries a graph, so copying
        an estimate derived from it into the buffer would leave `bias` holding a `grad_fn` —
        pinning that step's whole graph alive for the rest of training, and making a
        non-learnable buffer look learnable. The bias is *defined* to sit outside gradient
        flow (Eq. 13 omits it from the mixture weights); this keeps it there.
        """
        estimate = self.bias_estimator(routing.scores, routing.alpha, self.n_active)
        self.bias.copy_(estimate.detach())

    def forward(self, x: Tensor) -> tuple[Tensor, Routing]:
        flat = x.reshape(-1, x.shape[-1])
        scores = routing_scores(flat, self.router.weight)
        routed, weights, alpha = topk_route(scores, self.bias, self.n_active)

        latent = self.down(flat)
        u = torch.zeros_like(latent)
        for e in range(self.n_routed):  # nano-scale dispatch; real kernels use grouped GEMMs
            tok, slot = (routed == e).nonzero(as_tuple=True)
            if tok.numel():
                u.index_add_(
                    0, tok, weights[tok, slot].unsqueeze(-1) * self.experts[e](latent[tok])
                )

        out = self.up(self.norm(u))
        for expert in self.shared:
            out = out + expert(flat)

        return out.view_as(x), Routing(routed.view(*x.shape[:-1], self.n_active), scores, alpha)


def active_param_fraction(n_routed: int, n_active: int, n_shared: int) -> float:
    """Fraction of expert parameters touched per token — K3: 16 of 896 routed, plus 2 shared."""
    return (n_active + n_shared) / (n_routed + n_shared)
