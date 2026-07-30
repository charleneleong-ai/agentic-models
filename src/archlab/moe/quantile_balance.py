"""Quantile Balancing — set each expert's routing bias to the quantile matching its load.

Reference: K3 tech report §2.3.3 (Eq. 13-14), Appendix C (derivation), Appendix D (estimator).

Auxiliary-loss-free routing adds a per-expert bias b_j to the score used for Top-k selection,
but *omits* it from the mixture weights, so it steers dispatch without touching the router's
gradients:

    T_i = argtopk(s_i + b),    p_ij = s_ij / sum_{r in T_i} s_ir            (Eq. 13)

DeepSeek-V3 nudges b by a fixed-step sign rule, b <- b + gamma * sign(load_error). That is a
SignSGD step, and gamma trades adaptation speed against oscillation. It was adequate for 384
experts. LatentMoE takes K3 to 896 experts per layer, where slow equilibration means experts
that stay poorly trained and expert-parallel ranks that stay imbalanced.

QB jumps straight to the minimizer instead of stepping toward it. Appendix C shows the
balanced-assignment LP has an exact dual whose coordinate minimizers are quantiles:

    alpha_i* = quantile_{1-k/n}(s_i - beta)      (token side)
    beta_j*  = quantile_{1-k/n}(s_:,j - alpha)   (expert side)

Both updates are the same quantile along different axes — hence the name. The routing bias
is b = -beta; the token thresholds alpha are intermediates and are discarded, which is what
keeps train and inference consistent: at deployment the bias is frozen and routing is a
plain Top-k with no quantile computation anywhere.

Practically it is one extra forward pass: run Top-(k+1) on the biased score, take the first
k entries as the actual routes and the (k+1)-th as the cutoff alpha_i.
"""

from __future__ import annotations

import torch
from torch import Tensor


def routing_scores(x: Tensor, w_router: Tensor) -> Tensor:
    """Sigmoid router scores, s in (0, 1)^{m x n} — not a softmax, so experts are independent."""
    return torch.sigmoid(x @ w_router.T)


def topk_route(scores: Tensor, bias: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
    """Eq. 13. Returns (expert indices, mixture weights, per-token cutoff alpha).

    The cutoff comes free from Top-(k+1): it is the margin an expert must exceed to enter
    this token's Top-k, so no separate token-side quantile pass is needed.
    """
    n = scores.shape[-1]
    if k >= n:
        raise ValueError(
            f"n_active={k} must be < n_experts={n}: the cutoff is the (k+1)-th entry of "
            "Top-(k+1), so a token must leave at least one expert unselected"
        )
    biased = scores + bias
    _, idx = torch.topk(biased, k + 1, dim=-1)
    routed, cutoff_idx = idx[:, :k], idx[:, k]
    alpha = biased.gather(-1, cutoff_idx.unsqueeze(-1)).squeeze(-1)

    gathered = scores.gather(-1, routed)
    return routed, gathered / gathered.sum(-1, keepdim=True), alpha


def quantile_balance_update(scores: Tensor, alpha: Tensor, k: int) -> Tensor:
    """Eq. 14: exact per-expert bias from the (1 - k/n)-quantile of the margins, mean-centred.

    Margins s_:,j - alpha subtract the *biased* cutoff, so the old bias enters only through
    alpha; the update leaves Top-k unchanged and takes effect on the next step (causality —
    a batch is never routed with a bias derived from itself).
    """
    n = scores.shape[-1]
    margins = scores - alpha.unsqueeze(-1)  # (m, n)
    bias = -torch.quantile(margins, 1.0 - k / n, dim=0)
    return bias - bias.mean()


def alternating_qb_solve(scores: Tensor, k: int, iters: int = 8) -> Tensor:
    """Appendix C, Alg. 1: alternate the two quantile updates to the balanced assignment.

    This is the offline reference the online single-pass update approximates. Returns the
    expert bias b = -beta.
    """
    m, n = scores.shape
    beta = torch.zeros(n, dtype=scores.dtype, device=scores.device)
    alpha = torch.zeros(m, dtype=scores.dtype, device=scores.device)
    for _ in range(iters):
        alpha = torch.quantile(scores - beta, 1.0 - k / n, dim=1)
        beta = torch.quantile(scores - alpha.unsqueeze(-1), 1.0 - k / n, dim=0)
    return -(beta - beta.mean())


def histogram_quantile_bias(scores: Tensor, alpha: Tensor, k: int, n_bins: int = 1000) -> Tensor:
    """Appendix D: the estimator that makes Eq. 14 affordable at scale.

    An exact quantile would need all O(mn) margins gathered across data-parallel ranks and
    accumulation steps. But the update only needs each expert's *distribution*, which a
    binned histogram summarizes at fixed cost — and counts are additive, so a single
    all-reduce of per-rank bin counts yields the quantile of the true pooled global batch,
    invariant to how tokens were sharded. Communication is O(n * n_bins) per layer per step,
    independent of m.

    Error is bounded by the bin width; at n_bins = 1000 that is ~1e-3.
    """
    m, n = scores.shape
    required = alpha.unsqueeze(-1) - scores  # r_ij: bias placing expert j exactly at i's cutoff

    lo, hi = required.min().item(), required.max().item()
    width = (hi - lo) / n_bins if hi > lo else 1.0

    idx = ((required - lo) / width).long().clamp(0, n_bins - 1)  # (m, n)
    counts = torch.zeros(n, n_bins, dtype=torch.long, device=scores.device)
    counts.scatter_add_(1, idx.T, torch.ones_like(idx.T))  # the all-reduced quantity

    target = int(m * k / n)  # each expert's target load q
    cum = counts.cumsum(dim=1)
    bin_idx = (cum < max(target, 1)).sum(dim=1).clamp(max=n_bins - 1)

    before = torch.gather(cum - counts, 1, bin_idx.unsqueeze(-1)).squeeze(-1)
    height = torch.gather(counts, 1, bin_idx.unsqueeze(-1)).squeeze(-1).clamp(min=1)
    frac = ((target - before).float() / height.float()).clamp(0.0, 1.0)

    bias = lo + (bin_idx.float() + frac) * width
    return bias - bias.mean()


def load_imbalance(expert_counts: Tensor) -> float:
    """Max-over-mean load ratio. 1.0 is perfect balance."""
    return (expert_counts.float().max() / expert_counts.float().mean()).item()


def expert_load(routed: Tensor, n_experts: int) -> Tensor:
    """Tokens dispatched to each expert."""
    return torch.bincount(routed.flatten(), minlength=n_experts)
