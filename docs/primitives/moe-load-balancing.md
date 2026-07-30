# MoE load balancing — from auxiliary losses to exact quantiles

Routing is a bipartite assignment problem wearing a Top-k costume. Every method below is an
answer to: *how do you keep experts evenly loaded without corrupting what the router learns?*

## Why balance at all

Two distinct failures, often conflated:

1. **Systems.** Under expert parallelism, an overloaded rank sets the step time for every
   rank. Imbalance is a direct throughput tax, and dynamically varying shapes fragment memory.
2. **Learning.** Underutilized experts receive little gradient and stay poorly trained;
   in the limit they die. Capacity you paid for and cannot use.

The second is why "just drop overflow tokens" is not a fix.

## The progression

| Method | Mechanism | Cost |
|---|---|---|
| Auxiliary loss | Add a balance penalty to the training objective | Corrupts gradients — trades quality for balance |
| Expert choice | Invert selection: experts pick tokens | Balanced by construction, but breaks causality in decoding |
| BASE / BIP | Solve the assignment exactly (Hungarian / integer program) | Correct, too slow inside a training step |
| Loss-free bias (DeepSeek-V3) | `b ← b + γ·sign(load_error)`, omitted from mixture weights | No gradient corruption, but γ trades speed against oscillation |
| **Quantile Balancing (K3)** | Set `b` to the exact dual minimizer — a quantile | No hyperparameter, converges in a few steps |

The pivotal idea is **loss-free bias**: add `b_j` to the score used for *selection* but omit it
from the mixture weights `p_ij`. Dispatch is steered; the router's gradients never see it.
See [`topk_route`](../../src/archlab/moe/quantile_balance.py) — the test suite asserts the
weights are exactly the renormalized raw scores, bias-free by construction.

## Why quantiles

QB's derivation (K3 Appendix C) is the part worth internalizing. Start from the balanced
assignment LP — maximize total routed score subject to each token taking k experts and each
expert serving mk/n tokens. The bipartite b-matching polytope is integral, so the linear
relaxation is *exact*. Take the dual, introduce multipliers `α` (token side) and `β` (expert
side), and minimize by exact coordinate descent. Both subproblems are piecewise linear, and
both minimizers turn out to be the **same quantile along different axes**:

```
α_i = quantile_{1-k/n}(s_i,: − β)     token cutoffs
β_j = quantile_{1-k/n}(s_:,j − α)     expert biases      → routing bias b = −β
```

Two consequences fall out:

- **The sign rule is the same algorithm, crippled.** The expert-side subgradient is
  `mk/n − observed_load`; SignSGD on it keeps only the *direction* of the load error and
  discards the magnitude. QB jumps to the coordinate minimizer instead of stepping toward it —
  which is why it needs no learning rate.
- **Train/inference consistency is free.** Only `β` becomes a persistent bias; the token
  thresholds `α` are batch-dependent intermediates and get discarded. At deployment the bias
  is frozen and routing is plain Top-k — no quantile computation anywhere.

The cutoff `α` doesn't even cost an extra pass: run Top-(k+1) instead of Top-k and the
(k+1)-th entry *is* the cutoff.

## Making it affordable

An exact quantile needs all O(mn) margins gathered across data-parallel ranks and accumulation
steps — impractical inside a training step. The fix (Appendix D) is that the update only needs
each expert's *distribution*, which a binned histogram summarizes at fixed cost.

Histogram counts are **additive**, so a single all-reduce of per-rank bin counts yields the
quantile of the true pooled global batch — invariant to how tokens were sharded, and strictly
better than averaging per-rank quantiles. Communication is O(n·bins), independent of token
count.

> This is a reusable pattern: *any* distributed statistic currently requiring a gather is worth
> checking for an additive sufficient statistic.

## Two caveats found by testing, not in the report

Both from [`test_moe.py`](../../tests/test_moe.py):

- **Ties break the guarantee.** The derivation assumes no ties. A saturated sigmoid router
  emits scores of exactly 0.0/1.0; balance then plateaus near 1.29× max/mean instead of
  ~1.01×, because no threshold can split a tied group. Router temperature is load-bearing for
  balancing quality, not just for routing quality.
- **The histogram has an error floor.** Below ~4k bins, error is bin-width-bounded as claimed.
  Past that, the dominant term is rank discretization — the estimator lands on an integer rank
  while the interpolated quantile sits between ranks. More bins buy nothing.

## Open question

Balance is a *constraint*, and the report measures balance, not specialization. Forcing uniform
load may actively suppress genuinely popular experts. What does perfect balance cost in expert
specialization quality? Nothing published measures this.
