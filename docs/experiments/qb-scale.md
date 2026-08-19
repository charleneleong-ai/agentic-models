# qb-scale — does Quantile Balancing's edge widen with expert count?

**Spec:** [`configs/ablations/qb-scale.yaml`](../../configs/ablations/qb-scale.yaml) ·
**Runner:** [`archlab/ablations/qb_scale.py`](../../src/archlab/ablations/qb_scale.py) ·
**Run:** 2026-07-30, CPU, 77 s, 75 rows

## Hypothesis

K3 §2.3.3 motivates replacing DeepSeek-V3's fixed-step sign updates by saying they equilibrate
too slowly *"as LatentMoE increases the routed expert pool to 896 per layer"*. That is a claim
about a **trend**, not a single operating point. If the QB-vs-sign gap is flat in `n`, the
justification is weaker than stated — QB would be better, but not *because* of scale.

**Falsifier:** if the gap ratio `(sign − 1) / (QB − 1)` does not grow with `n`, the
scale-specific framing is wrong.

Fairness: QB has no learning rate, so sign-SGD's `gamma` is swept over five decades and its
**best** cell reported — a tuning budget QB never receives. Without that the baseline would be
sandbagged and the result meaningless.

## Results

Final max/mean expert load after 8 update steps; 1.000 is perfect. 16,384 tokens, top-16.

**Unsaturated router (`router_scale=0.15`, no ties)**

| n_experts | none | sign-SGD | qb-exact | qb-histogram | qb-alternating | gap ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.397 | 1.224 | **1.002** | 1.006 | 1.000 | 149× |
| 128 | 1.916 | 1.397 | **1.004** | 1.018 | 1.001 | 90× |
| 256 | 2.582 | 1.556 | **1.008** | 1.020 | 1.003 | 71× |
| 448 | 3.317 | 1.600 | **1.010** | 1.029 | 1.003 | 60× |
| 896 | 4.529 | 1.863 | **1.022** | 1.039 | 1.008 | 39× |

**Saturated router (`router_scale=1.0`, 1.9% of scores pinned at exactly 0.0/1.0)**

| n_experts | none | sign-SGD | qb-exact | qb-histogram | qb-alternating |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.212 | 1.132 | **1.025** | 1.034 | 1.000 |
| 128 | 1.437 | 1.303 | **1.033** | 1.130 | 1.002 |
| 256 | 1.699 | 1.699 | **1.053** | 1.597 | 1.029 |
| 448 | 2.080 | 2.080 | **1.227** | 2.196 | 1.193 |
| 896 | 2.622 | **2.123** | 2.396 | 3.377 | 2.396 |

**Steps to within 5% of uniform** (budget 8; `—` = never)

| | n=64 | 128 | 256 | 448 | 896 |
|---|---|---|---|---|---|
| sign-SGD | — | — | — | — | — |
| qb-exact | 1 | 2 | 3 | 3 | 5 |
| qb-histogram | 1 | 2 | 3 | 3 | 6 |

Headline cells replicated over 5 seeds; spreads are tight (`qb-exact` at n=896, scale 0.15:
1.021 mean, [1.019–1.022]; at scale 1.0: 2.332 mean, [2.095–2.492]).

## Verdict

**The stated hypothesis is falsified.** On an unsaturated router the gap ratio *narrows*
monotonically with expert count — 149× at n=64 down to 39× at n=896. QB degrades with `n` too
(1.002 → 1.022), just far more slowly than the baseline. The advantage is real and large, but
it does not come *from* scale.

**The report's underlying concern survives in a different form.** Sign-SGD never reaches 5% of
uniform at any expert count, under any of five step sizes, within the budget; QB always does.
So the correct statement is not "the gap widens with n" but "**QB converges and sign-SGD does
not** — at every scale tested". That is a stronger claim than the one the report makes, and it
holds at n=64 as much as at n=896.

**Unexpected: tie degradation compounds with expert count, and inverts the ranking.** The unit
tests already showed QB is weakened by score ties. This sweep shows the damage is not
scale-free. At a fixed 1.9% tie fraction QB is essentially unharmed at n=64 (1.025) but fails
outright at n=896 (2.396 vs 2.622 for no balancing at all). Two consequences worth carrying:

- **`qb-alternating` fails identically** (2.396), so this is not a defect of the online update.
  The balanced assignment itself becomes infeasible — no threshold can split a tied group.
- **At n=896 saturated, crude sign-SGD beats exact QB** (2.123 vs 2.396), and the histogram
  estimator is *worse than doing nothing* (3.377 vs 2.622). A method that is exact under its
  assumptions degrades past its baseline once they break.

The mechanism is the target load. Each expert should receive `mk/n` tokens — 4,096 at n=64 but
only 292 at n=896. As `n` grows the required quantile moves further into the tail of the score
distribution, where the saturated mass sits, so the same tie fraction bites progressively
harder.

> **Superseded.** This section originally concluded that router temperature *"is not a free
> hyperparameter at K3's 896-expert scale — it is a precondition for balancing to work at all"*.
> [`trained-router`](trained-router.md) tested that by training routers by gradient descent and
> measuring the precondition directly: at K3's exact configuration (896 routed, 16 active) a
> trained router shows **0.0000 saturation and 0.0000 ties**, with no trend across a 112x range
> of expert counts. The tie regime is not reached, so there is nothing for temperature to fix.
>
> What survives is everything above the leap: the tie sensitivity *is* real as a property of the
> quantile estimator given tied inputs, the target-load mechanism explaining why it worsens with
> `n` is unchanged, and the numbers below stand. What does not survive is the inference from an
> imposed input regime to a claim about routers at scale — the first caveat below turned out to
> be the load-bearing one.

## Caveats

- Synthetic routers. Saturation is imposed via `router_scale`, not observed in a trained model;
  a real router may never reach 1.9% pinned scores. **This caveat was correct and was the one
  that mattered** — [`trained-router`](trained-router.md) measured it and found trained routers
  sit 3-5x in logit scale below tie onset. The interaction with `n` is real within the tied
  regime; reaching that regime is the part that does not hold.
- Load imbalance is a proxy for the thing that matters. It measures the systems cost directly,
  but says nothing about whether balanced experts are *better specialized*, which is the open
  question in [`moe-load-balancing.md`](../primitives/moe-load-balancing.md).
- Single popularity-skew setting (0.25) and one token count (16,384).

## Next move

- ~~Measure tie fraction in a **trained** router~~ — done: [`trained-router`](trained-router.md).
  The saturated regime is not reachable, so this is the curiosity branch.
- ~~Add a temperature arm~~ — moot. There is no saturation to correct.
- **Explain the residual imbalance instead.** `trained-router` found load imbalance under QB
  rising 1.145 -> 2.188 from 8 to 896 experts *without* ties, so the difficulty K3 cites is real
  and has a cause this ablation has not identified.
- The `steps_to_within_5pct` growth (1 → 5) is QB's own scale cost and deserves its own look:
  it is mild, but it is not flat, and the report implies it is.
