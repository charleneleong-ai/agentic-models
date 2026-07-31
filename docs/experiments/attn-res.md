# attn-res — how much of K3's 2.5x is the depth axis alone?

**Spec:** [`configs/ablations/attn-res.yaml`](../../configs/ablations/attn-res.yaml) ·
**Runner:** [`archlab/ablations/attn_res.py`](../../src/archlab/ablations/attn_res.py) ·
**Run:** 2026-07-31, A100, ~28 min, 8 runs (4 arms x 2 seeds)

## Hypothesis

K3 credits ~2.5x scaling efficiency to KDA + AttnRes + Stable LatentMoE *jointly* and publishes
no per-component split. If Attention Residuals carries meaningful weight, replacing the residual
stream with attention over depth should lower loss at matched width, depth and attention core.

**Falsifier:** if `residual` matches or beats the AttnRes arms, the depth axis is not doing
independent work at this scale.

## Results

Validation loss after 600 steps, mean of 2 seeds. Uniform baseline is `ln(64) = 4.1589`.

| arm | live sources | local loss ↓ | seed spread | recall loss ↓ | params |
|---|---:|---:|---:|---:|---:|
| **residual** | 1 | **2.7910** | 0.0005 | 3.8365 | 3,437,616 |
| attnres-block-2 | 3 | 2.7955 | 0.0022 | 3.8354 | 3,439,280 |
| attnres-block-4 | 5 | 2.8126 | 0.0056 | 3.8384 | 3,439,280 |
| attnres-full | 13 | 2.8941 | 0.0683 | 3.8611 | 3,439,280 |

Two orderings, both monotone in the number of sources attended over, and both replicated in
each seed independently:

- **Loss rises with sources.** `2.7910 < 2.7955 < 2.8126 < 2.8941`, monotone in seed 0 and
  seed 1 separately.
- **Seed instability rises with sources**, from 0.0005 to 0.0683 — a **136x** spread. The
  residual baseline is essentially deterministic across seeds; full AttnRes is not.

Arm gaps (0.004–0.10) sit well above seed noise (0.0005), so the ordering is not sampling
variation. Parameter counts differ by 1,664 (one pseudo-query per layer plus a norm, 0.05%), so
AttnRes is not losing for want of capacity — nor winning by having extra.

## Verdict

**Falsified at this scale.** The plain residual stream wins, and the more depth-attention an
arm does, the worse it gets. On this evidence the depth axis contributes nothing independently
at 12 layers — it costs.

**The mechanism I proposed was wrong, and testing it is the useful part.** My first reading was
that untrained pseudo-queries give a near-uniform softmax, so attending over more sources
averages harder and averaging beats nothing — which would explain both orderings at once. That
predicts the queries stay near-uniform. Measured directly, as entropy over sources normalized
by its uniform maximum:

```
layer            1     2     3     4     5     6     7     8     9    10    11    12
at init        0.83  0.83  0.80  0.81  0.84  0.84  0.86  0.83  0.85  0.84  0.86  0.83
after 600      0.34  0.44  0.59  0.68  0.74  0.79  0.66  0.69  0.78  0.74  0.77  0.36
```

Mean 0.835 → 0.630, with the **first and last layers sharply selective** (0.34, 0.36). The
queries clearly do learn to select. So the deficit is *not* a failure to differentiate, and the
tidy explanation is dead.

What remains, untested: the higher seed variance points at an optimization difficulty rather
than a representational one, and 12 layers may simply not have enough depth structure worth
retrieving — a residual stream over 12 layers is not obviously a bottleneck. The
selectivity profile is suggestive here: the layers that specialize are the ones at the
*ends*, where "attend to the embedding" and "attend to everything" are useful primitives, while
the middle stays diffuse.

## What this does not show

It does not refute K3. The gap between this setup and the claim is large enough to matter:

| | here | K3 |
|---|---|---|
| layers | 12 | 93 |
| parameters | 3.4M | 2.8T |
| blocks | 2–4 | 8 |
| training | 600 steps | full pre-training |

AttnRes is a mechanism for *reaching across depth*. At 12 layers there is little depth to reach
across, and the report's own guidance (N ≈ 8 blocks) is not even expressible here. A negative
result at 1/8th the depth and 1/800,000th the parameters bounds nothing about 2.8T. What it does
establish is that the benefit is **not scale-free** — it does not appear the moment you wire the
mechanism in, which is worth knowing before assuming any published component transfers.

## The recall half is uninformative

Every arm lands at 3.835–3.861, only ~7.5% below the uniform baseline, with between-arm
differences comparable to seed noise. Local loss fell from 4.16 to 2.79 over the same run, so
the models trained — they just did not learn the long-range task. **The recall column should be
read as "no signal", not as a tie.** At 600 steps with 4 planted pairs and gaps of 74–251
tokens, the task is out of reach; that is a fact about this setup, not about the architectures.

## Caveats

- 600 steps is short. The learning curves show the ordering established by step ~60 and stable
  thereafter, so this is not obviously a convergence-rate artifact — but "stable for 600 steps"
  is not "stable at convergence".
- Two seeds. Enough given the gap-to-noise ratio, not enough for a tight interval on the
  variance claim.
- One corpus, one width, one depth.

## Next move

- **Sweep depth.** The single most informative follow-up: run 6 / 12 / 24 / 48 layers. If the
  AttnRes deficit shrinks with depth, the nano-scale result is a depth artifact and the paper's
  claim survives cleanly. If it is flat, something else is going on.
- **Fix the recall task** before drawing any long-range conclusion — fewer pairs, shorter gaps,
  or a much longer budget. Until then `kda-state-capacity` would inherit the same dead metric.
- Probe whether the optimization-difficulty reading holds: does a warmup on the pseudo-queries,
  or a lower LR on them specifically, close the gap?
