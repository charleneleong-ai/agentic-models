# attn-res-scale — does the ranking survive a 4× width increase?

**Config:** [`attn-res-scale.yaml`](../../configs/ablations/attn-res-scale.yaml) ·
**Runner:** [`archlab/ablations/scale_ladder.py`](../../src/archlab/ablations/scale_ladder.py) ·
**Run:** 2026-08-02, A100, 18 cells + 3-point LR search

```bash
archlab ablate attn-res-scale --device cuda
```

## Hypothesis

[`attn-res`](attn-res.md) found a plain residual stream beating attention-over-depth at 12
layers and 128 width. That is a claim about *learned model quality*, the class
[validity-threat work](https://arxiv.org/pdf/2606.05029) says may not transfer — proxy rankings
can fail even between 125M and 1B — and [`README`](../../README.md) flagged it as the finding
most exposed to that threat.

**Falsifier:** if the arm ordering changes as width grows, the original result is scale-specific.

The ladder measures the *ranking*, not the loss. Losses must fall as models get wider; comparing
them across rungs would measure capacity, not the mechanism.

## Why this was blocked until now

Under Standard Parametrization the optimal learning rate drifts with width — measured at 1e-2
for widths 128 and 256, 3e-3 at 512 ([`mup-attempt`](mup-attempt.md)). A fixed-LR ladder
therefore trains its wider rungs wrong, and a rank flip cannot be distinguished from the
optimizer becoming mistuned. muP holds the optimum in place and is verified here to do so; the
ladder inherits that guarantee, which is the only reason it can hold one learning rate across
rungs.

The base LR was still searched once at the base width, since muP transfers across *width* while
this ladder holds depth at 12 and the muP check ran at depth 4. It found 1e-2 with an interior
minimum (2.7302 / **2.7293** / 2.7322 over 3e-3 / 1e-2 / 3e-2), independently reproducing the
muP check's optimum at a different depth.

## Results

Seed-averaged, ± is the spread between the two seeds.

| width | ordering |
|---:|---|
| 128 | residual (2.7269 ±0.0048) < attnres-block-4 (2.7356 ±0.0081) < attnres-full (2.7420 ±0.0010) |
| 256 | residual (2.7268 ±0.0037) < attnres-full (2.7379 ±0.0006) < attnres-block-4 (2.7450 ±0.0179) |
| 512 | residual (2.7219 ±0.0005) < attnres-block-4 (2.7444 ±0.0219) < attnres-full (2.7572 ±0.0283) |

**Residual wins at every width — 6 of 6 pairwise comparisons.**

The two AttnRes arms swap places between widths, and the first version of this runner reported
that as a rank flip and declared the finding scale-specific. It was wrong: their gap is 0.0064
to 0.0127 against seed spreads of 0.0081 to 0.0283, so **that ordering is never resolvable at
any width**. A ladder that scores unresolvable swaps manufactures false negatives at exactly the
rate its arms are close together. The runner now compares every gap against the seed spread
beneath it and reports only what the noise can separate.

### How strong is the surviving claim?

| width | vs | gap | seed noise | margin |
|---:|---|---:|---:|---:|
| 128 | attnres-full | 0.0151 | 0.0048 | 3.15× |
| 128 | attnres-block-4 | 0.0088 | 0.0081 | 1.08× |
| 256 | attnres-full | 0.0111 | 0.0037 | 3.01× |
| 256 | attnres-block-4 | 0.0182 | 0.0179 | 1.02× |
| 512 | attnres-full | 0.0352 | 0.0283 | 1.24× |
| 512 | attnres-block-4 | 0.0225 | 0.0219 | 1.03× |

Four of six clear the noise by under 1.25×, and with two seeds the "noise" is a two-sample range —
a poor estimate of anything. **No individual margin carries this result.**

What carries it is consistency: residual wins 6/6 comparisons spanning a 4× width range. Treating
each as an independent coin flip puts that at *p* ≈ 0.016. That is the honest basis for the
verdict — an aggregate over consistent weak evidence, not a strong effect at any single rung.

## Verdict

**The attn-res finding survives its main threat over 128 → 512.** Residual beats
attention-over-depth at every width tested, and the *relative* ordering of the two AttnRes
variants was never resolvable to begin with, so it was never part of the claim.

This does not extend past 512, and it should not be read as evidence about K3's 93 layers.
[Published work](https://arxiv.org/abs/2606.15378) reports efficient-attention design changing
how *fast* long-context capability emerges rather than the eventual ceiling, which is exactly
the kind of effect a 600-step budget cannot see.

## Caveats

- **Two seeds.** The dominant limitation. Every margin is measured against a two-sample range,
  and four of six margins are under 1.25×. Five seeds would make the noise estimate meaningful
  and cost ~2.5 GPU-hours.
- **The noise grows with width** — residual's own spread falls (0.0048 → 0.0005) while the
  AttnRes arms' rises (0.0010 → 0.0283). The wider AttnRes models are less stable across seeds,
  which is a finding in itself and is not investigated here.
- **Width only.** muP is verified for width, not depth (Tensor Programs VI is not implemented),
  so `n_layers` is pinned at 12. A depth ladder is as confounded as the width ladder was.
- **One task, 600 steps.** The recall corpus at a short budget, the same operating point as the
  original ablation. Transfer is shown for that setting.
- The LR search grid is coarse and its three points span only 0.003 in loss, so the base LR is
  weakly determined. muP is what makes that tolerable — the same value is correct at every width,
  so being slightly off is at least *consistently* slightly off.

## Next move

- **Five seeds**, to convert six thin margins into one defensible one. Cheapest real improvement.
- **Ask why wider AttnRes is seed-unstable.** Spread rises 20× from width 128 to 512 for
  `attnres-full` while the residual baseline becomes *more* stable. That is unexplained.
- A depth ladder needs Tensor Programs VI depth transfer first.
