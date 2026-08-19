# muP — does the optimal learning rate hold still across width?

**Module:** [`archlab/mup.py`](../../src/archlab/mup.py) ·
**Check:** [`archlab/mup_check.py`](../../src/archlab/mup_check.py) ·
**Contract test:** [`tests/test_contracts.py`](../../tests/test_contracts.py) ·
**Run:** 2026-08-02, A100, 300 steps per cell

> **This page previously concluded that muP does not work and that the ladder stays blocked.**
> Both were wrong: the learning-rate groups were never reaching the optimizer, so the two arms
> being compared were the same run. The mistake is documented in
> [Two wrong turns](#two-wrong-turns), because how it survived verification is more useful than
> the result.

## Why muP was reached for

The plan was a width ladder: run the AttnRes arms at several widths and check whether the
*ranking* survives, since [validity-threat work](https://arxiv.org/pdf/2606.05029) says proxy
rankings can fail to transfer even between 125M and 1B.

That plan was confounded. Under Standard Parametrization the same learning rate is not right at
every width, and [Tensor Programs V](https://arxiv.org/abs/2203.03466) documents the
consequence: performance can improve with width and then suddenly worsen. A rank flip on an SP
ladder cannot distinguish "the mechanism does not transfer" from "the learning rate stopped
being right".

muP is the published fix — the parametrization under which optimal hyperparameters hold still
across width. Its operational definition doubles as the diagnostic: *a hyperparameter is
muTransferable if its optimal value is the same across model sizes.*

## Result

Optimal learning rate by width. 4-layer models, identical data and seed, 300 steps.

| SP | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | argmin |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 128 | 2.8871 | 2.8419 | 2.8176 | **2.7761** | 2.7771 | 2.8599 | 1e-2 |
| 256 | 2.8513 | 2.8295 | 2.8110 | **2.7745** | 2.8124 | 2.9011 | 1e-2 |
| 512 | 2.8336 | 2.7898 | **2.7674** | 2.8398 | 2.8435 | 2.8958 | **3e-3** |

| muP | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 | argmin |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 128 | 2.8871 | 2.8419 | 2.8176 | **2.7761** | 2.7771 | 2.8599 | 1e-2 |
| 256 | 2.8743 | 2.8361 | 2.8207 | **2.7650** | 2.7780 | 2.8561 | 1e-2 |
| 512 | 2.8679 | 2.8334 | 2.8226 | **2.7736** | 2.7747 | 2.8353 | 1e-2 |

**SP's optimum drifts; muP's holds.** SP moves from 1e-2 to 3e-3 between width 256 and 512, and
the move is decisive — 2.7674 against 2.8398, a gap of 0.072 where adjacent cells in that column
usually differ by ~0.01. muP stays at 1e-2 at every width.

The grid runs two cells past 1e-2 deliberately. The first version stopped at 1e-2, found muP's
optimum there at all three widths, and that was worth nothing: 1e-2 was the grid's own edge, and
an optimum pinned to a boundary agrees across widths for free. Extending to 3e-2 and 1e-1 makes
every muP minimum interior, so the agreement is a property of the parametrization rather than of
where the sweep was truncated.

**Not claimed:** the 1e-2 vs 3e-2 margin is ~0.001, so *which* of those two adjacent cells wins
is not robust to a seed change. The claim is the separation between parametrizations — SP's
optimum moves by a factor of ~3 over this width range and muP's does not.

## Two wrong turns

Both were caught by process rather than insight, which is why they are recorded.

### The output multiplier was a double-correction

Tensor Programs V prescribes a `1/fan_in` multiplier on the output. Applied here it made logits
shrink as `1/m` — 0.4651 → 0.0578 across an 8× width span. The paper's own coordinate check
caught it before any transfer study was built on it.

The cause is architectural. This model has an RMSNorm immediately before the head, so the head's
input is unit-scale at every width and logits are already width-invariant under SP (0.99× drift
over the same span). The forward-pass half of muP is absorbed by the normalization here; the
learning-rate half is not, and is what [`mup.py`](../../src/archlab/mup.py) supplies.

### The learning-rate groups never reached the optimizer

`train_arm` imported `mup_param_groups` and never called it. The optimizer was still built from
`model.parameters()`, so `mup=True` and `mup=False` differed only by an init rescale — and the
run reported, correctly for what it actually measured, that muP drifted identically to SP. That
was committed as a negative result and written up on this page.

The verification that failed is the instructive part. `mup_param_groups` *was* checked: called
directly, confirmed to produce hidden LR 0.0025 at width 512 with all 79 parameters covered.
That check was correct and irrelevant. **A correct unit is not evidence that anything calls it**,
and testing the builder is exactly what cannot detect a missing call to the builder.

Three fixes, none of them the one-line repair:

- The ruff gate went from complexity-only to `["C901", "PLR0915", "F", "ARG"]`. F401 would have
  flagged the unused import on the commit that introduced it. Adopting it cost 3 fixes
  repo-wide — it was available the whole time.
- The contract test spies on `AdamW` during a real `train_arm` call and asserts the groups
  arrive **at the optimizer**, rather than asserting the builder returns the right thing.
- `/simplify` belongs before the first commit, not after the third. Three of its four review
  agents flagged this independently; in the right order it would have caught the bug before the
  wrong claim was written down at all.

## Caveats

- 4-layer models, 300 steps, one seed per cell, three widths spanning 4×. Transfer is
  demonstrated over that range, not extrapolated past it.
- Only the learning rate is swept. Batch size, depth and training length are muTransferable in
  the literature but untested here.
- Depth transfer (Tensor Programs VI) is not implemented — `n_layers` is fixed at 4 throughout,
  so a *depth* ladder remains as confounded as the width ladder was.
- The two parametrizations are identical at width 128 by construction (`m = 1`), so that row is
  a consistency check, not evidence.

## Next move

The width ladder is unblocked, which was the original goal. It can vary width at a fixed 1e-2
and know that a ranking change is the mechanism rather than the optimizer — the confound that
made [`attn-res`](attn-res.md) the finding most exposed to the transfer threat.
