# trained-router — is the tie regime reachable by training?

**Runner:** [`archlab/ablations/trained_router.py`](../../src/archlab/ablations/trained_router.py) ·
**Tests:** [`tests/test_trained_router.py`](../../tests/test_trained_router.py) ·
**Run:** 2026-08-01, A100, 600 steps per arm

The next move [`qb-scale`](qb-scale.md) set for itself: *"Measure tie fraction in a trained
router to see whether the saturated regime is reachable in practice. If it is not, this is a
curiosity; if it is, it is a design constraint."*

It is not reachable. This writes down the curiosity verdict against the repo's own most-promoted
finding.

## Hypothesis

qb-scale concluded that **router temperature is a precondition for balancing at K3's 896-expert
scale**. That conclusion rests on ties being present. Ties were *imposed* there via
`router_scale`, never observed — so the claim assumes training visits a regime the experiment
never checked.

**Falsifier:** if gradient-trained routers show ~0 saturation and ~0 ties at every expert count,
the tie finding describes an input regime training does not reach, and the temperature claim is
unsupported.

## Setup

A `LatentMoE` trained standalone against a fixed random teacher — an MLP with more structure than
any single expert can fit, so the router is forced to differentiate. Expert count varies over
8/64/256/896 holding K3's active fraction (16/896), so the last row is K3's exact routing config.
The MoE is trained directly rather than inside a language model because the router is the object
under study; an LM wrapper adds confounds without adding signal.

## Results

| experts | active | \|logit\|max | saturated | ties | distinct | imbal QB | mse |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 2.22 | 0.0000 | 0.0000 | 1.0000 | 1.145 | 0.0208 |
| 64 | 2 | 3.07 | 0.0000 | 0.0000 | 1.0000 | 1.375 | 0.0216 |
| 256 | 5 | 2.80 | 0.0000 | 0.0000 | 1.0000 | 1.700 | 0.0222 |
| 896 | 16 | 3.12 | 0.0000 | 0.0000 | 0.9999 | 2.188 | 0.0215 |

**The detector is verified, not assumed.** A broken metric also reports zero, so a positive
control establishes what non-zero looks like:

| imposed logit scale | saturated | ties | distinct |
|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0000 | 1.0000 |
| 3 | 0.0011 | 0.0000 | 1.0000 |
| 10 | 0.3241 | 0.9941 | 0.9355 |
| 30 | 0.7427 | 1.0000 | 0.6961 |

Trained routers land at |logit|max 2.2–3.1 — the scale-1 to scale-3 band. Ties begin somewhere
above scale 3 and are total by scale 10, so trained routers sit a **3–5× logit-scale gap** below
onset. The zeros mean absence.

## Verdict

**The temperature claim is unsupported.** At K3's exact configuration — 896 routed, 16 active —
a trained router shows zero saturation and zero ties. There is nothing for temperature to fix,
and saturation shows no trend in expert count (2.22 → 3.12 across a 112× range), so no
extrapolation rescues it either.

**The phenomenon qb-scale was chasing is nonetheless real.** Load imbalance under QB rises
monotonically with expert count, 1.145 → 2.188, roughly doubling from 8 to 896 experts. K3's
concern that balancing gets harder as the pool grows reproduces in a trained router. So:

> qb-scale found a real problem and attributed it to the wrong mechanism.

That distinction is the result. The tie-degradation measurement remains valid as a **property of
the quantile estimator given tied inputs** — the algebra is unchanged, and the unit tests that
pin it still pass. What does not survive is the leap from that property to a claim about routers
at scale.

Worth noting the trained router is *harder* to balance than the synthetic one qb-scale used:
2.188 versus 1.022 for `qb-exact` at n=896 unsaturated. The synthetic setup, with its fixed 0.25
popularity skew, was optimistic about QB rather than pessimistic. The residual imbalance has
somewhere else to come from, and finding it is the open question.

## Caveats

Both directions, because the refutation deserves the same scepticism as the claim.

- **This is not an LM at scale.** 600 steps, a synthetic Gaussian-input teacher, one MoE layer.
  Production routers see structured non-Gaussian inputs and train orders of magnitude longer.
  This refutes "training reaches the tied regime *in this setup*"; it does not prove production
  routers never saturate. The right reading is that saturation must be *demonstrated* before it
  is designed around, not assumed from a synthetic sweep.
- **The `imbal T` column cannot isolate temperature.** It zeroes the bias and rescales the logits
  together, so it conflates two changes. It is reported in the runner as a load-imbalance
  datapoint and must not be read as a temperature result. A clean temperature arm would re-solve
  the bias on the rescaled scores.
- **Weight decay pulls against saturation.** The router is trained with `weight_decay=0.1`, which
  directly penalises the large router norms that saturation requires. This is standard practice
  and K3 trains with decay too, but a decay-free arm would separate "training does not produce
  saturation" from "this regulariser prevents it".
- **One seed per cell.** The effect is a categorical zero rather than a small difference, so seed
  variance is unlikely to change the sign — but it is not measured.

## Next move

- **Rewrite the qb-scale verdict**, done in [`qb-scale.md`](qb-scale.md), and downgrade the claim
  wherever it appears in [`README.md`](../../README.md) and
  [`landscape-2026.md`](../landscape-2026.md).
- **Find where the residual imbalance comes from**, since it is real and rises with `n` while
  ties do not. Candidates: expert popularity genuinely learned by the router, the bias chasing a
  moving target during training, or a target-load effect independent of ties.
- **A decay-free arm**, to separate the regulariser from the training dynamics.
