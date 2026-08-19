# Experiments

Each ablation exists to answer a question the paper leaves open. One YAML in
[`configs/ablations/`](../../configs/ablations/), one writeup here, same stem.

> **Status:** [`qb-scale`](qb-scale.md), [`attn-res`](attn-res.md),
> [`attn-res-depth`](attn-res-depth.md), [`activation-bound`](activation-bound.md), and
> [`kda-state-capacity`](kda-state-capacity.md) have run. The first two falsified their own
> hypothesis; the third found its *falsifier* invalid — the corpus is depth-saturated, so no
> depth ablation on it can mean anything. The fourth found SiTU-GLU's precision rationale
> inert at nano scale. The fifth found multi-head latent state tracking wins 2-4x at 64KB+.
>
> **Blocked:** the remaining depth/long-range work needs a corpus that is hard, learnable and
> depth-sensitive at once. Two designs have now failed that bar for the same reason — see
> [corpus-gate.md](corpus-gate.md). Validate any candidate with `archlab gate envelope` and
> `archlab gate depth` *before* spending sweep time.

## The five

| Ablation | Question | Why it is open | Needs GPU |
|---|---|---|---|
| [`attn-res`](../../configs/ablations/attn-res.yaml) **· [run →](attn-res.md)** | How much of the 2.5x is the depth axis alone? | The report credits KDA + AttnRes + LatentMoE *jointly* and publishes no per-component ablation | yes |
| [`hybrid-ratio`](../../configs/ablations/hybrid-ratio.yaml) | Does KDA delay retrieval head formation? | arXiv 2606.15378 found SWA causes laziness; KDA is different but has similar window property | yes |
| [`kda-state-capacity`](../../configs/ablations/kda-state-capacity.yaml) **· [run →](kda-state-capacity.md)** | When does a fixed state stop holding the sequence? | Multi-head tracking wins 2-4x at 64KB; state capacity matters | yes |
| [`activation-bound`](../../configs/ablations/activation-bound.yaml) **· [run →](activation-bound.md)** | Is capping activations free at BF16, load-bearing at FP8? | SiTU-GLU is justified on precision grounds with no quality comparison shown | yes |
| [`qb-scale`](../../configs/ablations/qb-scale.yaml) **· [run →](qb-scale.md)** | Does QB's edge over sign-updates widen with expert count? | The stated motivation is about the trend toward 896 experts, not a single point | **no** |

`qb-scale` ran first because routing depends only on scores and bias — no training, 77 s on
CPU. It falsified its own hypothesis (the gap *narrows* with expert count) and turned up
something better: tie degradation compounds with `n`, to the point where at 896 experts a
saturated router defeats every balancer tested. [Writeup](qb-scale.md).

`attn-res` also came out negative: at 12 layers the plain residual stream beats every AttnRes
arm, and loss rises monotonically with the number of sources attended over. The tidy
explanation — that untrained pseudo-queries average rather than select — was measured and
**refuted**; they do become selective. [Writeup](attn-res.md).

`attn-res-depth` swept 6/12/24/48 layers to test whether that was a depth artifact, and found
the question unanswerable: **an 8x depth increase changes the baseline loss by +0.006**, so the
corpus rewards no depth at all and a depth mechanism can only pay. It did establish that
AttnRes's cost scales with depth rather than with sources attended (blocked at 48 layers costs
5.5x more than full at 6, on comparable source counts), and that blocking cuts the cost ~3x
without stopping its growth — which suggests K3's blocking may be doing optimization work, not
just the memory work §2.2 claims for it. ⛔ marks ablations blocked on the corpus fix.
[Writeup](attn-res-depth.md).

`activation-bound` tested SiTU-GLU's precision rationale and found it **inert at this scale**:
peak activation 87.5 against `e4m3`'s 448 ceiling, zero overflows, and FP8-range quantization
shifting loss less than seed noise despite injecting 2.3% error per activation. The cap neither
costs nor buys measurable quality. The one robust finding is that a cap which *engages* raises
seed variance ~10x — the opposite of the stabilising role the report describes.
[Writeup](activation-bound.md).

## Scale-transfer validation

Proxy results are externally unverified by construction. The plan was a width ladder; it turned
out to need muP first, because SP's optimal learning rate demonstrably drifts with width — 1e-2
at widths 128 and 256, 3e-3 at 512 — so a rank flip on an SP ladder cannot be told apart from
the learning rate ceasing to be right.

**The ladder has since run** — the attn-res ranking holds across a 4x width increase, with the
caveat that it rests on 6/6 consistency rather than on any single margin
([`attn-res-scale.md`](attn-res-scale.md)).

**muP works and unblocked it** ([`mup-attempt.md`](mup-attempt.md)): the optimum
holds at 1e-2 across all three widths, with an interior minimum at each. It reported as *not*
working for three commits because `train_arm` imported `mup_param_groups` and never called it,
so both arms shared one flat-LR optimizer. That page is worth reading for the failure mode
rather than the result — the builder had been verified in isolation and was correct, which is
exactly why nothing caught that it was unused.

## Corpus gate

Depth and long-range ablations are only meaningful on a corpus where the capability under test
is actually learned. Two have failed that bar, for one shared reason: **requiring depth is not
the same as inducing a model to use depth** ([corpus-gate.md](corpus-gate.md)).

```bash
archlab gate envelope          # is there a hard-but-learnable regime at all?
archlab gate depth --chain-len 4   # does depth move the boundary?
```

Minutes, not the two GPU-hours `attn-res-depth` spent learning the same thing the slow way.

## Config contract

Every ablation YAML carries the same top-level keys:

| Key | Meaning |
|---|---|
| `name` | Matches the filename stem and the writeup stem |
| `question` | One line. If it cannot be stated in one line, the ablation is not scoped yet |
| `primitive` | The `archlab` module under test — the thing the arms vary |
| `arms` | The comparison. Each needs an `id`; **one must be the honest baseline** |
| `sweep` | Optional. Axes crossed with `arms` |
| `model` / `train` | Held fixed across arms — anything varying belongs in `arms` |
| `metrics.primary` | The single number the verdict turns on |
| `metrics.also` | Secondary, including the *cost* the mechanism is supposed to pay |
| `report` | Path to the writeup |
| `requires_training` | Default true; false marks CPU-only ablations |

Two rules the schema is shaped to enforce. **Every ablation has a baseline arm** — a sweep
over variants of one mechanism, with nothing to beat, measures nothing. **Every ablation
records the cost, not just the win** — `attn-res` measures peak memory alongside loss because
AttnRes buys quality *with* memory, and a result quoting only loss would be dishonest.

## Writeup structure

Mirrors the sweep-writeup convention: **Hypothesis** (what, and what would falsify it) →
**Config** (yaml inlined) → **Results** (table, primary metric first) → **Verdict** (did the
hypothesis survive) → **Next move**.

Write the hypothesis *before* running. A hypothesis composed after seeing results is a
description.

## Runner

`archlab ablate <name> [--device cuda]` runs any ablation with a registered runner
([`archlab/cli.py`](../../src/archlab/cli.py)). Four exist:

| runner | training? | notes |
|---|---|---|
| `qb-scale` | no | routing depends only on scores and bias — CPU, 77 s |
| `attn-res` | yes | shared [training loop](../../src/archlab/ablations/train.py) + [model assembler](../../src/archlab/model.py) |
| `attn-res-depth` | yes | crosses the arms with `sweep.n_layers` |
| `activation-bound` | yes | adds activation probes and simulated FP8 |
| `kda-state-capacity` | yes | crosses `d_head` × `nesting` against KDA arms |

One config still lacks a runner -- [`hybrid-ratio`](../../configs/ablations/hybrid-ratio.yaml) -- and the assembler
[`kda-state-capacity`](../../configs/ablations/kda-state-capacity.yaml) — and the assembler
already covers their arms, so each needs only its own runner. Both are blocked on the corpus
question first: `kda-state-capacity` shares the dead recall metric outright, and
`hybrid-ratio`'s re-scoped question (KDA laziness) needs a corpus where retrieval heads can be measured during training.

Layout:

```
experiments/<ablation>/results.jsonl    append-only, one row per arm x seed (gitignored)
experiments/<ablation>/progress.html    live chart (gitignored)
docs/experiments/<ablation>.md          the writeup (committed)
```

The primitives are the part that had to be right first, and they are tested. The rest is
plumbing around them.
