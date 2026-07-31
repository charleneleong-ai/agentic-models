# Experiments

Each ablation exists to answer a question the paper leaves open. One YAML in
[`configs/ablations/`](../../configs/ablations/), one writeup here, same stem.

> **Status:** [`qb-scale`](qb-scale.md), [`attn-res`](attn-res.md) and
> [`attn-res-depth`](attn-res-depth.md) have run. The first two falsified their own
> hypothesis; the third found its *falsifier* invalid — the corpus is depth-saturated, so no
> depth ablation on it can mean anything.
>
> **Blocked:** the remaining depth/long-range work needs a corpus that is hard, learnable and
> depth-sensitive at once. Two designs have now failed that bar for the same reason — see
> [corpus-gate.md](corpus-gate.md). Validate any candidate with `archlab gate envelope` and
> `archlab gate depth` *before* spending sweep time.

## The five

| Ablation | Question | Why it is open | Needs GPU |
|---|---|---|---|
| [`attn-res`](../../configs/ablations/attn-res.yaml) **· [run →](attn-res.md)** | How much of the 2.5x is the depth axis alone? | The report credits KDA + AttnRes + LatentMoE *jointly* and publishes no per-component ablation | yes |
| [`hybrid-ratio`](../../configs/ablations/hybrid-ratio.yaml) | Where does the KDA:MLA tradeoff actually sit? | 3:1 is asserted, never swept — anywhere | yes |
| [`kda-state-capacity`](../../configs/ablations/kda-state-capacity.yaml) ⛔ | When does a fixed state stop holding the sequence? | 1M benchmarks tolerate approximate recall, so they cannot expose the ceiling | yes |
| [`activation-bound`](../../configs/ablations/activation-bound.yaml) | Is capping activations free at BF16, load-bearing at FP8? | SiTU-GLU is justified on precision grounds with no quality comparison shown | yes |
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
([`archlab/cli.py`](../../src/archlab/cli.py)). Two exist: `qb-scale` (no training) and
`attn-res`, which uses the shared [training loop](../../src/archlab/ablations/train.py) and
[model assembler](../../src/archlab/model.py).

The remaining three need only their own runner — the assembler already covers their arms.
Layout:

```
experiments/<ablation>/results.jsonl    append-only, one row per arm x seed (gitignored)
experiments/<ablation>/progress.html    live chart (gitignored)
docs/experiments/<ablation>.md          the writeup (committed)
```

The primitives are the part that had to be right first, and they are tested. The rest is
plumbing around them.
