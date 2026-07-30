# Experiments

Each ablation exists to answer a question the paper leaves open. One YAML in
[`configs/ablations/`](../../configs/ablations/), one writeup here, same stem.

> **Status:** [`qb-scale`](qb-scale.md) has run — the only one that needs no training. The
> other four are specifications, written first so the question is fixed before the plumbing
> tempts the answer; their runner does not exist yet. See [Runner](#runner).

## The five

| Ablation | Question | Why it is open | Needs GPU |
|---|---|---|---|
| [`attn-res`](../../configs/ablations/attn-res.yaml) | How much of the 2.5x is the depth axis alone? | The report credits KDA + AttnRes + LatentMoE *jointly* and publishes no per-component ablation | yes |
| [`hybrid-ratio`](../../configs/ablations/hybrid-ratio.yaml) | Where does the KDA:MLA tradeoff actually sit? | 3:1 is asserted, never swept — anywhere | yes |
| [`kda-state-capacity`](../../configs/ablations/kda-state-capacity.yaml) | When does a fixed state stop holding the sequence? | 1M benchmarks tolerate approximate recall, so they cannot expose the ceiling | yes |
| [`activation-bound`](../../configs/ablations/activation-bound.yaml) | Is capping activations free at BF16, load-bearing at FP8? | SiTU-GLU is justified on precision grounds with no quality comparison shown | yes |
| [`qb-scale`](../../configs/ablations/qb-scale.yaml) **· [run →](qb-scale.md)** | Does QB's edge over sign-updates widen with expert count? | The stated motivation is about the trend toward 896 experts, not a single point | **no** |

`qb-scale` ran first because routing depends only on scores and bias — no training, 77 s on
CPU. It falsified its own hypothesis (the gap *narrows* with expert count) and turned up
something better: tie degradation compounds with `n`, to the point where at 896 experts a
saturated router defeats every balancer tested. [Writeup](qb-scale.md).

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

`archlab ablate <name>` runs any ablation with a registered runner
([`archlab/cli.py`](../../src/archlab/cli.py)). Only `qb-scale` has one
([`qb_scale.py`](../../src/archlab/ablations/qb_scale.py)) — it needs just a config loader and
a scoring loop.

The other four need a model assembler mapping `arms` onto `archlab` modules, plus a training
loop. Layout either way:

```
experiments/<ablation>/results.jsonl    append-only, one row per arm x seed (gitignored)
experiments/<ablation>/progress.html    live chart (gitignored)
docs/experiments/<ablation>.md          the writeup (committed)
```

The primitives are the part that had to be right first, and they are tested. The rest is
plumbing around them.
