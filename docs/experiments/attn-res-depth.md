# attn-res-depth — is the AttnRes deficit a depth artifact?

**Spec:** [`configs/ablations/attn-res-depth.yaml`](../../configs/ablations/attn-res-depth.yaml) ·
**Runner:** [`archlab/ablations/depth_sweep.py`](../../src/archlab/ablations/depth_sweep.py) ·
**Run:** 2026-07-31, A100, ~2h, 24 runs (3 arms x 4 depths x 2 seeds)

## Hypothesis

[`attn-res`](attn-res.md) found the residual stream beating every AttnRes arm at 12 layers.
The obvious confound is depth: AttnRes is a mechanism for reaching *across* depth, and 12
layers may not have enough to reach across, against K3's 93. So sweep depth and watch the gap.

**Falsifier as written:** if the gap is flat or widening with depth, depth is not the confound.

## The falsifier was invalid, and that is the finding

The inference assumes added depth is *useful*. It is not, on this corpus:

| depth | residual local loss | seeds |
|---:|---:|---|
| 6 | 2.7925 | 2.7933, 2.7917 |
| 12 | 2.7904 | 2.7917, 2.7891 |
| 24 | 2.7930 | 2.7946, 2.7914 |
| 48 | 2.7986 | 2.7990, 2.7982 |

**An 8x increase in depth changes loss by +0.0061 — in the wrong direction.** Forty-eight
layers are no better than six. The task is depth-saturated at or below the shallowest setting
tested.

With no depth-structured signal to retrieve, a depth-mixing mechanism has nothing to gain and
can only pay. So a widening gap cannot distinguish "AttnRes is bad at depth" from "this corpus
cannot reward *any* depth mechanism". **The primary question is unanswerable here**, and the
cause is my corpus design, not the architecture.

## What the run does establish

Gap to the residual baseline at matched depth and seed, mean [min, max]:

| depth | n_blocks | attnres-block-6 | attnres-full |
|---:|---:|---:|---:|
| 6 | 1 | −0.0013 [−0.003, +0.000] | +0.0135 [+0.005, +0.022] |
| 12 | 2 | +0.0024 [−0.000, +0.005] | +0.0763 [+0.070, +0.083] |
| 24 | 4 | +0.0364 [+0.032, +0.041] | +0.1192 [+0.067, +0.171] |
| 48 | **8** | +0.0741 [+0.052, +0.096] | +0.2326 [+0.210, +0.255] |

**1. AttnRes's cost scales with depth.** Both arms are monotone; past L=12, doubling depth
roughly doubles the gap. At K3's exact configuration — 48 layers, 8 blocks — the blocked form
costs +0.074.

**2. The cost is driven by depth, not by how many sources are attended over.** This is the
cleanest result in the sweep, and it falsifies the reading I was drifting toward mid-run:

| arm | depth | sources | gap |
|---|---:|---:|---:|
| attnres-full | 6 | 7 | +0.0135 |
| attnres-block-6 | 48 | 9 | **+0.0741** |

Comparable source counts, 5.5x difference in cost. Source count is not the driver; depth is.

**3. Blocking reduces the cost ~3x at every depth but does not stop it growing.** Blocked
AttnRes is nearly free to L=12 (|gap| < 0.003) and clearly not free by L=48.

**4. Full AttnRes is the less stable arm**, with seed spreads to 0.104 (L=24) against blocked
AttnRes's 0.003–0.045. One cell also failed to reproduce across runs: L=12 seed 0 gave +0.0827
here versus +0.1375 in [`attn-res`](attn-res.md), while L=12 seed 1 reproduced to 0.0012. Treat
individual full-AttnRes cells as ±0.05; the trend across four depths is well above that.

## Verdict

**Inconclusive on the question asked. Informative on cost.** The depth-artifact hypothesis is
neither confirmed nor refuted, because the experiment cannot see benefit — only cost — on a
corpus where depth does not pay.

The cost structure it does reveal is worth carrying, and it reframes something in the report.
K3 justifies Block AttnRes on memory grounds: `O(Ld) -> O(Nd)` (§2.2). These numbers suggest
blocking may also be doing *optimization* work — the full form's cost grows about 3x faster
with depth, and at 93 layers that extrapolates badly. The report's "N ≈ 8 recovers most of the
benefit" may understate why blocking is there.

That remains a hypothesis. Everything measured here is cost on a task with no depth benefit, so
the trade blocking makes — how much benefit it gives up for that stability — is exactly what
this corpus cannot show.

## The corpus is the binding constraint

Two independent signals now say so: local loss is flat in depth, and
[recall loss never moved](attn-res.md#the-recall-half-is-uninformative) in either ablation. The
Markov filler is learnable in a few layers; the recall task is not learned at all. Nothing in
between rewards composition.

This blocks more than this ablation.
[`kda-state-capacity`](../../configs/ablations/kda-state-capacity.yaml) depends on the same
recall metric and would inherit a dead measurement silently.

**Fix before running anything else that touches depth or long-range structure:** a corpus with
genuinely compositional structure — nested bracket grammar, or chained function application
where the answer requires evaluating `f(g(h(x)))` and each nesting level plausibly needs a
layer. The test is simple and should be run first: **does the residual baseline improve from 6
to 48 layers?** If not, the corpus is not ready and no depth ablation on it means anything.

## Caveats

- One corpus, one width (d_model=128), one step budget (600).
- Two seeds. Adequate for blocked AttnRes; thin for full, whose spread reaches 0.104.
- Depth was swept; width was not. A depth-saturated task might still be width-limited, so
  "saturated" here means specifically "adding layers does not help".

## Next move

1. **Build a depth-rewarding corpus and validate it with the residual baseline alone** — cheap,
   and a hard prerequisite for re-running this.
2. Re-run this sweep on it. The same three arms and four depths become meaningful the moment
   the baseline improves with depth.
3. Only then revisit `kda-state-capacity`, which needs a working long-range metric.
