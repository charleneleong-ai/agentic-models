# corpus-gate — why the depth sweep cannot be re-run yet

**Tooling:** [`archlab gate envelope|depth`](../../src/archlab/corpus_gate.py) ·
**Run:** 2026-07-31, A100, ~45 min across three diagnostics

## What this is

[`attn-res-depth`](attn-res-depth.md) ended blocked: its corpus was depth-saturated, so no
depth ablation on it could mean anything. The fix was supposed to be a corpus where depth is
*required* — and then to re-run the sweep.

The corpus was built and the sweep was **not** re-run, because the corpus failed its gate. This
records what was measured and why another attempt is not worth GPU time without a better idea.

## The compositional corpus

A chain gives its answer only after `chain_len` sequential function applications:

```
[CHAIN] x0 f_a f_b ... f_z [ANSWER] y        y = f_z(...f_b(f_a(x0)))
```

Intermediates are never emitted — emitting them would let a 1-layer model chain stepwise across
positions and destroy the depth requirement. Functions are fixed per corpus seed, so they are
learnable. [Tests](../../tests/test_data.py) assert every answer really is the composition, and
that the chain body contains function tokens only.

The corpus is sound. It is the *learning* that does not happen.

## Three measurements

**1. Depth does not help (the original gate).** Residual baseline, 600 steps:

| depth | 6 | 12 | 24 | 48 |
|---|---|---|---|---|
| compose loss | 2.2046 | 2.2245 | 2.2104 | 2.2657 |

Against chance = ln(16) = 2.773 and a marginal-only baseline of 2.70. The model is barely below
"predict the answer distribution and ignore the chain".

**2. There is a sharp learnability cliff, and it is not about budget.** Depth fixed at 12:

| chain_len | 600 steps | 2400 steps |
|---|---:|---:|
| 1 | 0.0023 | — |
| 2 | 0.0802 | **0.0006** |
| 4 | 2.1786 | 1.9065 |
| 8 | 2.2324 | — |

Length 1 and 2 are solved outright — so the task is well-formed and composition *is* learnable
in principle. Length 4 is not, and 4x the budget buys 0.27 nats. The cliff sits between 2 and 4
and does not move with training time.

**3. Depth does not cross the cliff either.** At `chain_len=4`, 600 steps:

| depth | 6 | 12 | 24 | 48 |
|---|---|---|---|---|
| compose loss | **2.1600** | 2.1746 | 2.2134 | 2.3117 |

Monotonically **worse** with depth: 6 -> 48 costs +0.1517.

## The mechanism, and why it explains both corpora

At length <= 2 there are at most 8^2 = 64 function sequences over 16 states — about a thousand
table entries, comfortably memorized. Depth is irrelevant because no algorithm is needed. At
length 4 there are 8^4 = 4096 sequences; memorization stops fitting, and solving it would
require *learning a sequential composition algorithm*. That never happens, at any depth tested,
with any budget tested.

**Requiring depth is not the same as inducing a model to use depth.** My design premise was
that a task which cannot be solved without composition would force the model to compose, and
depth would then pay. It does not. The model memorizes until memorization fails, then fails —
it does not fall back on an algorithm.

That single mechanism covers both corpus attempts. The recall corpus was trivial-or-impossible
with nothing between; this one is memorizable-or-impossible with nothing between. Neither has a
regime that is hard, learnable, and depth-sensitive at once, which is exactly what a depth
ablation needs.

## What this implies for the earlier result

It weakens the reading of [`attn-res`](attn-res.md) and [`attn-res-depth`](attn-res-depth.md).
Those found AttnRes losing to a plain residual, with the cost growing in depth. If nano-scale
models do not productively use depth on *any* task constructible here, then "attention over
depth does not help" may be a statement about nano-scale training rather than about the
mechanism. The measured cost is real; the absent benefit is now much less informative than it
looked, because there was no benefit available to anyone.

The finding that survives intact is the *relative* one: full AttnRes's cost grows ~3x faster
with depth than blocked AttnRes's. That is a comparison between two AttnRes variants under
identical conditions, and it does not depend on depth paying off.

## Why I am stopping rather than trying corpus #4

Three designs have failed for the same reason, and the reason is not a parameter I can nudge.
Candidate fixes and why each is unconvincing at this scale:

- **Shorter chains (length 3).** Likely lands in the memorization regime, so depth stays
  irrelevant. It would produce a number, not an answer.
- **Curriculum (length 1 -> 2 -> 4).** Plausible, and the standard remedy for exactly this
  failure — but it is a training-procedure research project, not a corpus fix, and its result
  would be about curricula rather than about AttnRes.
- **Bigger models.** Probably the real answer, and out of scope for a repo whose premise is
  laptop-scale primitives.

Running the sweep now would yield clean-looking numbers on a task where no arm learns the thing
being measured — the same error as `attn-res-depth`, repeated after being warned by it.

## Permutation composition: corpus #3 (2026-08-08)

Iterated permutations (non-contracting bijections) instead of arbitrary maps. The hypothesis
was that non-contracting chains prevent the memorization shortcut — every step must be applied,
so the answer depends on the full chain.

**Chain length envelope** (12 layers, residual, 600 steps unless noted):

| chain_len | steps | compose loss | Status |
|-----------|-------|-------------|--------|
| 1 | 600 | 0.0015 | Solved |
| 2 | 600 | 0.0025 | Solved |
| 4 | 600 | 1.3897 | **Chance** (1.386) |
| 8 | 600 | 1.3930 | **Chance** |
| 4 | 2400 | 1.3476 | Still chance |
| 8 | 2400 | 1.3912 | Still chance |

**Same cliff as corpus #1.** The non-contracting property did not help. The model solves
chain_len ≤ 2 outright, then fails completely at chain_len ≥ 4. Four times the training
budget buys essentially nothing.

Depth gate at chain_len=4 could not run (GPU contention), but the envelope alone is
sufficient: the cliff is in the same place, so depth-gate results would be redundant.

**What this eliminates:** the failure mode is not about contracting maps allowing memorization
shortcuts. Permutations are the hardest possible composition task (bijections, every step
irreversible), and the model still cannot compose at length 4. The limitation is more
fundamental — likely the model's capacity to track state through sequential operations.

## What would unblock it

A task that is **hard, learnable, and depth-sensitive** simultaneously. The gate is cheap and
now reusable — `archlab gate envelope` finds whether a learnable-but-hard regime exists, and
`archlab gate depth` finds whether depth moves the boundary. Any candidate corpus should clear
both before a sweep is launched.

Known families worth trying, in rough order of promise:

1. **Dyck / nested brackets.** Depth-separation results are strongest here, and difficulty is
   tunable continuously by nesting depth rather than in a cliff.
2. **Graph reachability at controlled hop counts** — hop count maps to required depth directly.
3. **Bigger models.** Probably the real answer, and out of scope for a repo whose premise is
   laptop-scale primitives.
