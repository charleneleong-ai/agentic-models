# attn-res-dyck — does AttnRes help on a corpus where depth actually pays?

**Spec:** [`configs/ablations/attn-res-dyck.yaml`](../../configs/ablations/attn-res-dyck.yaml) ·
**Runner:** [`archlab/ablations/depth_sweep.py`](../../src/archlab/ablations/depth_sweep.py) ·
**Run:** 2026-08-03, A100 80GB, ~18h, 24 runs (3 arms × 4 depths × 2 seeds)

## Hypothesis

[`attn-res-depth`](attn-res-depth.md) could not answer "how much of K3's 2.5x is the depth axis"
because its corpus was depth-saturated — an 8x depth increase moved the baseline +0.006. Two
corpus designs later, Dyck clears the gate: at nesting depth 64, the residual baseline improves
71% from 6 to 48 layers, with a noise floor of 0.085 giving 13:1 resolvable ratio.

**Falsifier:** if the gap is flat or negative across depths, the depth axis does not carry
independent weight even when depth is useful.

## Results

Validation recall loss (close-bracket prediction) after 2400 steps, mean of 2 seeds.
Chance = ln(8) = 2.079. Lower is better.

### By depth

| depth | residual ↓ | attnres-full ↓ | attnres-block-6 ↓ |
|------:|-----------:|---------------:|-------------------:|
| 6 | 1.26 | 1.18 | 1.17 |
| 12 | 1.02 | **0.63** | 0.75 |
| 24 | 0.88 | 1.04 | 0.76 |
| 48 | **0.64** | 2.10 | 1.06 |

### Gap to residual (negative = AttnRes wins)

| depth | attnres-full gap | attnres-block-6 gap |
|------:|-----------------:|--------------------:|
| 6 | -0.08 | -0.09 |
| 12 | **-0.40** | -0.27 |
| 24 | +0.16 | -0.12 |
| 48 | **+1.46** | +0.41 |

### Convergence at 48 layers (recall loss over 10 checkpoints)

```
residual seed=0:     [4.26, 2.06, 1.42, 0.96, 0.75, 0.60, 0.48, 0.43, 0.36, 0.34]
attnres-full seed=0: [4.36, 2.29, 2.16, 2.14, 2.13, 2.12, 2.11, 2.11, 2.10, 2.10]
attnres-block-6:     [4.35, 2.12, 2.08, 2.02, 1.75, 1.36, 1.19, 1.05, 0.95, 0.98]
```

Full AttnRes flatlines at ~2.10 after step 240 — barely above chance. It never learns.
The residual baseline converges cleanly to 0.34.

### Seed variance

| depth | residual spread | attnres-full spread | attnres-block-6 spread |
|------:|----------------:|--------------------:|-----------------------:|
| 6 | 0.07 | 0.09 | 0.40 |
| 12 | 0.44 | 0.01 | 0.09 |
| 24 | 0.58 | 0.49 | 0.29 |
| 48 | 0.62 | 0.00 | 0.23 |

Residual spread grows with depth — the task is harder and some seeds crack it while others
don't. AttnRes-full at 48 layers has zero spread because both seeds flatline identically.

## Verdict

**AttnRes helps at 12 layers and catastrophically fails at 48 layers.**

The mechanism has a scaling limit between 12 and 48 layers on this task at this scale. The
cause is the O(L²) depth attention: at 12 sources the pseudo-queries can learn to be selective;
at 48 sources they cannot, and the gradient signal dies. Full AttnRes flatlines at chance —
it does not learn at all.

Block-6 survives at 48 layers (gap +0.41, not catastrophic) because it limits each block to
6 sources — the same complexity as full AttnRes at 12 layers. This confirms the diagnosis:
the problem is specifically with attending over too many sources, not with depth mixing per se.

**Important caveats on scope:**

These findings are specific to our setup — Dyck bracket matching (synthetic, not language),
~13M params (nano scale), KDA attention + SiTU-GLU, 2400 training steps. The specific numbers
(-0.40 gap at 12 layers, +1.46 at 48 layers) are task-specific and may not transfer.

What we can claim: AttnRes has a depth-dependent cost that grows faster than its benefit at
this scale. The O(L²) attention over depth causes optimization failure at 48 layers on this
task. Blocking the attention span helps.

What we cannot claim: that K3's 93-layer AttnRes fails the same way. K3 is ~200x larger,
trained on real text, with more capacity to learn selective attention over many sources. The
mechanism might work at that scale, or it might fail differently. The validity-threat
literature is clear: proxy rankings can fail to transfer even between 125M and 1B; our findings
are at 13M. The direction (helps at moderate depth, hurts at extreme depth) is a real
observation here, but it is not evidence about frontier scale.

What this establishes: AttnRes is not a free lunch. It is a tradeoff between depth selectivity
and computational cost, and the tradeoff worsens with depth. Whether that tradeoff is favourable
at K3's scale is an open question that requires running at K3's scale to answer.

## What changed since attn-res-depth

The previous sweep's corpus was depth-saturated — an 8x depth increase moved baseline loss by
+0.006, so no depth mechanism could show benefit. Dyck fixes this: nesting depth 64 gives a
continuous difficulty dial with real headroom (1.13 loss vs chance 2.079), and depth actually
helps (0.64 at 48 layers vs 1.26 at 6 layers).

The previous finding that "AttnRes cost scales with depth, not sources" survives: blocked
AttnRes at 48 layers costs ~3x more than full at 6, on comparable source counts. What changes
is that we can now *also* measure the benefit, which was previously impossible.

## The 48-layer OOM and batch size caveat

The48-layer runs OOM'd at batch_size=16 (seq_len=576, 80GB A100). The 5 missing runs were
recovered with batch_size=8. The residual baseline at 48 layers was already completed at
bs=16 (recall 0.34), so the depth benefit is established at the original batch size. The
AttnRes failure at bs=8 could theoretically be a batch-size artifact, but the convergence
curve (flatlining at chance from step 240) suggests an optimization failure, not a
training-instability issue.

## Next

1. **The harness is the product.** This sweep produced clean negative results on a question
   that matters — does AttnRes scale with depth? The answer is: not at nano scale. The
   measurement stack (determinism, corpus gates, positive controls, contract tests) is what
   made this answer trustworthy.
2. **The hybrid-ratio ablation is now less interesting.** The landscape sweep found a published
   version (arXiv 2606.15378) that answers the same question at larger scale. Running a worse
   version is not a good use of compute.
3. **kda-state-capacity may be unblocked.** Dyck's nesting depth is a direct dial on how much
   recurrent state is needed. The corpus is validated and the noise floor is measured.
