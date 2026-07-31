# activation-bound — what does capping the activation cost, and what does it buy?

**Spec:** [`configs/ablations/activation-bound.yaml`](../../configs/ablations/activation-bound.yaml) ·
**Runner:** [`archlab/ablations/activation_bound.py`](../../src/archlab/ablations/activation_bound.py) ·
**Run:** 2026-07-31, A100, ~2.7 h, 16 runs (4 arms x 2 precisions x 2 seeds)

## Hypothesis

K3 §2.3.2 justifies SiTU-GLU entirely on precision: SwiGLU multiplies two *unbounded* factors,
so coincident large coordinates compound into outliers, and at MXFP4 weights / MXFP8 activations
an outlier is an overflow. SiTU-GLU bounds the product at `b1*b2 = 100`. No quality comparison
is shown, and the constants 4 and 25 are given without justification.

Two questions that need separating:

- **(a)** at bf16, does the cap cost anything? If yes it is a precision tax, not a free win.
- **(b)** under FP8-range activations, does the unbounded baseline actually break?

**Falsifier:** if capped and uncapped are indistinguishable at bf16 *and* nothing overflows
under FP8 range, the mechanism is inert at this scale.

## Simulated precision — what this can and cannot say

The A100 is sm_80. `float8` dtypes and casts work, but `torch._scaled_mm` needs sm_90, so real
FP8 matmuls are unavailable. `fp8_sim` fake-quantizes activations through `e4m3` — cast down and
back — reproducing the format's **range and precision** exactly while arithmetic stays in bf16.

Verified rather than assumed: the quantizer is attached to every FFN, changes forward outputs by
6.4e-2, and introduces **2.3% mean relative error** per activation at the magnitudes observed.

This is faithful for the numerical question and says **nothing about throughput**. No speed
conclusion may be drawn from it.

## Results

### (a) At bf16 the cap does not measurably help or hurt

| arm | up cap | binds? | mean loss | seed spread | vs swiglu |
|---|---:|---|---:|---:|---:|
| swiglu | inf | no | 2.7666 | 0.0023 | — |
| situ-16-100 | 100 | no | 2.7671 | 0.0000 | +0.0005 |
| situ-4-25 | 25 | **yes** | 2.7529 | 0.0260 | −0.0137 |
| situ-2-8 | 8 | **yes** | 2.7599 | 0.0165 | −0.0067 |

Both binding arms have a lower mean, but **every gap is smaller than that arm's own seed
spread**. Not resolvable at two seeds. The honest statement is "no measurable difference",
not "capping helps".

### (b) FP8-range quantization changes nothing

| arm | bf16 | fp8_sim | delta | seed spread |
|---|---:|---:|---:|---:|
| swiglu | 2.7666 | 2.7672 | +0.0006 | 0.0023 |
| situ-16-100 | 2.7671 | 2.7671 | −0.0000 | 0.0000 |
| situ-4-25 | 2.7529 | 2.7602 | +0.0073 | 0.0260 |
| situ-2-8 | 2.7599 | 2.7641 | +0.0041 | 0.0165 |

For every arm the precision effect is smaller than seed noise — despite 2.3% relative error
injected on every activation at every step. The model trains straight through it.

### The threshold question, which noise cannot touch

| | |
|---|---|
| peak activation, all 16 runs | **87.5** |
| `e4m3` ceiling | 448 |
| headroom | **5.1x** |
| overflow events | **0** |
| non-finite values | **0** |
| gradient spikes / non-finite steps | 1 / 0 |

Nothing came within 5x of the range the cap exists to protect. **The precision motivation is
inert at this scale** — which is a threshold fact, not a small effect, so unlike the loss
comparisons it is not vulnerable to the seed noise above.

## The one robust finding: binding predicts variance

Sorting by whether the cap is below the natural activation scale (~58) separates the arms
cleanly, pooled over precisions:

| arm | up cap | binds? | seed spread |
|---|---:|---|---:|
| swiglu | inf | no | 0.0023 |
| situ-16-100 | 100 | no | 0.0015 |
| situ-4-25 | 25 | **yes** | **0.0263** |
| situ-2-8 | 8 | **yes** | **0.0165** |

A ~10x gap with no overlap, and group membership predicted in advance by the mechanism rather
than fitted after the fact. `situ-16-100` is the cleanest confirmation: a cap set above the
natural scale is a *no-op*, landing within 0.0005 of SwiGLU with a seed spread of 0.0015.

**This runs against the report's framing.** SiTU-GLU is presented as the stabilising choice; here
the arms whose cap actually engages are the run-to-run *unstable* ones. Two seeds per arm is thin
for a variance claim, but the split tracks the mechanism, not the labels.

## Verdict

**Both questions answered, both negative at this scale.** The cap neither costs nor buys
measurable quality at bf16, and there is nothing for it to protect against under FP8 range. What
it does measurably do is increase seed variance when it engages.

This does not contradict K3. Their concern is 2.8T-parameter models where activation outliers
are documented, and 3M parameters reaching 87 of 448 says nothing about what 2.8T reaches. What
it does establish: **the mechanism is not self-evidently beneficial** — it does not pay for
itself on quality grounds, so the case for it rests entirely on the overflow argument holding at
scale, which this cannot check.

## Two corrections made during the run

Recorded because both changed the analysis, and both are now pinned by tests
([`test_precision.py`](../../tests/test_precision.py)):

1. **`b1*b2 = 100` bounds the product; branches cap separately at 4 and 25.** I predicted
   `situ-4-25` would leave activations near SwiGLU's ~58 peak since 58 < 100. It did not — peak
   fell to 25.2, because the *up branch* caps at 25. This determines which arms bind at all, and
   I had it wrong when the run started.
2. **`e4m3fn` overflow saturates at 448; it does not produce NaN.** Overflow would therefore be
   silent information loss rather than a crash — worse for an experiment, since only the
   activation statistics would reveal it.

A third correction was to my own over-reading: on seed 0 alone, `situ-4-25` beat SwiGLU by 0.028
and I described an inverted-U with K3's constants at the optimum. Seed 1 showed a tie. The effect
was inside that arm's seed variance, and a result that flattering to the source paper deserved
more suspicion than I gave it at the time.

## Caveats

- Two seeds. Adequate for the threshold result, thin for the loss and variance comparisons.
- 3M parameters against 2.8T; nano scale cannot reach the regime the report is about.
- Simulated FP8: range and precision faithful, throughput not modelled.
- One corpus, one width, 1200 steps.

## Next move

- **More seeds on the variance finding** — it is the only result with a mechanism behind it, and
  5 seeds would make it a claim rather than an observation.
- **Force the regime**: scale `d_hidden` or remove the pre-FFN RMSNorm until activations
  genuinely approach 448, then re-run. That would test the overflow argument directly instead of
  observing that it never triggers.
- The `beta` sweep deserves a finer grid if the variance effect holds, to see whether it is
  monotone in cap tightness or has structure.
