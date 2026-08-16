# agentic-models

Understanding frontier agentic-model architectures from first principles — by building the
primitives small enough to run on a laptop, and testing them against the claims their papers
make.

The unit of work is a **study**: one model or paper, read closely, with every mechanism it
introduces implemented in [`archlab`](src/archlab/) and pinned down by a test that asserts the
property the paper claims for it. A mechanism you can only describe is a mechanism you don't
understand yet.

## Studies

- **[`kimi-k3`](studies/kimi-k3/)** *(active)* — Moonshot's 2.8T/104B open MoE. KDA hybrid
  attention, Attention Residuals, Stable LatentMoE with Quantile Balancing, 1M context via NoPE.
- _planned_ — DeepSeek V3/V4 · GLM-5.2 · Qwen3.x · the linear-attention lineage
  (DeltaNet → Gated DeltaNet → Mamba-2 → RWKV-7 → Kimi Linear) · agentic RL and serving stacks.

## Primitives

| Module | Mechanism | Originates in | K3's change |
|---|---|---|---|
| [`attention/kda.py`](src/archlab/attention/kda.py) | Delta rule + channel-wise forget gate; naive recurrence *and* chunkwise form | Kimi Linear, on DeltaNet/Gated DeltaNet | Lower-bounded decay (Eq. 5), full-rank output gate (Eq. 6) |
| [`attention/mla.py`](src/archlab/attention/mla.py) | Latent-KV global attention | DeepSeek-V2 | NoPE, full-rank output gate (Eq. 7) |
| [`attention/heads.py`](src/archlab/attention/heads.py) | Head plumbing + the gate both attention layers share | — | The gate itself (Eq. 6/7) |
| [`depth/attn_res.py`](src/archlab/depth/attn_res.py) | Attention over depth — full and blocked | **Kimi Team (AttnRes preprint)** | The mechanism, and the blocked form (§2.2) |
| [`moe/latent_moe.py`](src/archlab/moe/latent_moe.py) | Routed experts in a compressed latent space | LatentMoE (Elango et al.) | RMSNorm before up-projection; the "Stable" fixes (§2.3.1) |
| [`moe/quantile_balance.py`](src/archlab/moe/quantile_balance.py) | Load balancing by exact dual quantiles | **K3** (loss-free bias from DeepSeek-V3) | The whole method (§2.3.3, App. C-D) |
| [`activations/situ_glu.py`](src/archlab/activations/situ_glu.py) | Smoothly-capped SwiGLU | **K3** (SwiGLU baseline: Shazeer) | The whole method (§2.3.2, App. B) |
| [`memory.py`](src/archlab/memory.py) | Cache/state byte accounting | — | — |

Only three rows are genuinely K3's invention (bold): AttnRes, Quantile Balancing, SiTU-GLU.
The rest K3 inherited and modified, which is exactly why these live in shared `archlab` rather
than under `studies/kimi-k3/` — the next studies vary the *same* modules. Mamba-2 and RWKV-7
swap KDA's decay parameterization while reusing the chunkwise scaffolding; DeepSeek studies
reuse MLA with RoPE restored. A primitive moves into a study folder only if no second study
would ever touch it.

Clarity beats speed everywhere: KDA loops over chunks in Python, LatentMoE dispatches with a
loop over experts. Where a paper gives both a reference form and a fast form, both are here and
a test asserts they agree — that equivalence is usually where the real content lives.

## Setup

```bash
uv sync
uv run pytest                      # 78 tests, CPU, ~2s
uv run archlab ablate qb-scale     # the one ablation needing no training, ~80s
```

Runs anywhere on CPU. A GPU box is for the four ablations that need training, not for the
suite.

## What the tests are for

They are the check on whether a mechanism was understood, so they assert claims rather than
shapes. A few that earned their keep:

- **KDA chunkwise ≡ serial recurrence** across chunk sizes 1–64 — the UT-transform algebra is
  only trustworthy if it reproduces Eq. 1 exactly.
- **Bounded decay keeps `1/Γ` finite** where the unbounded `-Softplus` baseline provably
  overflows — the numerical argument that unlocks dense Tensor Core tiles.
- **Quantile Balancing beats sign-updates** at a matched step budget, and its exactness
  **degrades under score ties** — a caveat the report leaves implicit, and one the
  [`qb-scale`](docs/experiments/qb-scale.md) sweep showed gets worse as the expert pool grows.
- **SiTU-GLU is bounded by β₁β₂ = 100**, matches SwiGLU near the origin, and recovers it as
  β→∞ — but its gradient *does* still underflow deep in saturation.

Two of those findings came out of tests failing for real reasons; they're written up in the
study notes rather than smoothed over.

## Layout

```
src/archlab/              shared primitives — reused across studies, one module per mechanism
tests/                    one file per area, asserting paper claims
studies/<model>/          per-model close reading + the papers themselves
configs/ablations/*.yaml  one spec per open question (see docs/experiments/)
docs/                     cross-cutting notes — see docs/README.md for the split
```

## Findings

Five ablations run, each with its hypothesis written down *before* its runner existed. Four
came back negative; the fifth — the Dyck depth sweep — produced the first positive result and
also the most destructive one. Full writeups in [`docs/experiments/`](docs/experiments/).

| Ablation | Question | Result |
|---|---|---|
| [`qb-scale`](docs/experiments/qb-scale.md) | Does QB's edge over sign updates widen with expert count? | **No** — it *narrows*, 149x to 39x |
| [`attn-res`](docs/experiments/attn-res.md) | How much of the 2.5x is the depth axis alone? | Residual beats every AttnRes arm at 12 layers |
| [`attn-res-depth`](docs/experiments/attn-res-depth.md) | Is that a depth artifact? | Unanswerable — the corpus rewards no depth |
| [`activation-bound`](docs/experiments/activation-bound.md) | Is capping activations free at BF16, load-bearing at FP8? | Neither — nothing came within 5x of FP8 range |
| [`attn-res-dyck`](docs/experiments/attn-res-dyck.md) | Does AttnRes help on a corpus where depth pays? | **Helps at 12 layers (-0.40 gap), catastrophically fails at 48 (+1.46)** |

### What holds up

Ordered by how much weight the evidence bears:

1. **AttnRes has a depth-dependent scaling limit.** On a corpus where depth actually helps (Dyck,
   nesting 64), AttnRes-full helps at 12 layers (gap -0.40) but catastrophically fails at 48
   layers (gap +1.46, flatlines at chance). The O(L²) depth attention becomes too diffuse to
   learn over 48 sources. Block-6 survives because it limits each block to 6 sources — the same
   complexity as full AttnRes at 12 layers. This is an optimization failure of the mechanism,
   not a corpus artifact — the residual baseline at 48 layers converges cleanly to 0.34.
   **Caveat:** these numbers are specific to nano scale (~13M params) on synthetic data. Whether
   K3's 93-layer AttnRes fails the same way is an open question that requires running at K3's
   scale.
2. **Load imbalance under QB rises with expert count — but not for the reason I claimed.**
   Trained routers at K3's exact configuration (896 routed, 16 active) show imbalance climbing
   1.145 → 2.188 from 8 to 896 experts, so the difficulty K3 cites is real. The mechanism I
   proposed for it is not: the same routers show **0.0000 saturation and 0.0000 ties**, sitting
   3–5× in logit scale below where ties begin ([`trained-router`](docs/experiments/trained-router.md)).
   Tie degradation remains real as a property of the quantile estimator given tied inputs — the
   algebra and the 5-seed replication stand — but the claim that *"router temperature is a
   precondition for balancing at K3's scale"* is **withdrawn**. It generalised from an input
   regime training does not visit.
3. **A cap that engages raises seed variance ~10x.** Clean separation, no overlap, and group
   membership predicted in advance by the mechanism. This is the *opposite* of the stabilising
   role §2.3.2 describes for SiTU-GLU.
4. **AttnRes's cost scales with depth, not with sources attended.** Blocked at 48 layers costs
   5.5x more than full at 6, on comparable source counts. Blocking cuts cost ~3x without
   stopping its growth — suggesting K3's blocking may do optimization work, not only the memory
   work §2.2 claims.
5. **Requiring depth is not the same as inducing a model to use depth.** One mechanism explains
   two failed corpus designs: models memorize until memorization fails, then fail, rather than
   falling back on an algorithm.

### Which findings actually depend on scale

Proxy-model results are externally unverified by construction, and
[validity-threat work](https://arxiv.org/pdf/2606.05029) is blunt about it: predictability
across scale cannot be established by proxy experiments alone, and rankings can fail to
transfer even between 125M and 1B. That threat is real — but it applies to claims about
**learned model quality**, not to claims about **algorithms and estimators**, which are
mathematical properties verified numerically.

Sorting the findings that way changes what half of them are claiming:

| Finding | Kind of claim | Needs scale transfer? |
|---|---|---|
| KDA chunkwise ≡ serial recurrence | algebraic identity | **no** — holds to 7e-15 |
| SiTU-GLU bounded by β₁β₂ | mathematical | **no** |
| QB tie degradation at 896 experts | property of the quantile solver given tied inputs | **no** — but see below |
| Histogram estimator error floor | numerical | **no** |
| MoE bias autograd leak | correctness bug | **no** |
| Cross-process nondeterminism | measurement property | **no** |
| AttnRes helps at 12 layers, fails at 48 | learned model quality | **yes** — nano scale on synthetic data; may not transfer to K3's 93 layers |
| AttnRes does not help at 12 layers (original) | learned model quality | **yes** — and it now survives a 4x width ladder ([`attn-res-scale`](docs/experiments/attn-res-scale.md)) |
| Activation cap does not help | learned model quality | **yes** |

`qb-scale` ran at **896 experts — K3's actual count** — on synthetic routers, so it is not a
proxy for anything; the same holds for the histogram floor and the algebraic checks. Scale was
never the limiting factor for those.

**And that turned out to be the wrong thing to feel safe about.** This table sorts findings by
whether they need *scale* transfer, and puts the QB tie result in the safe column — correctly,
since the algebra holds at any size. It was still the finding that broke, because the threat was
never scale: it was whether the **input regime is one training actually produces**. Synthetic
inputs are exactly as unverified as a small model, and a taxonomy organised around parameter
count cannot see that. Read the "no" column as *"this claim is not threatened by scale"* — which
is narrower than *"this claim is safe"*.

So the honest statement is not "these findings may not transfer". It is: **two of them may not,
and both are already reported as negatives.** Those two are being tested directly — see
[`attn-res-scale`](configs/ablations/attn-res-scale.yaml), which measures whether the *ranking*
of arms is stable across a width ladder, and fixes the fixed-hyperparameter confound that
[ICLR 2026](https://arxiv.org/html/2512.24503v2) identifies as a known cause of proxy
non-transfer.

**No claim here contradicts K3.** The report's arguments concern 93 layers, documented
activation outliers and 896 experts under real load; the two quality claims above cannot reach
that regime and do not pretend to.

### Blocked

[`hybrid-ratio`](configs/ablations/hybrid-ratio.yaml) should be re-scoped or dropped — a
published version (arXiv 2606.15378) already answers the same question at larger scale.

[`kda-state-capacity`](configs/ablations/kda-state-capacity.yaml) may be unblocked — Dyck's
nesting depth is a direct dial on how much recurrent state is needed, and the corpus is
validated with a measured noise floor.

## Why the harness is the product

The five ablations produced four negatives and one positive that is also negative at scale. The
thing worth keeping is the measurement stack, because of what it caught.

**The real subject is whether learning-by-ablation produces knowledge at all.** Nobody can train
a frontier model to test a design idea, so the field runs small proxies and takes the winner.
[Validity-threat work](https://arxiv.org/pdf/2606.05029) is blunt that this may not work:
rankings can fail to transfer even between 125M and 1B. If A beats B at 3M and B beats A at
100M, every small ablation is theatre. That question sits underneath every result here.

**Two claims have been withdrawn, both from the same failure**, and it is one that generalises
past this repo:

> A component was verified, and the system was assumed to use it.

- [`qb-scale`](docs/experiments/qb-scale.md) verified that the quantile estimator degrades on
  tied router scores. It never checked whether training *produces* tied scores. It does not —
  0.0000 at K3's exact 896/16 configuration ([`trained-router`](docs/experiments/trained-router.md)).
- [`mup-attempt`](docs/experiments/mup-attempt.md) verified that `mup_param_groups` returns
  correctly scaled groups. It never checked that the optimizer *received* them. It did not, for
  three commits, and a wrong negative result was published from the difference.

In both cases the unit test passed and was correct. That is exactly what hid the problem: a
green test on a function nothing calls is indistinguishable from a green test on a function that
works. The same shape appears in any eval harness whose metric is right but wired to the wrong
column, or any training run where one config flag never reaches the model — code correct, result
meaningless, nothing failing loudly.

**What the harness does about it**, each earned by a specific failure rather than adopted on
principle:

| Guard | Catches | Paid for by |
|---|---|---|
| Determinism on by default | cross-process noise read as an effect | a 0.20 spread at identical config and seed |
| `archlab gate` before a sweep | corpora where the capability is never learned | two GPU-hours spent learning it slowly |
| Positive controls on null results | a broken detector reporting zero | `trained-router`'s 0.0000, meaningless without one |
| muP before a width ladder | the optimizer's drift read as the mechanism's | SP's optimum moving 1e-2 → 3e-3 |
| Contract tests at the seam | a correct component nothing calls | the muP wiring bug |
| `F`/`ARG` in the ruff gate | dead imports and unused parameters | the same bug, one commit earlier |

The last two are the general lesson. **Test where the component meets the system, not the
component again** — the muP test now spies on the optimizer during a real training call, because
re-testing the builder could never detect a missing call to the builder. And let the linter own
the whole class: `F401` would have caught that bug on the commit that introduced it, and
adopting it cost three fixes repo-wide. It was free the entire time.

For evaluation-driven work that stack is the asset — a setup whose *negative* results are worth
trusting.

## Related

[`small-smart-models`](../small-smart-models) asks the complementary question — how far these
models compress. Expert-activation structure understood here feeds the expert-dropping and
importance-aware quantization studies there.
