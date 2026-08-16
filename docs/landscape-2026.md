# The agentic-model landscape, mid-2026

A sweep of where frontier agentic architectures and their training stacks actually are, done
after the [K3 study](../studies/kimi-k3/) and four ablations — so the last section is the point
of it: **where the literature corroborates, anticipates, or contradicts what this repo
measured.**

Sources are linked inline and **verified in [`landscape-sources.md`](landscape-sources.md)** —
every arXiv ID resolves and every title was fetched and matched against how it is cited here.
That file also lists the claims below that have *no* archived source and should not be trusted
without checking. Read dates: 2026-08-01, verified 2026-08-02. Where a claim is a single blog
post rather than a paper it is marked as such.

---

## 1. Architecture: the convergence is real

Twenty-two open-weight models later, the 2026 frontier design has largely settled on the same
four choices, and K3 sits squarely inside that consensus rather than outside it:

| Choice | Who |
|---|---|
| **Sparse MoE**, low active fraction | K3 (16/896), DeepSeek V4, Nemotron 3 (550B/55B active), Arcee Trinity (400B/13B) |
| **Hybrid attention** — cheap mixer + periodic full attention | K3 (KDA 3:1), Nemotron 3 (Mamba-2 + sparse GQA), Arcee Trinity (SWA 3:1) |
| **1M context** | K3, DeepSeek V4, Nemotron 3 |
| **4-bit weights** | K3 (MXFP4), Nemotron 3 (NVFP4) |

Two things worth noting for anyone reading the K3 report in isolation:

- **The 3:1 hybrid ratio is not idiosyncratic.** Arcee Trinity independently uses 3:1
  (sliding-window to global), and Nemotron 3 Nano goes much further — most layers Mamba-2, full
  attention in only a small subset. K3's ratio is mid-range, not aggressive.
- **NoPE-on-global-layers is a converged pattern too.** Arcee Trinity uses RoPE in local layers
  and NoPE in global layers; K3 uses NoPE on all MLA layers. Same idea: let the
  position-sensitive mixer carry position, free the global path from a frequency base.

### The paper that most directly overlaps this repo

[Rethinking the Role of Efficient Attention in Hybrid Architectures](https://arxiv.org/abs/2606.15378)
(Qiao et al., Tsinghua/OpenBMB) is essentially the published version of the
[`hybrid-ratio`](../configs/ablations/hybrid-ratio.yaml) ablation specified here and never run.
Its headline: **efficient-attention design affects how *fast* long-context capability emerges,
not the eventual ceiling** — different hybrids converge to comparable long-context performance
given enough training.

That result would have changed how I framed my own depth work, and it rhymes with what I found:
a mechanism can look decisively worse at a short budget and be indistinguishable at
convergence. It also reports a "large-window laziness" effect, which is the kind of
mechanism-level finding my nano-scale setup is too small to see.

**Conclusion for this repo:** `hybrid-ratio` should be re-scoped or dropped. Running a worse
version of a published result is not a good use of an A100.

---

## 2. Post-training: a consolidated stack

The 2026 recipe has stabilised into SFT (format, cold start) → preference optimisation
(DPO/SimPO) → RL with verifiable rewards (GRPO/DAPO). K3's pipeline — SFT → RL across
domains × effort levels → multi-teacher on-policy distillation — is a variant of this, with the
distillation consolidation step being its distinctive move.

The framing I found most clarifying: **agentic RL expands what the trajectory contains while
preserving the GRPO update rule.** The action space grows from tokens to structured multi-step
decisions interleaved with observations, and *introducing an environment* is the single
structural change that induces every new failure mode.

Named difficulties, all of which K3's report addresses in some form:

| Problem | Field's approach | K3's approach |
|---|---|---|
| Cascading failure — one bad tool call poisons the rest of a trajectory | [PivotRL](https://arxiv.org/pdf/2603.21383): reward functionally-equivalent actions at high-variance "pivot" turns | Per-token regularisation tolerating stale off-policy data |
| Long-horizon rollout cost | Async rollout frameworks | Partial rollouts + resumable microVM sandboxes |
| Context as the bottleneck | Context folding, summarisation, memory-as-action | 1M window + context-management modules in the harness |
| Coarse trajectory-level supervision | [Self-Distilled Agentic RL](https://arxiv.org/html/2605.15155v1): dense token-level teacher signal, gated as an auxiliary objective | Multi-teacher on-policy distillation (MOPD) |

**"The verifier is the new dataset"** is the sharpest one-line summary of where effort is
going — verifier engineering as its own discipline. K3's §4.2 (white-box environments, hidden
verifiers paired with public ones, hacking-detection for kernel tasks) is a substantial
instance of exactly that, and reading it next to this literature makes clear the environments,
not the RL algorithm, are the contribution.

---

## 3. Harness co-design: the newest thread

This is the part of the user's question the field has only recently named, and it is moving
fast:

- **[The Interplay of Harness Design and Post-Training in LLM Agents](https://arxiv.org/pdf/2606.25447)**
  — the direct treatment of "how do you optimise model *to* harness".
- **[Polar](https://arxiv.org/html/2605.24220v1)** — runs agentic RL against *any* harness by
  intercepting LLM API traffic through a proxy rather than instrumenting the harness. The agent
  runs unchanged; existing harnesses become RL environments with no internal code changes.
- **[HarnessForge](https://arxiv.org/pdf/2606.01779)** — joint harness/policy evolution. Its
  stated gap is that prior work treated harnesses as optimisation targets but optimised
  external structure while leaving *model-side compatibility implicit*.

**K3 is on the same trajectory and arrived early.** Its unified white-box RL environment —
composing harnesses from configurable modules so the model does not overfit one tool schema,
and instantiating Claude Code / Codex / OpenClaw / Kimi Code as configurations — is
harness-co-design in all but name, shipped in a frontier model rather than a paper. Worth
recognising, because the report presents it as infrastructure rather than as a research
contribution.

---

## 4. Where this repo's findings sit

The genuinely useful output of the sweep. Three of four ablations turn out to have published
counterparts, and the comparison is not flattering in every case.

### Partly corroborated, partly withdrawn — [`qb-scale`](experiments/qb-scale.md)

I found that Quantile Balancing degrades under router score ties, and that at 896 experts a
saturated router defeats every balancer including the offline solver — concluding router
temperature is a precondition for balancing at scale.

**The conclusion has since been withdrawn.** [`trained-router`](experiments/trained-router.md)
trained routers by gradient descent and found zero saturation and zero ties at every expert
count including K3's 896/16. The section below is left as written because the reasoning error it
illustrates is instructive, with the correction inline.

The MoE literature reports the same failure from the other direction: under **super-high MoE
sparsity, auxiliary-loss-free balancing causes significant load imbalance in lower layers**, and
at least one 2026 study finds loss-free balancing *insufficient alone*, with the emerging
default being ALF **plus a low-weight auxiliary loss** rather than bias-updates outright.

So the field has independently converged on "the loss-free bias mechanism has a
sparsity-dependent failure mode".

I read that as corroborating the tie-degradation result, on the grounds that ties are a specific
instance of the general pattern. That was the error — **a shared conclusion is not evidence for
a shared mechanism.** The field observes imbalance rising with sparsity; so did I, in a trained
router (1.145 → 2.188 from 8 to 896 experts). But the tie mechanism I proposed to explain it is
absent from those same routers. Both observations are real and the explanation connecting them
is not.

The lesson generalises past this repo: finding published work that agrees with your *conclusion*
feels like validation and tests nothing. The corroboration to look for is of the **mechanism**,
which is also the part a literature sweep is worst at confirming.

### Anticipated — the depth ablations

[`attn-res`](experiments/attn-res.md) and [`attn-res-depth`](experiments/attn-res-depth.md)
concluded that a mechanism's benefit is not scale-free and may not appear at short budgets. The
hybrid-attention paper above says the same thing more rigorously and at larger scale: efficient
attention changes *how fast* capability emerges, not the ceiling. My version is a weaker
instance of a stronger published result.

### Known — the determinism finding

The [cross-process nondeterminism result](experiments/dyck-corpus.md) (0.20 spread at identical
config and seed; bit-identical once `use_deterministic_algorithms` is set) is well-trodden
ground. [Non-Determinism in TensorFlow ResNets](https://arxiv.org/pdf/2001.11396) attributes
**74% of test-accuracy variance and 87% of loss variance** to GPU nondeterminism alone, and the
RL literature notes deviations compound as training progresses. Standard guidance is exactly
what I landed on: make runs deterministic, or report seeds *and* error bars, because
single-run comparison may be measuring floating-point noise.

Not novel. But it was worth measuring here, because the harness had been producing
cross-invocation comparisons for four ablations and nothing in the setup announced that they
were invalid.

### Still open — proxy validity

The uncomfortable one. [Validity Threats for Foundation Model Research](https://arxiv.org/pdf/2606.05029)
states it directly: **predictability across scale cannot be determined by proxy-model
experiments alone; it must be verified by conducting them at scale.** And an ICLR 2026 paper
finds proxy-model rankings can fail to transfer even between 125M and 1B under tuned
hyperparameters.

That is the central limitation of everything in this repo, and it is sharper than the caveat I
have been writing. "Nano-scale results indicate direction, never magnitude" is optimistic —
the literature says even *direction* (ranking) may not transfer. Transfer improves when the
proxy's training regime matches the target's optimality regime and when the proxy metric is
aligned to the target task; my setup does neither by construction.

---

## 5. What I would change

1. **Drop or re-scope [`hybrid-ratio`](../configs/ablations/hybrid-ratio.yaml).** It is a worse
   version of arXiv 2606.15378.
2. **Weaken the transfer caveat everywhere.** Current wording implies direction transfers and
   magnitude does not; the evidence does not support even that.
3. **Read the harness co-design thread properly** — Polar and HarnessForge are the closest
   published work to what K3's white-box environment does, and the K3 study note currently
   treats that section as infrastructure rather than as the contribution it probably is.
4. ~~**Keep the tie-degradation result.**~~ Withdrawn as a claim about routers at scale — the
   mechanism is absent from trained routers. It stands as a property of the quantile estimator
   given tied inputs, which is a narrower and much less interesting statement.
5. **Verify preconditions before generalising from a synthetic sweep.** The tie result was the
   repo's most-promoted finding and survived a literature check that appeared to corroborate it.
   What it had never survived was a measurement of its own precondition, which cost one 20-minute
   run.

## Caveats on this sweep

Web search, not a systematic review. The queries and result sets were never saved, so recall is
not merely unknown but unrecoverable — there is no way to tell what this missed. Read each claim
as "this exists", not as "this is the state of the art". Several sources are blog posts rather
than papers (marked where it matters). Numeric claims from secondary sources —
particularly the Nemotron and Arcee configurations — are not verified against primary technical
reports. Treat the *shape* of the landscape as reliable and individual figures as needing a
check before citation.
