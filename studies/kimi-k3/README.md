# Kimi K3 — Open Frontier Intelligence

**Source:** [`papers/k3_tech_report.pdf`](papers/) · [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) · [weights](https://huggingface.co/moonshotai/Kimi-K3) · [blog](https://www.kimi.com/blog/kimi-k3)
**Released:** weights + report 16 Jul 2026, report rev. 27 Jul 2026 (47pp) · **Read:** 30 Jul 2026

2.8T total / 104B active MoE, native vision, 1M context. First open 3T-class model. Trails
Claude Fable 5 and GPT-5.6 Sol overall; beats every other model in its eval suite.

---

## The thesis

Test-time scaling (RL, reasoning effort, long-horizon agency) advanced fast in the open
ecosystem; pre-trained *foundations* did not, clustering at or just above 1T. Applying ever
better RL to similar-size bases means open progress converges while the gap to the strongest
proprietary systems widens. K3 pushes **both axes at once** — 3T-class base *and* RL across
domains and reasoning-effort levels.

The unifying engineering idea is **algorithm–system co-design**: architecture and
infrastructure designed together, not sequentially. Nearly every architectural choice below
has a systems justification, and several exist *only* for systems reasons.

## Architecture: scale information flow along three axes

Not "more parameters" — more *flow*, along sequence, depth, and width. Claimed result:
**~2.5× scaling efficiency over K2** (§3.2, Fig. 7 — equal validation loss at 1/2.5 the FLOPs).

| Axis | Mechanism | Implemented in |
|---|---|---|
| Sequence | Hybrid attention: 3 KDA + 1 Gated MLA per block | [`attention/kda.py`](../../src/archlab/attention/kda.py) · [`attention/mla.py`](../../src/archlab/attention/mla.py) |
| Depth | Attention Residuals — retrieve across layers, not accumulate | [`depth/attn_res.py`](../../src/archlab/depth/attn_res.py) |
| Width | Stable LatentMoE — 896 routed experts, 16 active | [`moe/latent_moe.py`](../../src/archlab/moe/latent_moe.py) |

### KDA — linear attention that pays for itself

A delta-rule recurrence with a channel-wise forget gate, carrying a fixed-size state
`S ∈ R^{d_k×d_v}` instead of a growing KV cache (Eq. 1). Three KDA layers per Gated MLA layer.

The change I found most instructive is **why the decay is lower-bounded** (Eq. 5). Kimi Linear
used an unbounded `-Softplus` log-decay. The chunkwise form divides keys by the cumulative
decay `Γ`, and since `Γ` is a product of factors in (0,1), `1/Γ` overflows. Bounding
log-decay below with a scaled sigmoid (`g_min = -5`) keeps cumulative log-decay over a
16-token tile inside `(-80, 0)` — within BF16 range. **That is a numerical-range argument that
buys a hardware win**: every causal tile can now use dense Tensor Core matmuls instead of the
explicit position-pair path Kimi Linear needed on diagonal tiles. Pure co-design.

> Verified in [`test_kda.py`](../../tests/test_kda.py): the chunkwise form reproduces the
> serial recurrence exactly across chunk sizes 1–64, and the unbounded baseline provably
> overflows where the bounded one does not.

### Attention Residuals — the historical rhyme

A standard residual stream compresses all prior layers into one state and adds to it — which
is what an RNN does over *time*, and precisely the bottleneck attention was invented to
remove. AttnRes makes the same move over **depth**: each layer holds a learnable pseudo-query
and *retrieves* from all preceding layers with data-dependent weights (Eq. 8–9). The RMSNorm
on keys is load-bearing — without it, layers with large-magnitude outputs dominate on scale
alone.

Full AttnRes costs `O(L²d)` arithmetic (free — depth < 100) but `O(Ld)` *memory*. Block
AttnRes (Eq. 10) partitions L layers into N blocks, summing within a block and attending
across only N block representations → `O(Nd)`. K3 uses 8 blocks × 12 layers; N ≈ 8 recovers
most of the benefit.

### Stable LatentMoE — and what "Stable" is doing

896 routed experts with 16 active (sparsity 56) is affordable because routed experts operate
in a **latent space** of half the model width, while shared experts keep a full-width path
(Eq. 11). Two failure modes appear at that sparsity, and each fix is worth knowing:

1. **Exploding activations.** The routed path chains `W_down` → gated multi-branch FFN →
   `W_up` into ~four consecutive matmuls; ill-conditioned at 2.8T. Fixed by an **RMSNorm
   before the up-projection** (§2.3.1) and by **SiTU-GLU** inside experts. Notably the RMSNorm
   improved validation loss on its own — not merely a stability patch.
2. **Balancing ~10³ experts.** DeepSeek-V3-style fixed-step sign updates equilibrate too
   slowly. Replaced by **Quantile Balancing**.

### Quantile Balancing — my favourite result in the report

Auxiliary-loss-free routing adds a per-expert bias to the Top-k score but omits it from the
mixture weights, so it steers dispatch without touching router gradients. V3 nudges that bias
by `b ← b + γ·sign(load_error)` — a SignSGD step, with γ trading speed against oscillation.

QB **jumps straight to the minimizer**. Appendix C shows the balanced-assignment LP has an
exact dual whose coordinate minimizers are *quantiles*, one along the token axis and one along
the expert axis — the same quantile, different axis, hence the name. No learning rate exists
to tune. The cutoff comes free from running Top-(k+1) instead of Top-k. At scale the exact
quantile is replaced by a **histogram estimator** (Appendix D) whose counts are additive, so
one all-reduce of bin counts recovers the quantile of the true pooled global batch regardless
of sharding — communication independent of token count.

> Reproduced in [`test_moe.py`](../../tests/test_moe.py). The alternating solver (Alg. 1)
> reaches 1.008× max/mean load from a 2× imbalance; a single deferred update recovers >80% of
> the imbalance in one step, beating a matched budget of sign steps.
>
> **Two things I found by testing that the report does not spell out:**
> - QB's exactness **degrades under score ties**. The derivation assumes no ties; a saturated
>   sigmoid router produces scores of exactly 0.0/1.0, and balance then plateaus around 1.29×
>   instead of ~1.01×. Router temperature is therefore load-bearing for balancing quality.
> - The histogram estimator has an **error floor below ~4k bins** set by rank discretization
>   (it lands on an integer rank; the interpolated quantile sits between ranks), not by bin
>   width. Refining bins past that point buys nothing.

### SiTU-GLU

SwiGLU multiplies two *unbounded* factors, so coincident large coordinates compound into
outliers — fatal with MXFP4 weights / MXFP8 activations. SiTU-GLU applies a smooth cap
`β·tanh(x/β)` to both branches: matches SwiGLU to first order near the origin, recovers it as
β→∞, and bounds output at `β₁β₂ = 100` (Appendix B). Unlike hard clamping it keeps gradients
alive past the cap — though [testing shows](../../tests/test_activations.py) `tanh` still
reaches exactly 1.0 in fp32, so the non-vanishing band is wide, not unconditional.

### Vision, trained from scratch

MoonViT-V2 (401M) is trained **from scratch with next-token prediction**, not initialized from
a contrastive encoder like SigLIP. Rationale is stability — SigLIP-initialized towers showed
persistent gradient spikes under joint optimization (Fig. 6). The finding that matters:
**from-scratch matched the SigLIP-initialized baseline on vision evals**, i.e. contrastive
pre-training is unnecessary as an initialization for multimodal LMs at scale.

### NoPE

No positional encoding anywhere; position information flows only through KDA's decay and
gating. Consequence: K3 extrapolates to 1M with **no RoPE rescaling, no YaRN, no frequency-base
retuning**. Long-context extension becomes a data-and-curriculum problem (8K→64K in
pre-training, 256K→1M in cooldown) rather than a surgery problem.

## Post-training

Three stages: SFT cold start → RL across three domains × three reasoning-effort levels (nine
expert models) → **Multi-Teacher On-Policy Distillation** consolidating all nine into one.

- **Partial rollouts** cut long-horizon tail latency: pause generation once a fraction λ of
  trajectories finish, resume next iteration from persisted sandbox state. Creates severe data
  staleness, absorbed by per-token regularization constraining updates to a local neighbourhood.
- **Reasoning-effort RL** via a per-problem token budget with reward −1 on overrun; anneal the
  budget multiplier to obtain low/high/max variants.
- **Reward hacking** is treated as a first-class adversary throughout — hacking detection for
  kernel tasks (CUDA graph replay, input caching, precision reduction), verbosity caps on the
  generative reward model, hidden verifiers paired with public ones in agentic environments.
- **QAT from SFT onward** (MXFP4/MXFP8), so rollout and training share one quantization scheme
  and there is no train–inference mismatch.

Environments are the real substrate: a *unified white-box RL environment* that composes agent
harnesses from configurable modules (so the model doesn't overfit one tool schema), a
self-evolving knowledge graph driving task synthesis, and **51,219,741 sandboxes** created
across training on microVM infrastructure with pause/resume — a paused sandbox consumes no
resources, which matters when up to 98% of a sandbox's lifetime is spent waiting on inference.

## Results, read honestly

**Strong:** BrowseComp 91.2, DeepSearchQA 95.0, ProgramBench 77.8, SWE-Marathon 42.0 (7 pts
ahead of Fable 5), OmniDocBench 91.1. First open model to top WebDev Arena (1,678 Elo).
Cost-efficiency is the standout — BrowseComp at $2.03/task, half GPT-5.6 Sol's and an order of
magnitude under Claude at max effort.

**Weak, per the report's own framing:** CritPt 23.4 and HLE-Full 43.5/56.0 — research-level
reasoning is explicitly named as the key remaining gap. Also trails on Agent Behavior Bench,
MIRA, 24/7 ClawBench, GDPval-AA v2, AA-Briefcase.

Worth noting the report evaluates *itself* under Kimi Code while giving competitors Claude
Code/Codex, and several numbers come with fallback/refusal caveats. Third-party (Artificial
Analysis #4/580, Vals #2/39) is the cleaner read.

**Cyber:** ~70% of human-reviewed findings genuine including 16 previously-unknown
vulnerabilities across six projects; 38.9% on an exploit suite vs GLM-5.2's 22.2%, but 0/41 on
CAISI's end-to-end exploit completion. Strong at discovery, weak at chaining.

## What transfers to my work

- **Bound a quantity to unlock a kernel.** The `g_min` decay floor is a numerical-range
  argument that converts a slow path into dense matmuls. Generalizes: when a kernel has a
  fallback path, ask what bound would eliminate it.
- **Jump to the minimizer instead of stepping toward it.** QB removes a hyperparameter by
  solving the dual exactly. Worth asking of any controller tuned with a step size.
- **Additive statistics survive sharding.** Histogram counts all-reduce into the exact global
  quantile. A pattern for any distributed statistic that currently needs a gather.
- **Remove an initialization rather than improve it.** From-scratch ViT matched SigLIP init
  *and* trained stably — the transfer question is where else a pre-trained component is
  cargo-cult.
- **Environments over algorithms.** The RL algorithm is a modest evolution of K2.5; the
  investment is in verifiable environments and sandbox infrastructure.

## Open questions

- Is the 2.5× scaling gain mostly AttnRes, mostly LatentMoE, or genuinely compositional? The
  report gives no per-component ablation — this is the thing I'd most want to measure.
- Does KDA's fixed state actually retain 1M-token detail, or does BrowseComp-style retrieval
  hide degradation that a needle-in-haystack at depth would expose?
- What does QB cost in *specialization*? Perfect balance is a constraint; forcing uniform load
  may suppress genuinely popular experts. Report shows balance, not specialization quality.

## Next

- [ ] Ablate AttnRes vs plain residual at nano scale — isolate the depth-axis contribution
- [ ] Measure KDA state saturation vs sequence length (fixed state, growing information)
- [ ] Compare QB against sign-updates at 896 experts, not 128
- [ ] Read [Kimi Linear](https://arxiv.org/abs/2510.26692) and [LatentMoE](https://arxiv.org/abs/2601.18089) — the two loads K3 leans on hardest
