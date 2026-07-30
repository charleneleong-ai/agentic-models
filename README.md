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
uv run pytest          # 59 tests, CPU, ~5s
```

Runs anywhere on CPU. A GPU box is for the ablations in `configs/ablations/`, not for the suite.

## What the tests are for

They are the check on whether a mechanism was understood, so they assert claims rather than
shapes. A few that earned their keep:

- **KDA chunkwise ≡ serial recurrence** across chunk sizes 1–64 — the UT-transform algebra is
  only trustworthy if it reproduces Eq. 1 exactly.
- **Bounded decay keeps `1/Γ` finite** where the unbounded `-Softplus` baseline provably
  overflows — the numerical argument that unlocks dense Tensor Core tiles.
- **Quantile Balancing beats sign-updates** at a matched step budget, and its exactness
  **degrades under score ties** — a caveat the report leaves implicit.
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

## Open questions, specified

Five ablations are specified in [`configs/ablations/`](configs/ablations/), each targeting
something the K3 report leaves unanswered — the per-component split of the 2.5x scaling claim,
the unswept KDA:MLA ratio, where a fixed recurrent state saturates, what capping activations
costs, and whether QB's edge widens with expert count. Configs only so far; the runner is not
built. [`docs/experiments/`](docs/experiments/) has the contract and the reasoning.

## Related

[`small-smart-models`](../small-smart-models) asks the complementary question — how far these
models compress. Expert-activation structure understood here feeds the expert-dropping and
importance-aware quantization studies there.
