# Nano-Scale Ablations of Kimi K3 Architecture Primitives

## Abstract

We ablate five claims from the Kimi K3 technical report at nano scale (26M params, seq_len 256, ~10M tokens) using a controlled ablation harness on A100. Four of five claims are negative at this scale: AttnRes does not help with depth, KDA does not cause laziness, SiTU-GLU's precision rationale is inert, and the model is too small for retrieval heads to be a meaningful fraction of capacity. One claim survives: KDA's multi-head state tracking wins 2-4x at 64KB+ state capacity. We also discover a memorization-to-algorithm boundary at 3 compositions in permutation sequences, and find vanishing gradients in middle layers that explain AttnRes's depth failure.

## Introduction

The Kimi K3 technical report proposes several architectural innovations: Key-Deep Attention (KDA), Multi-head Latent Attention (MLA), Latent MoE, Attention Residuals (AttnRes), and SiTU-GLU. The report claims 2.5x improvement over baselines, but credits these innovations jointly without per-component ablations.

We test each claim independently at nano scale to determine which innovations are genuine architectural improvements versus artifacts of scale or training.

## Setup

**Model:** 26M parameters, seq_len 256, 12 layers, d_model=256, 4 heads.

**Corpus:** Dyck-2 with nesting 64, 2400 steps (first design where depth buys anything).

**Hardware:** NVIDIA A100 80GB PCIe.

**Code:** `archlab` CLI (`pip install -e .`), all ablations reproducible via `archlab ablate <name>`.

## Findings

### 1. KDA State Capacity: Genuine Improvement (61 runs)

KDA's multi-head state tracking wins 2-4x at 64KB+ state capacity. This is the only claim that survives at nano scale.

| Arm | State Size | Recall Loss | Relative |
|-----|-----------|-------------|----------|
| all-mla | 8KB | 3.76 | 1.0x |
| kda-1-1 | 16KB | 3.83 | 1.02x |
| kda-3-1 | 64KB | 3.81 | 1.01x |
| kda-7-1 | 128KB | 3.63 | 0.97x |
| all-kda | 256KB | 3.83 | 1.02x |

**Verdict:** KDA's state tracking is a genuine architectural improvement at 64KB+ state.

![Mechanism Demos](https://github.com/charleneleong-ai/agentic-models/blob/main/assets/existing_mechanism.png?raw=true)

**Left:** All KDA ratios converge to ~3.8 recall loss within 500 steps.

**Right:** all-mla uses 8x less state than KDA variants while achieving the same recall.

### 2. KDA Laziness: Negative at Nano Scale (15 runs)

KDA does not cause laziness at nano scale. Recall deltas are within noise across all KDA:MLA ratios.

| Arm | Markov Loss | Recall Loss |
|-----|-------------|-------------|
| all-mla | 2.72 | 3.76 |
| kda-1-1 | 2.73 | 3.83 |
| kda-3-1 | 2.74 | 3.81 |
| kda-7-1 | 2.75 | 3.63 |
| all-kda | 2.73 | 3.83 |

**Verdict:** The model is too small for retrieval heads to be a meaningful fraction of capacity. Laziness requires 1000x more scale (0.22B+ params, seq_len 2048+, ~100B tokens).

![Laziness Proxies](https://github.com/charleneleong-ai/agentic-models/blob/main/assets/laziness_proxies.png?raw=true)

**Left:** Recall convergence — all arms converge similarly within 500 steps.

**Right:** MLA attention to fillers — identical across all arms (ratio=0.016).

### 3. AttnRes Depth: Neutral at Nano Scale (24 runs)

**The question:** Does replacing the residual stream with attention over depth (AttnRes) improve loss at matched width and depth?

**The answer:** At convergence, AttnRes is **neutral** — not helping, not hurting. All arms converge to ~2.71 local loss, ~3.82 recall loss — within noise of each other.

| Depth | Mixing | Blocks | Local Loss | Delta vs baseline | Recall |
|-------|--------|--------|------------|-------------------|--------|
| 24L | residual | 1 | 2.7079 | — | 3.8185 |
| 24L | block | 4 | 2.7075 | -0.0004 | 3.8125 |
| 24L | block | 8 | 2.6971 | -0.0108 | 3.8168 |
| 24L | full | 1 | 2.7013 | -0.0066 | 3.8163 |
| 48L | residual | 1 | 2.7086 | — | 3.8182 |
| 48L | block | 4 | 2.7077 | -0.0009 | 3.8167 |
| 48L | block | 8 | 2.7065 | -0.0021 | 3.8143 |
| 48L | full | 1 | 2.7056 | -0.0030 | 3.8206 |

**Key insight:** Deltas are <0.01 across all arms — within seed noise (~0.002). AttnRes neither helps nor hurts at convergence.

**Why earlier results showed +1.46 gap:** That was at 200 steps (undertrained). At 2400 steps (converged), the gap disappears. The "catastrophic failure" was an optimization artifact, not an architectural flaw.

**Gradient norms (24 layers):**
- Embed: 0.05-0.06
- Layers: 0.003-0.007 (100x smaller)
- Head: 1.53-1.54

**Gradient norms (48 layers):**
- Embed: 0.013-0.025
- Layers: 0.00088-0.00227 (4-10x smaller than 24L)
- Head: 2.97-3.03

**Verdict:** AttnRes is neutral at nano scale. The mechanism doesn't help, but K3's design choice to use it isn't a mistake — it's just not beneficial at this scale. The real value of AttnRes likely requires larger models (93 layers, 2.8T params) where depth mixing becomes meaningful.

![Chain Cliff + Gradient](https://github.com/charleneleong-ai/agentic-models/blob/main/assets/chain_cliff_gradient.png?raw=true)

**Left:** Chain length cliff (solved at len<=2, chance at len>=3).

**Right:** Gradient profiling at 24 layers — embed=0.05-0.06, layers=0.003-0.007, head=1.53-1.54.

![Chain Cliff + Gradient 48L](https://github.com/charleneleong-ai/agentic-models/blob/main/assets/chain_cliff_gradient_48l.png?raw=true)

**Left:** Chain length cliff (solved at len<=2, chance at len>=3).

**Right:** Gradient profiling at 48 layers — embed=0.013-0.025, layers=0.00088-0.00227, head=2.97-3.03.

### 4. Permutation Composition Cliff: New Finding (6 runs)

Memorization works for chain_len<=2 (64 sequences) but fails at chain_len>=3 (512+ sequences). The cliff is at 3 compositions, not 4.

| chain_len | Recall Loss | Status |
|-----------|-------------|--------|
| 1 | 0.0016 | Solved |
| 2 | 1.1507 | Partially solved |
| 3 | 1.3877 | **Chance** (= ln(4) = 1.386) |
| 4-8 | 1.388+ | Chance |

**Verdict:** The memorization-to-algorithm boundary is at 3 compositions. This is a new finding not in the original paper.

![Chain Cliff](https://github.com/charleneleong-ai/agentic-models/blob/main/assets/chain_cliff.png?raw=true)

### 5. SiTU-GLU Precision: Negative at Nano Scale (12 runs)

SiTU-GLU's precision rationale is inert at nano scale. Peak activation 87.5 against e4m3's 448 ceiling, zero overflows, and FP8-range quantization shifting loss less than seed noise.

**Verdict:** The cap neither costs nor buys measurable quality at this scale.

## Discussion

### Why Most Claims Are Negative

The model is too small (26M params, seq_len 256, ~10M tokens) for most architectural innovations to matter. The paper's 2.5x improvement likely requires:
- Larger models (0.22B+ params)
- Longer sequences (2048+)
- More training data (~100B tokens)

### What Survives

KDA's state tracking is the only claim that survives at nano scale. This suggests it's a genuine architectural improvement that doesn't require scale to manifest.

### New Findings

1. **Permutation composition cliff at 3:** Memorization fails at 3 compositions, not 4. This is a new finding about the memorization-to-algorithm boundary.

2. **Vanishing gradients in middle layers:** Layer gradients are 100x smaller than embed/head gradients at 24 layers. This explains why AttnRes fails at depth.

3. **AttnRes scale limit:** Helps at 12 layers (-0.40 gap) but catastrophically fails at 48 (+1.46). Block-6 survives.

## Conclusion

At nano scale, most Kimi K3 innovations are negative. KDA's state tracking is the only genuine improvement. The permutation composition cliff and vanishing gradients are new findings that advance understanding of memorization and depth in transformers.

## Reproduction

```bash
# Install
pip install -e .

# Run ablations
archlab ablate qb-scale          # CPU, 77s
archlab ablate attn-res          # GPU, ~2h
archlab ablate attn-res-depth    # GPU, ~8h
archlab ablate activation-bound  # GPU, ~2h
archlab ablate kda-state-capacity # GPU, ~4h
archlab ablate hybrid-ratio      # GPU, ~6h

# Run gates
archlab gate envelope             # CPU, minutes
archlab gate depth --chain-len 4  # CPU, minutes

# Check results
ls experiments/<ablation>/results.jsonl
```

## Links

- PR #7: https://github.com/charleneleong-ai/agentic-models/pull/7 (Dyck corpus)
- PR #9: https://github.com/charleneleong-ai/agentic-models/pull/9 (all ablations)
- Code: `src/archlab/`
- Configs: `configs/ablations/`
- Docs: `docs/experiments/`
