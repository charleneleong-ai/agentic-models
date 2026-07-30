# Attention mechanisms — what each one pays and what it buys

The axis that matters at 1M context is **what grows with sequence length**. Everything else
follows from that.

| | State per token | Global? | Position encoding | Where |
|---|---|---|---|---|
| MHA | 2·H·d_head | yes | RoPE / ALiBi | baseline |
| GQA / MQA | 2·G·d_head | yes | RoPE | Llama, Mistral |
| MLA | d_latent (shared across heads) | yes | RoPE (decoupled) | DeepSeek-V2+ |
| MLA (K3) | d_latent | yes | **none (NoPE)** | [`mla.py`](../../src/archlab/attention/mla.py) |
| Linear / DeltaNet | **0** — fixed state d_k×d_v | no | implicit via decay | DeltaNet, Mamba-2 |
| KDA | **0** — fixed state d_k×d_v | no | implicit via decay | [`kda.py`](../../src/archlab/attention/kda.py) |

Linear attention's "0 per token" is the whole game: a fixed-size recurrent state means memory
is **constant** in sequence length. The cost is that mixing is lossy — a fixed state cannot
hold arbitrary detail — so pure linear attention degrades on tasks needing exact recall.

Hence **hybrids**. K3 runs 3 KDA layers per 1 Gated MLA layer: KDA carries the sequence
cheaply, MLA periodically restores unrestricted token-to-token interaction. The ratio is the
tunable, and 3:1 is K3's answer. `test_mla.py` works the arithmetic: at 1M tokens the latent
cache is ~40× smaller than full MHA, and KDA's fixed state is ~100× smaller again.

## The delta-rule lineage

Each step adds one mechanism to the recurrence:

```
Linear attention   S_t = S_{t-1} + k v^T                    pure accumulation, never forgets
DeltaNet           S_t = (I - b k k^T) S_{t-1} + b k v^T    delta rule: overwrite, not just add
Gated DeltaNet     + scalar decay                           can forget
Mamba-2            + structured state-space duality
Kimi Linear        + channel-wise decay Diag(a)             forget per channel, not per token
KDA               + lower-bounded decay, full-rank gate     the same, made kernel-friendly
```

The delta rule is the pivotal step: `(I - βkkᵀ)` *removes* the component of the state along
`k` before writing the new value, so repeated keys overwrite instead of superposing. That is
what makes a fixed state usable for retrieval at all.

KDA's own contribution is less about expressiveness than about **execution** — see
[`kda.py`](../../src/archlab/attention/kda.py) on why bounding the decay from below converts
diagonal tiles from explicit position-pair math into dense matmuls.

## NoPE and why it matters for context extension

Every explicit positional scheme becomes a liability when the window grows: RoPE needs
frequency-base retuning or YaRN interpolation, and both are surgery with their own failure
modes. K3 applies **no positional encoding at all** — position reaches the model only through
KDA's decay and gating.

The consequence is structural: extending 8K → 1M is a *data and curriculum* problem
(upsample long documents, synthesize tasks whose answers require attending across the full
window) rather than a modification to the attention math. This only works because the hybrid
guarantees a position-sensitive path exists — NoPE on a pure-softmax stack would be
permutation-invariant.

## Open questions

- What is the actual 3:1 tradeoff curve? Nothing published sweeps it.
- Does KDA's fixed state saturate at some effective context, with retrieval benchmarks hiding
  it? A needle-at-depth probe would separate "1M window" from "1M of usable state".
