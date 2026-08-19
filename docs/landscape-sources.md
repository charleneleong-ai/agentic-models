# Sources — the 2026 landscape sweep

[`landscape-2026.md`](landscape-2026.md) was written from web search. Search results are not
persisted anywhere: they lived in one conversation and are gone. This file exists so the claims
in that document can be checked without re-running the sweep and hoping for the same hits.

**Verified 2026-08-02.** Every arXiv ID below resolves, and every title was fetched and matched
against how the sweep cites it. Fabricated or mis-attributed identifiers are a real failure mode
for search-derived writing, and that document is load-bearing — it is where two of this repo's
claims were withdrawn.

Re-run the check:

```bash
for id in 2606.15378 2605.15155 2605.24220 2001.11396 2603.21383 2606.01779 2606.05029 2606.25447; do
  curl -s -L "https://arxiv.org/abs/$id" \
    | grep -o '<meta name="citation_title" content="[^"]*"' | head -1
done
```

## Cited, verified

| arXiv | Title (fetched, not recalled) | Used in the sweep for |
|---|---|---|
| [2606.15378](https://arxiv.org/abs/2606.15378) | Rethinking the Role of Efficient Attention in Hybrid Architectures | The published version of the unrun `hybrid-ratio` ablation; efficient attention changes how *fast* long-context capability emerges, not the ceiling |
| [2606.05029](https://arxiv.org/abs/2606.05029) | Validity Threats **for** Foundation Model Research | The scale-transfer threat the whole ladder exists to answer |
| [2001.11396](https://arxiv.org/abs/2001.11396) | Non-Determinism in TensorFlow ResNets | GPU nondeterminism as a known result, making this repo's determinism finding a replication |
| [2603.21383](https://arxiv.org/abs/2603.21383) | PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost | Cascading-failure mitigation in agentic RL |
| [2605.15155](https://arxiv.org/abs/2605.15155) | Self-Distilled Agentic Reinforcement Learning | Dense token-level teacher signal vs K3's MOPD |
| [2605.24220](https://arxiv.org/abs/2605.24220) | Polar: Agentic RL on Any Harness at Scale | Harness co-design via API-traffic interception |
| [2606.01779](https://arxiv.org/abs/2606.01779) | HarnessForge: Joint Harness and Policy Evolution for Adaptive Agent Systems | Joint harness/policy evolution |
| [2606.25447](https://arxiv.org/abs/2606.25447) | The Interplay of Harness Design and Post-Training in LLM Agents | The direct treatment of optimising model *to* harness |

The title as cited was corrected in one place: the sweep wrote "Validity Threats *in* Foundation
Model Research"; it is "*for*".

## Cited but not verifiable from this file

These carry claims in `landscape-2026.md` with no archived source, and should be treated as
unverified until someone checks them against a primary document:

- **Nemotron 3 and Arcee Trinity configurations** — parameter counts, active fractions, and the
  hybrid-attention ratios in §1's table came from secondary reporting, not technical reports.
  The sweep's own caveat says as much. The *shape* of the convergence argument does not depend
  on the exact figures; any individual number does.
- **The MoE super-high-sparsity balancing results** in §4 — attributed to "the MoE literature"
  and "at least one 2026 study" without a specific citation. This is the weakest link in the
  document, and it matters because it is the passage that appeared to corroborate the
  now-withdrawn tie-degradation claim. Corroboration that cannot be located is not corroboration.
- **Blog-sourced claims**, marked inline where they occur.

## What is not saved, and why that is a limitation

The queries, the result sets, and the sources read and discarded are all gone. So the sweep's
*recall* is unknown and unrecoverable — there is no way to tell what it missed, or whether a
differently-phrased query would have surfaced work that contradicts it. Every claim in
`landscape-2026.md` should be read as "this exists" rather than "this is the state of the art".

The cheap fix, if the sweep is repeated: write the queries into this file as they are run.
