# docs

Three kinds of writing live here, split by what changes them.

| Directory | Holds | Rewritten when |
|---|---|---|
| [`primitives/`](primitives/) | How one mechanism evolved — the lineage behind a module in `archlab` | A new paper extends the lineage |
| [`comparisons/`](comparisons/) | Cross-architecture tables — what each design pays and buys | A new study adds a column |
| [`experiments/`](experiments/) | One plan + result per ablation, mirroring `configs/ablations/*.yaml` | An ablation runs |

Per-model close readings do **not** live here — they belong to their study
([`studies/kimi-k3/`](../studies/kimi-k3/)), next to that model's papers. The rule: if it is
about *one* model, it goes in `studies/`; if it is about a *mechanism across* models, it goes
in `docs/`.

## Current pages

**Primitives**
- [`moe-load-balancing.md`](primitives/moe-load-balancing.md) — auxiliary losses → expert
  choice → BASE/BIP → loss-free bias → Quantile Balancing, and why the dual's coordinate
  minimizers turn out to be quantiles.

**Comparisons**
- [`attention-mechanisms.md`](comparisons/attention-mechanisms.md) — state-per-token as the
  axis that decides everything at 1M context; the delta-rule lineage; why NoPE makes context
  extension a data problem.

**Experiments**
- [`experiments/README.md`](experiments/README.md) — the five planned ablations, what each
  isolates, and the config contract.

## Writing rules

Same hygiene as a PR body. No enumeration the code already shows; link every symbol mentioned
to its source; drop sections that exist only out of habit. A doc that restates a docstring is
worse than no doc — put the explanation in the docstring and link to it.

Findings that contradict or extend a paper get recorded explicitly, with the test that
produced them. Two so far, both in `moe-load-balancing.md`: Quantile Balancing degrades under
score ties, and the histogram estimator has a rank-discretization error floor.
