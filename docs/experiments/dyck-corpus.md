# dyck-corpus — the first corpus where depth buys something

**Tooling:** [`archlab gate dyck|dyck-nesting`](../../src/archlab/corpus_gate.py) ·
**Corpus:** [`DyckSpec`](../../src/archlab/data.py) ·
**Run:** 2026-08-01, A100, ~1 h across three diagnostics

## Why a third corpus

[`corpus-gate.md`](corpus-gate.md) ended blocked. Two designs had failed the same way, and the
shared cause was not a parameter but a *shape*: difficulty jumped in a cliff, so there was no
regime that was hard, learnable and depth-sensitive at once.

| corpus | easy end | hard end | anything between |
|---|---|---|---|
| recall | Markov filler, learned in ~2 layers | planted recall, never learned | no |
| composition | chain_len <= 2, memorized | chain_len >= 4, never learned | no |
| **Dyck** | nesting 8, solved | nesting 64, at chance | **yes** |

Dyck was chosen for that specific property, not because bracket-matching is intrinsically
interesting. Nesting depth is a **continuous** dial, and partial credit exists — getting the
inner brackets right is worth something even when the outer ones are lost — so the loss should
move smoothly instead of sitting at chance until it collapses.

## The corpus

```
<mark> ( [ { ... } ] )
       \____ d opens ____/\____ d closes ____/
```

Opens are random; the closes are then **fully determined** — close *j* must match open
*(d-1-j)*. So the target is perfectly predictable in principle, and predicting it requires
reading the stack in *reverse*, which is the operation that costs depth. A model tracking only
recent context closes the innermost pairs and guesses the outermost, and
[`dyck_close_positions`](../../src/archlab/data.py) exposes that per-nesting-level breakdown
directly.

Tests assert the closes really are the reverse of the opens, that every nesting level is
represented, and that the outermost bracket type is not guessable from its marginal
distribution.

## Result 1: depth helps — a first

Nesting depth 8, 600 steps, residual baseline:

| layers | 6 | 12 | 24 | 48 |
|---|---:|---:|---:|---:|
| close loss | 0.0686 | 0.0490 | 0.0520 | **0.0197** |

**A 71% reduction from 6 to 48 layers.** Neither prior corpus produced any improvement at all —
the recall corpus moved +0.006 in the *wrong* direction across the same range.

## Result 2: difficulty degrades smoothly

Nesting depth swept at 6 layers, chance = ln(8) = 2.079:

| nesting | seq_len | close loss | state |
|---:|---:|---:|---|
| 8 | 256 | 0.0605 | solved |
| **16** | 256 | **0.9274** | **partial — the usable band** |
| 32 | 320 | 1.7103 | mostly failing |
| 64 | 576 | 2.0238 | at chance |

This is the property the previous corpora lacked. The composition corpus jumped 0.08 -> 2.18
between chain lengths 2 and 4 with nothing in between; Dyck gives four distinct difficulty
levels, so an operating point can be *chosen* rather than hoped for.

**Nesting 16 is that point:** a 6-layer baseline at 0.93 leaves ~0.93 of headroom to measure
into, with chance still a full 1.15 further away.

## A correction to the gate itself

The first Dyck run *failed* its gate — and the gate was wrong, not the corpus. I had used an
absolute threshold (`deep < shallow - 0.05`) on a task whose entire range is 0-0.07, so a 71%
reduction missed the bar by 0.001.

Replaced with a relative criterion, plus a third verdict the binary version could not express:

| verdict | meaning | fix |
|---|---|---|
| `GATE FAILS` | depth buys < 25% | change the corpus |
| `GATE PARTIAL` | depth helps, but the shallow baseline is already < 0.2 | raise the difficulty |
| `GATE PASSES` | depth helps *and* there is headroom to measure it | run the sweep |

Nesting 8 is `GATE PARTIAL`. That is a genuinely different situation from the two earlier
failures, and the binary gate would have lumped them together and sent me to design corpus #4
instead of turning one dial.

## Also fixed

- **Corpus sizing ignored a kernel constraint.** KDA requires `seq_len % chunk_size == 0`;
  `4 * (2*32+1) = 260` is not divisible by 64 and crashed the run. Sequence length now rounds
  up, and all four nesting settings are checked against both that constraint and the
  minimum-cells requirement before launching.
- **A CLI bug that only appeared under `python -m`.** The `gate` command was defined *after*
  `if __name__ == "__main__": app()`, so running as a module invoked the app before the command
  existed. It worked through the console entry point, which imports the module first — and the
  remote box uses `-m`. Both paths are now exercised.

## What this unblocks

The depth sweep that has been blocked since [`attn-res-depth`](attn-res-depth.md). Re-running
the AttnRes arms on Dyck at nesting 16 finally asks the original question — *how much of K3's
2.5x is the depth axis* — on a corpus where depth demonstrably matters.

The caveat that survives: this is bracket matching, not language. Depth helping here does not
mean AttnRes's benefit at 93 layers on real text will look the same. What it does give is a
setting where the *absence* of a depth effect would be informative, which is precisely what
`attn-res-depth` could not offer.

## Next

1. Confirm depth helps at nesting 16 with real headroom (`archlab gate dyck --chain-len 16`).
2. Re-run the AttnRes depth sweep on Dyck at that operating point.
3. `kda-state-capacity` may also be unblocked — a fixed recurrent state has to hold the bracket
   stack, and nesting depth is now a direct dial on how much state that requires.
