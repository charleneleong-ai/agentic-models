"""Runner for attn-res-scale: does the arm *ranking* survive a 4x width increase?

[`attn-res`](attn_res.py) found a plain residual stream beating attention-over-depth at 12
layers and 128 width. That is a claim about learned model quality — the class the
validity-threat literature says may not transfer, since proxy rankings can fail even between
125M and 1B. This grows the width and checks whether the ordering holds.

**The output is the ranking, not the loss.** Losses must fall as width grows; a ladder that
compared them across rungs would be measuring capacity. What transfers or fails to transfer is
the order of the arms, so that is what `report_ranking` scores — and only over pairs whose gap
exceeds the seed spread beneath it, since an ordering the noise can flip is not an ordering.

This is only runnable because muP is verified here. Under SP the optimal learning rate drifts
with width, so a fixed-LR ladder trains its wider rungs wrong and a rank flip is ambiguous
between "the mechanism does not transfer" and "the optimizer was mistuned". muP holds the
optimum in place ([`mup_check`](../mup_check.py)), and the ladder inherits that.

The base learning rate is nonetheless tuned once, at the base width. muP transfers across
*width*, and this ladder holds depth at 12 while the muP verification ran at depth 4 — so the
optimum is established at the ladder's own depth and muP carries it outward. That is muTransfer
used as intended rather than assumed past its evidence. The search runs on the baseline arm
only, because tuning per-arm or per-width would discard the very property being relied on.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import CorpusSpec
from archlab.model import ModelSpec


def model_spec_for(
    arm: dict[str, Any], cfg: dict[str, Any], width: int, vocab_size: int
) -> ModelSpec:
    """Width is the swept axis; every other shape derives from it.

    `n_heads` derives as `width // head_dim` so heads scale with width and per-head dimension
    stays fixed — the convention muP's width multiplier assumes. Hard-coding `n_heads` instead
    would change what "wider" means partway up the ladder.
    """
    m = cfg["model"]
    return ModelSpec(
        vocab_size=vocab_size,
        d_model=width,
        n_layers=m["n_layers"],
        n_heads=max(1, width // m["head_dim"]),
        d_head=m["head_dim"],
        d_hidden=m["hidden_mult"] * width,
        attention_pattern=[m.get("attention", "kda")],
        depth_mixing=arm["depth_mixing"],
        n_blocks=arm.get("n_blocks", 4),
        ffn=m.get("ffn", "situ_glu"),
        chunk_size=m.get("chunk_size", 64),
        mup=m.get("mup", True),
        base_width=cfg["base_width"],
    )


def train_spec_for(cfg: dict[str, Any], lr: float, seed: int, device: str) -> TrainSpec:
    t = cfg["train"]
    return TrainSpec(
        steps=t["steps"],
        batch_size=t["batch_size"],
        lr=lr,
        warmup_frac=t.get("warmup_frac", 0.02),
        weight_decay=t.get("weight_decay", 0.1),
        seed=seed,
        device=device,
    )


def search_base_lr(cfg: dict[str, Any], corpus: CorpusSpec, device: str) -> float:
    """Tune the learning rate once, at the base width, on the baseline arm."""
    baseline = cfg["arms"][0]
    width, seed = cfg["base_width"], cfg["train"]["seeds"][0]
    print(f"=== base-LR search: {baseline['id']} at width {width} ===", flush=True)

    losses = {}
    for lr in cfg["lr_search"]:
        spec = model_spec_for(baseline, cfg, width, corpus.vocab_size)
        metrics = train_arm(spec, train_spec_for(cfg, lr, seed, device), corpus)
        losses[lr] = metrics[cfg["metrics"]["primary"]]
        print(f"  lr={lr:<9.0e} {cfg['metrics']['primary']}={losses[lr]:.4f}", flush=True)

    best = min(losses, key=lambda k: losses[k])
    edge = best in (cfg["lr_search"][0], cfg["lr_search"][-1])
    print(f"  -> base lr {best:.0e}" + ("  [WARNING: at grid edge]" if edge else ""), flush=True)
    return best


def run(config_path: Path, out_dir: Path, device: str = "cpu") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    c = cfg["corpus"]
    corpus = CorpusSpec(
        vocab_size=c["vocab_size"],
        seq_len=c["seq_len"],
        n_pairs=c["n_pairs"],
        key_vocab=c["key_vocab"],
    )
    primary = cfg["metrics"]["primary"]
    lr = search_base_lr(cfg, corpus, device)

    results: list[dict[str, Any]] = []
    for width in cfg["widths"]:
        print(f"\n=== width {width} (lr {lr:.0e}, muP) ===", flush=True)
        for seed in cfg["train"]["seeds"]:
            for arm in cfg["arms"]:
                spec = model_spec_for(arm, cfg, width, corpus.vocab_size)
                metrics = train_arm(spec, train_spec_for(cfg, lr, seed, device), corpus)
                results.append(
                    {"width": width, "arm": arm["id"], "seed": seed, "lr": lr, **metrics}
                )
                print(
                    f"{arm['id']:>16} seed={seed}  {primary}={metrics[primary]:.4f} "
                    f"params={metrics['n_params']:,}",
                    flush=True,
                )

    report_ranking(results, cfg, primary)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    return results


def resolvable(gap: float, spread_a: float, spread_b: float) -> bool:
    """Is a gap between two arms larger than the seed noise underneath it?

    Without this the ladder reports orderings it cannot actually see. On the first run the two
    AttnRes arms swapped places between widths and the naive all-or-nothing check called it a
    rank flip — but their gap was 0.0064 to 0.0127 against seed spreads of 0.0081 to 0.0283, so
    every one of those orderings was noise. A ladder that reports unresolvable swaps as findings
    manufactures false negatives at exactly the rate its arms are close together.
    """
    return gap > max(spread_a, spread_b)


def report_ranking(results: list[dict[str, Any]], cfg: dict[str, Any], primary: str) -> None:
    """Per-width orderings, scored only over pairs the seed noise can actually separate."""
    ids = [arm["id"] for arm in cfg["arms"]]
    order: dict[int, list[str]] = {}
    mean: dict[tuple[int, str], float] = {}
    spread: dict[tuple[int, str], float] = {}

    for width in cfg["widths"]:
        for arm in ids:
            vals = [r[primary] for r in results if r["width"] == width and r["arm"] == arm]
            mean[width, arm] = sum(vals) / len(vals)
            spread[width, arm] = max(vals) - min(vals)
        order[width] = sorted(ids, key=lambda a: mean[width, a])
        ranked = "  <  ".join(
            f"{a} ({mean[width, a]:.4f} +-{spread[width, a]:.4f})" for a in order[width]
        )
        print(f"\nwidth {width:>4}: {ranked}", flush=True)

    print("\n=== which pairwise orderings are resolvable at all? ===", flush=True)
    stable: set[tuple[str, str]] = set()
    for a, b in combinations(ids, 2):
        verdicts = []
        for width in cfg["widths"]:
            gap = abs(mean[width, a] - mean[width, b])
            ok = resolvable(gap, spread[width, a], spread[width, b])
            winner = a if mean[width, a] < mean[width, b] else b
            verdicts.append((width, ok, winner, gap))
        seen = {w for _, ok, w, _ in verdicts if ok}
        n_ok = sum(ok for _, ok, _, _ in verdicts)
        detail = "  ".join(
            f"{w}:{'*' if not ok else ''}{win.replace('attnres-', '')}"
            for w, ok, win, _ in verdicts
        )
        if n_ok == 0:
            note = "never resolvable — gaps sit under the seed noise"
        elif len(seen) == 1:
            note = f"consistent: {next(iter(seen))} wins wherever it is measurable"
            stable.add((a, b))
        else:
            note = "GENUINE FLIP — different winners, both resolvable"
        print(f"  {a:>16} vs {b:<16} {detail:<34} {note}", flush=True)

    print("\n(* marks a width where the gap is smaller than the seed spread.)", flush=True)

    baseline = ids[0]
    holds = all(
        (baseline, other) in stable or (other, baseline) in stable
        for other in ids
        if other != baseline
    ) and all(order[w][0] == baseline for w in cfg["widths"])

    if holds:
        print(f"\nPRIMARY CLAIM HOLDS — '{baseline}' wins at every width, resolvably.")
        gaps = [
            min(mean[w, o] for o in ids if o != baseline) - mean[w, baseline] for w in cfg["widths"]
        ]
        trend = "widens" if gaps[-1] > gaps[0] else "narrows"
        print(
            f"Its margin over the best alternative {trend}: "
            + " -> ".join(f"{g:+.4f}" for g in gaps)
        )
    else:
        print(
            f"\nPRIMARY CLAIM DOES NOT HOLD — '{baseline}' does not win resolvably at every width."
        )
    print("Losses are not comparable across widths; read the orderings, not the levels.")
