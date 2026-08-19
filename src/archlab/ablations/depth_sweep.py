"""Runner for attn-res-depth: is the AttnRes deficit a depth artifact?

Crosses the depth-mixing arms with `sweep.n_layers`. The reported metric is the *gap* to the
residual baseline at matched depth and seed, because absolute loss falls with depth for every
arm — comparing losses across depths would measure capacity, not the mechanism.

Runs the baseline first at each depth so every other arm can be differenced against it
immediately, and so a crash mid-sweep still leaves interpretable rows behind.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import build_corpus
from archlab.model import ModelSpec

BASELINE = "residual"

# Which loss the gap is measured on. The recall corpus put the depth-sensitive signal in
# `val_local_loss`; Dyck puts it in `val_recall_loss` (the close brackets) and leaves local as
# depth-insensitive Markov filler. Getting this wrong measures the wrong quantity, so it is
# read from the config rather than assumed — and both gaps are recorded either way.
GAP_METRIC = {"val_local_loss": "gap_local", "val_recall_loss": "gap_close"}


def model_spec_for(
    arm: dict[str, Any], cfg: dict[str, Any], n_layers: int, vocab_size: int
) -> ModelSpec:
    m = cfg["model"]
    block_size = arm.get("block_size", 6)
    return ModelSpec(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_layers=n_layers,
        n_heads=m["n_heads"],
        d_head=m["d_head"],
        d_hidden=m["d_hidden"],
        attention_pattern=[m.get("attention", "kda")],
        depth_mixing=arm["depth_mixing"],
        n_blocks=max(1, n_layers // block_size),
        ffn=m.get("ffn", "situ_glu"),
        chunk_size=m.get("chunk_size", 32),
    )


def ordered_arms(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baseline first — everything else is differenced against it."""
    return sorted(arms, key=lambda a: a["id"] != BASELINE)


def run(config_path: Path, out_dir: Path, device: str = "cpu") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    t = cfg["train"]
    corpus = build_corpus(cfg["corpus"])
    primary = cfg.get("metrics", {}).get("gap_on", "val_local_loss")
    if primary not in GAP_METRIC:
        raise ValueError(f"metrics.gap_on must be one of {sorted(GAP_METRIC)}, got {primary!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    sink = (out_dir / "results.jsonl").open("w")

    for n_layers in cfg["sweep"]["n_layers"]:
        for seed in t["seeds"]:
            baseline: dict[str, float] = {}
            for arm in ordered_arms(cfg["arms"]):
                spec = model_spec_for(arm, cfg, n_layers, corpus.vocab_size)
                metrics = train_arm(
                    spec,
                    TrainSpec(
                        steps=t["steps"],
                        batch_size=t["batch_size"],
                        lr=t["lr"],
                        warmup_frac=t.get("warmup_frac", 0.02),
                        weight_decay=t.get("weight_decay", 0.1),
                        seed=seed,
                        device=device,
                    ),
                    corpus,
                )
                if arm["id"] == BASELINE:
                    baseline = {k: metrics[k] for k in ("val_local_loss", "val_recall_loss")}
                gaps = {
                    "gap_local": round(metrics["val_local_loss"] - baseline["val_local_loss"], 4),
                    "gap_close": round(metrics["val_recall_loss"] - baseline["val_recall_loss"], 4),
                }
                gap = gaps[GAP_METRIC[primary]]

                row = {
                    "arm": arm["id"],
                    "n_layers": n_layers,
                    "n_blocks": spec.n_blocks,
                    "seed": seed,
                    "gap_vs_residual": gap,
                    "gap_metric": primary,
                    **gaps,
                    **metrics,
                }
                results.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()  # a killed sweep must leave usable rows behind
                print(
                    f"L={n_layers:>2} seed={seed} {arm['id']:>16}  "
                    f"local={row['val_local_loss']:.4f} gap={gap:+.4f} "
                    f"recall={row['val_recall_loss']:.4f} sources={row['peak_live_sources']}"
                )

    sink.close()
    return results
