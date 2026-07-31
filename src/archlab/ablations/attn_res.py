"""Runner for the attn-res ablation: how much of K3's 2.5x is the depth axis alone?

Every arm holds width, depth, attention core and activation fixed; only the depth mixer is
swapped. `ResidualStack`, `FullAttnRes` and `BlockAttnRes` share one interface precisely so
that swap is the *only* difference between arms.

Parameter counts differ slightly — AttnRes adds one learnable pseudo-query vector per layer,
which is `n_layers * d_model` parameters, well under 1% here. Recorded per arm so the reader
can check the comparison is not just buying capacity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import CorpusSpec
from archlab.model import ModelSpec


def model_spec_for(arm: dict[str, Any], cfg: dict[str, Any], vocab_size: int) -> ModelSpec:
    m = cfg["model"]
    return ModelSpec(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_head=m["d_head"],
        d_hidden=m["d_hidden"],
        attention_pattern=[m.get("attention", "kda")],
        depth_mixing=arm["depth_mixing"],
        n_blocks=arm.get("n_blocks", 4),
        ffn=m.get("ffn", "situ_glu"),
        chunk_size=m.get("chunk_size", 32),
    )


def run(config_path: Path, out_dir: Path, device: str = "cpu") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    t, c = cfg["train"], cfg["corpus"]
    corpus = CorpusSpec(
        vocab_size=c["vocab_size"],
        seq_len=c["seq_len"],
        n_pairs=c["n_pairs"],
        key_vocab=c["key_vocab"],
    )

    results = []
    for seed in t["seeds"]:
        for arm in cfg["arms"]:
            spec = model_spec_for(arm, cfg, corpus.vocab_size)
            train_spec = TrainSpec(
                steps=t["steps"],
                batch_size=t["batch_size"],
                lr=t["lr"],
                warmup_frac=t.get("warmup_frac", 0.02),
                weight_decay=t.get("weight_decay", 0.1),
                seed=seed,
                device=device,
            )
            metrics = train_arm(spec, train_spec, corpus)
            row = {"arm": arm["id"], "seed": seed, **metrics}
            results.append(row)
            print(
                f"{arm['id']:>16} seed={seed}  local={row['val_local_loss']:.4f} "
                f"recall={row['val_recall_loss']:.4f} params={row['n_params']:,} "
                f"sources={row['peak_live_sources']}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    return results
