"""Runner for state-compression: how much can we compress state before recall degrades?

Sweeps d_head to find the minimum state size that maintains competitive recall.

See configs/ablations/state-compression.yaml for the sweep spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import CorpusSpec
from archlab.model import ModelSpec


def model_spec_for(
    arm: dict[str, Any],
    cfg: dict[str, Any],
    d_head: int,
) -> ModelSpec:
    m = cfg["model"]
    return ModelSpec(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_head=d_head,
        d_hidden=m["d_hidden"],
        attention_pattern=arm["pattern"],
        depth_mixing=m.get("depth_mixing", "residual"),
        n_blocks=m.get("n_blocks", 1),
    )


def run(config_path: Path, out_dir: Path, device: str = "cpu") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    t = cfg["train"]
    sweep = cfg.get("sweep", {})
    d_heads = sweep.get("d_head", [64])

    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in cfg["arms"]:
        for d_head in d_heads:
            for seed in range(t.get("seeds", 1)):
                spec = model_spec_for(arm, cfg, d_head)
                corpus = CorpusSpec(
                    vocab_size=cfg["model"]["vocab_size"],
                    seq_len=t["seq_len"],
                    seed=seed,
                )
                train_spec = TrainSpec(
                    steps=t["steps"],
                    lr=t["lr"],
                    batch_size=t["batch_size"],
                )
                result = train_arm(spec, corpus, train_spec, device=device)
                row = {
                    "arm": arm["id"],
                    "d_head": d_head,
                    "state_size_kb": (d_head * d_head * cfg["model"]["n_heads"]) / 1024,
                    "seed": seed,
                    **result,
                }
                rows.append(row)
                with open(out_dir / "results.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")

    return rows
