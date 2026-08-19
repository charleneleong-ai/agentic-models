"""Runner for hybrid-ratio: does KDA delay retrieval head formation like SWA?

Sweeps attention patterns (all-MLA, kda-1-1, kda-3-1, kda-7-1, all-KDA) and measures
retrieval accuracy at various training stages. The laziness hypothesis: more KDA layers
→ slower retrieval head formation → worse recall at early training, possibly recovering
at convergence.

See configs/ablations/hybrid-ratio.yaml for the sweep spec.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import build_corpus
from archlab.model import ModelSpec


def model_spec_for(
    arm: dict[str, Any],
    cfg: dict[str, Any],
    vocab_size: int,
) -> ModelSpec:
    m = cfg["model"]
    pattern = arm["pattern"]
    return ModelSpec(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_head=m["d_head"],
        d_hidden=m["d_hidden"],
        attention_pattern=pattern,
        depth_mixing=m.get("depth_mixing", "block"),
        n_blocks=m.get("n_blocks", 1),
        ffn=m.get("ffn", "situ_glu"),
        chunk_size=m.get("chunk_size", 64),
        d_latent=m.get("d_latent", 0),
    )


def run(config_path: Path, out_dir: Path, device: str = "cpu") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    t = cfg["train"]
    corpus = build_corpus(cfg["corpus"])

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    sink = (out_dir / "results.jsonl").open("a")

    for arm in cfg["arms"]:
        for seed in t["seeds"]:
            spec = model_spec_for(arm, cfg, corpus.vocab_size)
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
                    wandb_project=cfg.get("wandb", {}).get("project"),
                    wandb_run_name=f"hybrid-ratio-{arm['id']}-s{seed}",
                    wandb_config={"arm": arm["id"], "pattern": arm["pattern"]},
                ),
                corpus,
            )
            row = {
                "arm": arm["id"],
                "pattern": ",".join(arm["pattern"]),
                "n_layers": cfg["model"]["n_layers"],
                "seed": seed,
                **metrics,
            }
            results.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(
                f"seed={seed} {arm['id']:>12}  "
                f"pattern={','.join(arm['pattern']):<20}  "
                f"local={metrics['val_markov_loss']:.4f}  "
                f"recall={metrics['val_recall_loss']:.4f}"
            )

    sink.close()
    return results
