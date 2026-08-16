"""Runner for kda-state-capacity: when does a fixed recurrent state stop holding the sequence?

Sweeps nesting depth (state requirements) against d_head (state capacity) on Dyck. The expected
failure signature: recall loss degrades smoothly as nesting approaches d_head, then collapses.
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


def model_spec_for(
    arm: dict[str, Any],
    cfg: dict[str, Any],
    d_head: int,
    nesting: int,
    vocab_size: int,
) -> ModelSpec:
    m = cfg["model"]
    pattern = arm.get("attention_pattern", [arm.get("attention", "kda")])
    return ModelSpec(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_head=d_head,
        d_hidden=m["d_hidden"],
        attention_pattern=pattern,
        depth_mixing=m.get("depth_mixing", "residual"),
        n_blocks=m.get("n_blocks", 1),
        ffn=m.get("ffn", "situ_glu"),
        chunk_size=m.get("chunk_size", 64),
    )


def corpus_for(cfg: dict[str, Any], nesting: int):
    """Build Dyck corpus at the specified nesting depth."""
    corpus_cfg = dict(cfg["corpus"])
    corpus_cfg["depth"] = nesting
    # seq_len must accommodate n_groups of width (2*depth+1) and be a multiple of chunk_size
    n_groups = corpus_cfg.get("n_groups", 2)
    width = 2 * nesting + 1
    min_seq = n_groups * width
    chunk = cfg["model"].get("chunk_size", 64)
    seq_len = ((min_seq + chunk - 1) // chunk) * chunk
    corpus_cfg["seq_len"] = max(seq_len, chunk)
    return build_corpus(corpus_cfg)


def run(config_path: Path, out_dir: Path, device: str = "cpu") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    t = cfg["train"]
    sweep = cfg["sweep"]

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    sink = (out_dir / "results.jsonl").open("w")

    for d_head in sweep["d_head"]:
        for nesting in sweep["nesting"]:
            corpus = corpus_for(cfg, nesting)
            for seed in t["seeds"]:
                for arm in cfg["arms"]:
                    spec = model_spec_for(arm, cfg, d_head, nesting, corpus.vocab_size)
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
                    row = {
                        "arm": arm["id"],
                        "d_head": d_head,
                        "nesting": nesting,
                        "n_heads": cfg["model"]["n_heads"],
                        "state_bytes": cfg["model"]["n_heads"] * d_head * d_head * 4,
                        "seed": seed,
                        **metrics,
                    }
                    results.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    rl = metrics["val_recall_loss"]
                    state_kb = row["state_bytes"] / 1024
                    print(
                        f"d_head={d_head:>3} nesting={nesting:>2} "
                        f"seed={seed} {arm['id']:>12}  "
                        f"recall={rl:.4f}  state={state_kb:.0f}KB"
                    )

    sink.close()
    return results
