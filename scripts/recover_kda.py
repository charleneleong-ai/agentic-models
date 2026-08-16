"""Recover remaining KDA state-capacity runs (append to existing results.jsonl)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path("/home/ubuntu/agentic-models")
sys.path.insert(0, str(REPO / "src"))

from archlab.ablations.kda_state_capacity import corpus_for, model_spec_for
from archlab.ablations.train import TrainSpec, train_arm

CONFIG = REPO / "configs" / "ablations" / "kda-state-capacity.yaml"
RESULTS = REPO / "experiments" / "kda-state-capacity" / "results.jsonl"
DEVICE = "cuda"


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    t = cfg["train"]
    sweep = cfg["sweep"]

    # Load existing results to find what's done
    existing = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    done = {(r["d_head"], r["nesting"], r["arm"], r["seed"]) for r in existing}
    print(f"Existing: {len(done)} runs")

    sink = RESULTS.open("a")
    total = 0

    for d_head in sweep["d_head"]:
        for nesting in sweep["nesting"]:
            corpus = corpus_for(cfg, nesting)
            for seed in t["seeds"]:
                for arm in cfg["arms"]:
                    if (d_head, nesting, arm["id"], seed) in done:
                        continue

                    spec = model_spec_for(arm, cfg, d_head, nesting, corpus.vocab_size)
                    state_kb = spec.n_heads * spec.d_head * spec.d_head * 4 / 1024
                    print(
                        f"d_head={d_head:>3} nesting={nesting:>2} "
                        f"seed={seed} {arm['id']:>12}  state={state_kb:.0f}KB ...",
                        flush=True,
                    )
                    metrics = train_arm(
                        spec,
                        TrainSpec(
                            steps=t["steps"],
                            batch_size=t["batch_size"],
                            lr=t["lr"],
                            warmup_frac=t.get("warmup_frac", 0.02),
                            weight_decay=t.get("weight_decay", 0.1),
                            seed=seed,
                            device=DEVICE,
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
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    total += 1
                    print(f"  -> recall={metrics['val_recall_loss']:.4f}")

    sink.close()
    print(f"\nDone. {total} new runs. Total: {len(existing) + total}")


if __name__ == "__main__":
    main()
