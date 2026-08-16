"""Recover the5 OOM-killed48-layer runs with batch_size=8."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO = Path("/home/ubuntu/agentic-models")
sys.path.insert(0, str(REPO / "src"))

from archlab.ablations.depth_sweep import GAP_METRIC, BASELINE, model_spec_for, ordered_arms
from archlab.ablations.train import TrainSpec, train_arm
from archlab.data import build_corpus
CONFIG = REPO / "configs" / "ablations" / "attn-res-dyck.yaml"
RESULTS = REPO / "experiments" / "attn-res-dyck" / "results.jsonl"
DEVICE = "cuda"

# The5 runs that OOM'd
MISSING = {
    (48, "residual", 1),
    (48, "attnres-full", 0),
    (48, "attnres-full", 1),
    (48, "attnres-block-6", 0),
    (48, "attnres-block-6", 1),
}


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    t = cfg["train"]
    t["batch_size"] = 8  # halved for OOM recovery
    corpus = build_corpus(cfg["corpus"])
    primary = cfg.get("metrics", {}).get("gap_on", "val_local_loss")

    sink = RESULTS.open("a")  # append, not overwrite
    done = 0

    # Build baseline lookup from existing results
    existing = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    baseline_cache: dict[int, dict[str, float]] = {}
    for r in existing:
        if r["arm"] == BASELINE and r["n_layers"] == 48:
            baseline_cache[r["seed"]] = {
                "val_local_loss": r["val_local_loss"],
                "val_recall_loss": r["val_recall_loss"],
            }

    for n_layers in [48]:
        for seed in t["seeds"]:
            # Load baseline from existing results for this seed, if it exists
            baseline: dict[str, float] = {}
            if seed in baseline_cache:
                baseline = baseline_cache[seed]

            for arm in ordered_arms(cfg["arms"]):
                if (n_layers, arm["id"], seed) not in MISSING:
                    continue

                # Use existing baseline if available
                if arm["id"] == BASELINE and seed in baseline_cache:
                    print(f"L=48 seed={seed} residual  (cached) skip")
                    continue

                spec = model_spec_for(arm, cfg, n_layers, corpus.vocab_size)
                print(f"L=48 seed={seed} {arm['id']:>16}  training with bs=8 ...", flush=True)
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
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                done += 1
                print(
                    f"  -> local={metrics['val_local_loss']:.4f} "
                    f"recall={metrics['val_recall_loss']:.4f} gap={gap:+.4f}"
                )

    sink.close()
    print(f"\nDone. {done} runs recovered. Total rows: {len(existing) + done}")


if __name__ == "__main__":
    main()
