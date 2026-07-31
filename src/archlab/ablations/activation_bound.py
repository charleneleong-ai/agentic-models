"""Runner for activation-bound: what does capping the activation cost, and what does it buy?

K3 justifies SiTU-GLU entirely on precision grounds (§2.3.2) and shows no quality comparison.
Two questions, and they need separating:

  (a) at bf16, does the cap cost anything? If yes it is a precision tax, not a free win.
  (b) under FP8-range activations, does the unbounded baseline actually break?

Both arms are instrumented identically and share one activation accumulator per run, because
the failure mode is a rare coincident outlier *anywhere* in the network — a per-layer average
would dilute exactly the event of interest. Peak magnitude is recorded before quantization, so
it reports the true activation rather than the already-clipped one.

Runs on the recall corpus, not the compositional one. This ablation asks nothing about depth or
composition, so the corpus only needs a learnable signal to make loss differences meaningful,
and the recall corpus's local loss provides that (baseline ~2.79).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from archlab.ablations.train import TrainSpec, evaluate, lr_at, make_batches
from archlab.data import CorpusSpec
from archlab.model import ModelSpec, NanoLM, losses


def model_spec_for(
    arm: dict[str, Any], precision: str, cfg: dict[str, Any], vocab_size: int
) -> ModelSpec:
    m = cfg["model"]
    return ModelSpec(
        vocab_size=vocab_size,
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_head=m["d_head"],
        d_hidden=m["d_hidden"],
        attention_pattern=[m.get("attention", "kda")],
        depth_mixing=m.get("depth_mixing", "block"),
        n_blocks=m.get("n_blocks", 4),
        chunk_size=m.get("chunk_size", 64),
        ffn=arm["activation"],
        precision=precision,
        beta_gate=arm.get("beta_gate", 4.0),
        beta_up=arm.get("beta_up", 25.0),
    )


def train_instrumented(
    model_spec: ModelSpec, train_spec: TrainSpec, corpus: CorpusSpec
) -> dict[str, Any]:
    """Like `train_arm`, but records activation statistics and instability counts.

    Kept separate rather than folded into the shared loop: every other ablation would pay the
    probe's cost for numbers it never reads.
    """
    torch.manual_seed(train_spec.seed)
    device = train_spec.device
    model = NanoLM(model_spec).to(device)
    stats = model.attach_activation_probes()
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_spec.lr, weight_decay=train_spec.weight_decay
    )

    train_data = make_batches(corpus, train_spec.batch_size, train_spec.steps, train_spec.seed)
    eval_data = make_batches(corpus, train_spec.batch_size, train_spec.eval_batches, 99991)

    nonfinite_steps, grad_spikes, grad_norms = 0, 0, []
    for step, (tokens, mask) in enumerate(train_data):
        tokens, mask = tokens.to(device), mask.to(device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, train_spec)

        local, recall = losses(model(tokens), tokens, mask)
        loss = local + recall
        if not torch.isfinite(loss):
            nonfinite_steps += 1
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        # A spike is relative to this run's own history, so the count is comparable across arms
        # whose gradient scales differ.
        if grad_norms and grad_norm > 5 * (sum(grad_norms[-50:]) / len(grad_norms[-50:])):
            grad_spikes += 1
        grad_norms.append(grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

    val_local, val_recall = evaluate(model, eval_data, device)
    return {
        "val_local_loss": round(val_local, 4),
        "val_recall_loss": round(val_recall, 4),
        "nonfinite_steps": nonfinite_steps,
        "grad_norm_spikes": grad_spikes,
        "n_params": model.n_params(),
        **stats.summary(),
    }


def run(config_path: Path, out_dir: Path, device: str = "cpu") -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    t, c = cfg["train"], cfg["corpus"]
    corpus = CorpusSpec(
        vocab_size=c["vocab_size"],
        seq_len=c["seq_len"],
        n_pairs=c["n_pairs"],
        key_vocab=c["key_vocab"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    sink = (out_dir / "results.jsonl").open("w")

    for precision in cfg["precision"]:
        for seed in t["seeds"]:
            for arm in cfg["arms"]:
                spec = model_spec_for(arm, precision, cfg, corpus.vocab_size)
                metrics = train_instrumented(
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
                row = {"arm": arm["id"], "precision": precision, "seed": seed, **metrics}
                results.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                print(
                    f"{precision:>8} seed={seed} {arm['id']:>14}  "
                    f"local={row['val_local_loss']:.4f} max|h|={row['max_abs_activation']:>9.2f} "
                    f"fp8_over={row['fp8_overflow_frac']:.2e} spikes={row['grad_norm_spikes']} "
                    f"nonfinite={row['nonfinite_steps']}",
                    flush=True,
                )

    sink.close()
    return results
