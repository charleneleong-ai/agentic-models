"""Fine-grained laziness proxy: training curves + attention weight analysis.

Uses the existing train_arm but with finer logging interval, plus captures
attention weights on filler tokens to detect if KDA delays retrieval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from archlab.ablations.hybrid_ratio import model_spec_for
from archlab.ablations.train import (
    TrainSpec, make_batches, enforce_determinism, losses, evaluate, lr_at,
)
from archlab.data import CorpusSpec, build_corpus
from archlab.model import NanoLM


def train_arm_curve(
    model_spec: Any,
    train_spec: TrainSpec,
    corpus: CorpusSpec,
    log_interval: int = 100,
) -> dict[str, Any]:
    """Train one arm with fine-grained recall logging."""
    if train_spec.deterministic:
        enforce_determinism()
    torch.manual_seed(train_spec.seed)
    device = train_spec.device
    model = NanoLM(model_spec).to(device)

    params = model.parameters()
    opt = torch.optim.AdamW(params, lr=train_spec.lr, weight_decay=train_spec.weight_decay)

    train_data = make_batches(corpus, train_spec.batch_size, train_spec.steps, train_spec.seed)
    eval_data = make_batches(corpus, train_spec.batch_size, train_spec.eval_batches, 99991)

    base_lrs = [g["lr"] for g in opt.param_groups]
    curve = []
    for step, (tokens, mask) in enumerate(train_data):
        tokens, mask = tokens.to(device), mask.to(device)
        schedule = lr_at(step, train_spec) / train_spec.lr
        for group, base in zip(opt.param_groups, base_lrs, strict=True):
            group["lr"] = base * schedule

        local, recall = losses(model(tokens), tokens, mask)
        (local + recall).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % log_interval == 0:
            curve.append({
                "step": step,
                "local": round(local.item(), 4),
                "recall": round(recall.item(), 4),
            })

    val_local, val_recall = evaluate(model, eval_data, device)
    return {
        "val_markov_loss": round(val_local, 4),
        "val_recall_loss": round(val_recall, 4),
        "curve": curve,
        "n_params": model.n_params(),
    }


@torch.no_grad()
def capture_attention_weights(
    model_spec: Any,
    corpus: CorpusSpec,
    device: str = "cpu",
) -> dict[str, Any]:
    """Capture MLA attention patterns: softmax scores on filler vs non-filler tokens."""
    enforce_determinism()
    torch.manual_seed(0)
    model = NanoLM(model_spec).to(device)
    model.eval()

    eval_data = make_batches(corpus, 16, 20, 99991)

    layer_filler: dict[int, list[float]] = {i: [] for i in range(model.spec.n_layers)}
    layer_non_filler: dict[int, list[float]] = {i: [] for i in range(model.spec.n_layers)}

    for tokens, mask in eval_data:
        tokens = tokens.to(device)
        h = model.embed(tokens)

        for i, layer in enumerate(model.layers):
            attn_in = layer.attn_norm(h)
            a = layer.attn(attn_in)

            # For MLA layers, compute attention scores
            if layer.attn.kind == "mla" and hasattr(layer.attn.inner, 'kv_down'):
                q = layer.attn.inner.split_heads(layer.attn.inner.q_proj(attn_in))
                c = layer.attn.inner.kv_down(attn_in)
                k = layer.attn.inner.split_heads(layer.attn.inner.k_up(c))

                scores = (q @ k.transpose(-1, -2)) / math.sqrt(model.spec.d_head)
                attn_weights = torch.softmax(scores, dim=-1)
                avg_attn = attn_weights.mean(dim=1).mean(dim=0)

                filler_mask = mask.bool()
                if filler_mask.any():
                    filler_mass = float(avg_attn[:, filler_mask].sum(dim=-1).mean())
                    non_filler_mass = float(avg_attn[:, ~filler_mask].sum(dim=-1).mean())
                    layer_filler[i].append(filler_mass)
                    layer_non_filler[i].append(non_filler_mass)

            h = h + a + layer.ffn(layer.ffn_norm(h + a))

    model.train()
    return {
        "filler_attention": {i: sum(v)/len(v) if v else 0.0 for i, v in layer_filler.items()},
        "non_filler_attention": {i: sum(v)/len(v) if v else 0.0 for i, v in layer_non_filler.items()},
    }


def run_laziness_proxies(
    config_path: Path,
    out_dir: Path,
    device: str = "cpu",
    log_interval: int = 100,
) -> dict[str, Any]:
    """Run both proxies: fine-grained curves + attention analysis."""
    cfg = yaml.safe_load(config_path.read_text())
    corpus = build_corpus(cfg["corpus"])
    t = cfg["train"]

    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    print("=== Fine-grained training curves ===")
    for arm in cfg["arms"]:
        for seed in t["seeds"]:
            spec = model_spec_for(arm, cfg, corpus.vocab_size)
            metrics = train_arm_curve(
                spec,
                TrainSpec(
                    steps=t["steps"],
                    batch_size=t["batch_size"],
                    lr=t["lr"],
                    seed=seed,
                    device=device,
                    eval_batches=t.get("eval_batches", 50),
                ),
                corpus,
                log_interval=log_interval,
            )
            row = {
                "arm": arm["id"],
                "pattern": ",".join(arm["pattern"]),
                "seed": seed,
                **metrics,
            }
            results.append(row)
            print(f"  seed={seed} {arm['id']:>12}  recall={metrics['val_recall_loss']:.4f}")

    curves_path = out_dir / "fine_grained_curves.jsonl"
    with curves_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print("\n=== Attention weight analysis ===")
    attention_results = {}
    for arm in cfg["arms"]:
        spec = model_spec_for(arm, cfg, corpus.vocab_size)
        attn = capture_attention_weights(spec, corpus, device)
        attention_results[arm["id"]] = attn
        filler_avg = sum(attn["filler_attention"].values()) / len(attn["filler_attention"])
        print(f"  {arm['id']:>12}  filler_attn={filler_avg:.4f}")

    attn_path = out_dir / "attention_analysis.json"
    with attn_path.open("w") as f:
        json.dump(attention_results, f, indent=2)

    return {
        "fine_grained_curves": results,
        "attention_analysis": attention_results,
        "summary": {
            "total_runs": len(results),
            "arms_tested": [a["id"] for a in cfg["arms"]],
            "log_interval": log_interval,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    output = run_laziness_proxies(args.config, args.out, args.device, args.log_interval)
    print(json.dumps(output["summary"], indent=2))
