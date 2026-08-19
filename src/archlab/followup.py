"""Follow-up: permutation composition chain_len=3 and gradient profiling at 48L."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from archlab.ablations.train import (
    TrainSpec, make_batches, enforce_determinism, losses, evaluate, lr_at,
)
from archlab.data import CorpusSpec, PermSpec, perm_batches
from archlab.model import ModelSpec, NanoLM


def train_chain_len_sweep(
    chain_lens: list[int],
    steps: int = 2400,
    batch_size: int = 16,
    lr: float = 3e-4,
    n_layers: int = 12,
    seed: int = 0,
    device: str = "cpu",
) -> dict[int, dict[str, Any]]:
    """Test chain_len=3 to find exact cliff boundary."""
    results = {}
    for cl in chain_lens:
        enforce_determinism()
        torch.manual_seed(seed)

        spec = ModelSpec(
            vocab_size=64, d_model=128, n_layers=n_layers, n_heads=4,
            d_head=32, d_hidden=512, attention_pattern=["mla"],
        )
        model = NanoLM(spec).to(device)
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        corpus = PermSpec(vocab_size=64, seq_len=256, chain_len=cl, seed=seed)
        train_data = perm_batches(corpus, batch_size, steps, seed)
        eval_data = perm_batches(corpus, batch_size, 50, 99991)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        base_lrs = [g["lr"] for g in opt.param_groups]

        curve = []
        for step, (tokens, mask) in enumerate(train_data):
            tokens, mask = tokens.to(device), mask.to(device)
            schedule = lr_at(step, TrainSpec(steps=steps, lr=lr)) / lr
            for group, base in zip(opt.param_groups, base_lrs, strict=True):
                group["lr"] = base * schedule

            local, recall = losses(model(tokens), tokens, mask)
            (local + recall).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if step % 200 == 0:
                curve.append({"step": step, "recall": round(recall.item(), 4)})

        val_local, val_recall = evaluate(model, eval_data, device)
        results[cl] = {
            "val_recall": round(val_recall, 4),
            "val_local": round(val_local, 4),
            "curve": curve,
        }
        print("  chain_len=%d  recall=%.4f  local=%.4f" % (cl, val_recall, val_local))

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    return results


def profile_gradients_at_48L(
    steps: int = 400,
    batch_size: int = 16,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Profile gradient norms at 48 layers for full vs blocked AttnRes."""
    results = {}

    for depth_mixing, n_blocks in [("residual", 1), ("block", 8), ("full", 1)]:
        enforce_determinism()
        torch.manual_seed(seed)

        spec = ModelSpec(
            vocab_size=64, d_model=128, n_layers=48, n_heads=4,
            d_head=32, d_hidden=512, attention_pattern=["mla"],
            depth_mixing=depth_mixing, n_blocks=n_blocks,
        )
        model = NanoLM(spec).to(device)
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        corpus = CorpusSpec(vocab_size=64, seq_len=256, seed=seed)
        train_data = make_batches(corpus, batch_size, steps, seed)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        base_lrs = [g["lr"] for g in opt.param_groups]

        grad_norms = {"embed": [], "layers": [], "depth": [], "head": []}
        for step, (tokens, mask) in enumerate(train_data):
            tokens, mask = tokens.to(device), mask.to(device)
            schedule = lr_at(step, TrainSpec(steps=steps, lr=lr)) / lr
            for group, base in zip(opt.param_groups, base_lrs, strict=True):
                group["lr"] = base * schedule

            local, recall = losses(model(tokens), tokens, mask)
            (local + recall).backward()

            # Record gradient norms before clipping
            embed_norm = float(model.embed.weight.grad.norm()) if model.embed.weight.grad is not None else 0
            layer_norms = []
            for layer in model.layers:
                if layer.attn_norm.weight.grad is not None:
                    layer_norms.append(float(layer.attn_norm.weight.grad.norm()))
            depth_norm = float(model.depth.parameters().__next__().grad.norm()) if model.depth.parameters().__next__().grad is not None else 0
            head_norm = float(model.head.weight.grad.norm()) if model.head.weight.grad is not None else 0

            grad_norms["embed"].append(embed_norm)
            grad_norms["layers"].append(sum(layer_norms) / len(layer_norms) if layer_norms else 0)
            grad_norms["depth"].append(depth_norm)
            grad_norms["head"].append(head_norm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        results[depth_mixing] = {
            "embed_grad_avg": sum(grad_norms["embed"]) / len(grad_norms["embed"]),
            "layer_grad_avg": sum(grad_norms["layers"]) / len(grad_norms["layers"]),
            "depth_grad_avg": sum(grad_norms["depth"]) / len(grad_norms["depth"]),
            "head_grad_avg": sum(grad_norms["head"]) / len(grad_norms["head"]),
            "embed_grad_max": max(grad_norms["embed"]),
            "layer_grad_max": max(grad_norms["layers"]),
            "depth_grad_max": max(grad_norms["depth"]),
            "head_grad_max": max(grad_norms["head"]),
        }
        print("  %s: embed=%.4f  layers=%.4f  depth=%.4f  head=%.4f" % (
            depth_mixing,
            results[depth_mixing]["embed_grad_avg"],
            results[depth_mixing]["layer_grad_avg"],
            results[depth_mixing]["depth_grad_avg"],
            results[depth_mixing]["head_grad_avg"],
        ))

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    return results


def run_followups(
    out_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run both follow-ups."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Chain Length Sweep (finding the cliff) ===")
    chain_results = train_chain_len_sweep(
        [1, 2, 3, 4, 6, 8], steps=2400, batch_size=16, lr=3e-4,
        n_layers=12, seed=0, device=device,
    )

    print("\n=== Gradient Profiling at 48L ===")
    grad_results = profile_gradients_at_48L(
        steps=400, batch_size=16, lr=3e-4, seed=0, device=device,
    )

    output = {
        "chain_len_sweep": {str(k): v for k, v in chain_results.items()},
        "gradient_profiling": grad_results,
        "summary": {
            "chain_lens_tested": [1, 2, 3, 4, 6, 8],
            "depth_mixings_tested": ["residual", "block", "full"],
        },
    }

    (out_dir / "followup_results.json").write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = run_followups(args.out, args.device)
    print(json.dumps(output["summary"], indent=2))
