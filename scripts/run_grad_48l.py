#!/usr/bin/env python3
"""Run AttnRes gradient profiling at 48L when GPU is free."""

import json
import sys
import torch
from pathlib import Path

sys.path.insert(0, "src")
from archlab.ablations.train import enforce_determinism, losses, lr_at, TrainSpec, make_batches
from archlab.model import ModelSpec, NanoLM
from archlab.data import CorpusSpec
from archlab.visualize_followup import plot_followups


def main():
    print("=== Gradient Profiling at 48L ===")
    
    grad_results = {}
    for depth_mixing, n_blocks in [("residual", 1), ("block", 4), ("full", 1)]:
        enforce_determinism()
        torch.manual_seed(0)
        
        # Full model: d_model=256, 48 layers
        spec = ModelSpec(
            vocab_size=64, d_model=256, n_layers=48, n_heads=4, d_head=64,
            d_hidden=1024, attention_pattern=["mla"],
            depth_mixing=depth_mixing, n_blocks=n_blocks,
        )
        model = NanoLM(spec).to("cuda")
        model.train()
        for p in model.parameters():
            p.requires_grad = True
        
        corpus = CorpusSpec(vocab_size=64, seq_len=256, seed=0)
        train_data = make_batches(corpus, 8, 200, 0)
        
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        base_lrs = [g["lr"] for g in opt.param_groups]
        
        grad_norms = {"embed": [], "layers": [], "head": []}
        for step, (tokens, mask) in enumerate(train_data):
            tokens, mask = tokens.to("cuda"), mask.to("cuda")
            schedule = lr_at(step, TrainSpec(steps=200, lr=3e-4)) / 3e-4
            for group, base in zip(opt.param_groups, base_lrs, strict=True):
                group["lr"] = base * schedule
            
            local, recall = losses(model(tokens), tokens, mask)
            (local + recall).backward()
            
            embed_norm = float(model.embed.weight.grad.norm()) if model.embed.weight.grad is not None else 0
            layer_norms = []
            for layer in model.layers:
                if layer.attn_norm.weight.grad is not None:
                    layer_norms.append(float(layer.attn_norm.weight.grad.norm()))
            head_norm = float(model.head.weight.grad.norm()) if model.head.weight.grad is not None else 0
            
            grad_norms["embed"].append(embed_norm)
            grad_norms["layers"].append(sum(layer_norms) / len(layer_norms) if layer_norms else 0)
            grad_norms["head"].append(head_norm)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        
        grad_results[depth_mixing] = {
            "embed_grad_avg": sum(grad_norms["embed"]) / len(grad_norms["embed"]),
            "layer_grad_avg": sum(grad_norms["layers"]) / len(grad_norms["layers"]),
            "head_grad_avg": sum(grad_norms["head"]) / len(grad_norms["head"]),
        }
        print("  %s: embed=%.6f  layers=%.6f  head=%.6f" % (
            depth_mixing,
            grad_results[depth_mixing]["embed_grad_avg"],
            grad_results[depth_mixing]["layer_grad_avg"],
            grad_results[depth_mixing]["head_grad_avg"],
        ))
        del model
        torch.cuda.empty_cache()
    
    # Load existing chain length results
    chain_file = Path("experiments/hybrid-ratio/followup/followup_results.json")
    if chain_file.exists():
        existing = json.loads(chain_file.read_text())
        chain_results = existing.get("chain_len_sweep", {})
    else:
        chain_results = {}
    
    output = {
        "chain_len_sweep": chain_results,
        "gradient_profiling": grad_results,
        "summary": {
            "chain_lens_tested": sorted(int(k) for k in chain_results.keys()),
            "depth_mixings_tested": ["residual", "block", "full"],
        },
    }
    
    out_dir = Path("experiments/hybrid-ratio/followup")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "followup_results.json").write_text(json.dumps(output, indent=2))
    print("\nSaved to %s" % (out_dir / "followup_results.json"))
    
    plot_followups(
        out_dir / "followup_results.json",
        out_dir,
    )
    
    print("\nDone! Copy PNG to assets:")
    print("  scp pi-a100-80gb:~/agentic-models/experiments/hybrid-ratio/followup/followup_results.png assets/")


if __name__ == "__main__":
    main()
