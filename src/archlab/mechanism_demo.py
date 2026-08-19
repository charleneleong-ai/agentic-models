"""Mechanism demo: needle-in-haystack distance sweep + chunk size ablation.

Demonstrates WHY KDA doesn't cause laziness at nano scale:
1. Distance sweep: does KDA's fixed state degrade with longer dependencies?
2. Chunk size sweep: does larger KDA window delay retrieval head formation?
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from archlab.ablations.train import (
    TrainSpec, make_batches, enforce_determinism, losses, evaluate, lr_at,
)
from archlab.data import CorpusSpec, N_SPECIAL, QUERY_TOKEN, markov_chain
from archlab.model import ModelSpec, NanoLM


# ─── Needle-in-Haystack Corpus ──────────────────────────────────────────────

@dataclass(frozen=True)
class NeedleCorpus:
    """Corpus where answer depends on a token planted at a controlled distance.

    Plant position is fixed (position 2), query position varies (distance away).
    This lets us sweep distance and measure how recall degrades.
    """
    vocab_size: int = 64
    seq_len: int = 256
    distance: int = 128  # plant-to-query gap
    n_pairs: int = 4
    key_vocab: int = 16
    seed: int = 0

    @property
    def n_filler(self) -> int:
        return self.vocab_size - N_SPECIAL - self.key_vocab


def generate_needle(spec: NeedleCorpus, n_seqs: int, seed: int | None = None) -> tuple[Tensor, Tensor]:
    """Generate sequences with plant-query pairs at fixed distance."""
    g = torch.Generator().manual_seed(spec.seed if seed is None else seed)
    trans = markov_chain(CorpusSpec(vocab_size=spec.vocab_size, seed=spec.seed))
    key_base = N_SPECIAL + spec.n_filler

    tokens = torch.empty(n_seqs, spec.seq_len, dtype=torch.long)
    state = torch.randint(spec.n_filler, (n_seqs,), generator=g)
    for t in range(spec.seq_len):
        tokens[:, t] = state + N_SPECIAL
        state = torch.multinomial(trans[state], 1, generator=g).squeeze(-1)

    answer_mask = torch.zeros(n_seqs, spec.seq_len, dtype=torch.bool)

    # Plant at position 2, query at position 2 + distance
    plant_pos = 2
    query_pos = plant_pos + spec.distance

    if query_pos + 2 >= spec.seq_len:
        raise ValueError(f"distance={spec.distance} too large for seq_len={spec.seq_len}")

    for i in range(n_seqs):
        keys = torch.randperm(spec.key_vocab, generator=g)[:spec.n_pairs]
        values = torch.randint(spec.n_filler, (spec.n_pairs,), generator=g) + N_SPECIAL

        # Plant key-value pairs
        for j, (k, v) in enumerate(zip(keys, values, strict=True)):
            pos = plant_pos + j * 2
            if pos + 1 < query_pos:
                tokens[i, pos] = key_base + int(k)
                tokens[i, pos + 1] = v

        # Query key-value pairs
        for j, (k, v) in enumerate(zip(keys, values, strict=True)):
            pos = query_pos + j * 3
            if pos + 2 < spec.seq_len:
                tokens[i, pos] = QUERY_TOKEN
                tokens[i, pos + 1] = key_base + int(k)
                tokens[i, pos + 2] = v
                answer_mask[i, pos + 2] = True

    return tokens, answer_mask


def needle_batches(spec: NeedleCorpus, batch_size: int, n_batches: int, seed: int) -> list[tuple[Tensor, Tensor]]:
    return [generate_needle(spec, batch_size, seed=seed * 100_000 + i) for i in range(n_batches)]


# ─── Training Functions ──────────────────────────────────────────────────────

def train_distance_sweep(
    model_spec: ModelSpec,
    distances: list[int],
    steps: int = 1200,
    batch_size: int = 16,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
) -> dict[int, dict[str, Any]]:
    """Train one model on needle corpus at different distances."""
    results = {}
    for dist in distances:
        enforce_determinism()
        torch.manual_seed(seed)
        model = NanoLM(model_spec).to(device)
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        corpus = NeedleCorpus(vocab_size=model_spec.vocab_size, distance=dist, seed=seed)
        train_data = needle_batches(corpus, batch_size, steps, seed)
        eval_data = needle_batches(corpus, batch_size, 50, 99991)

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

            if step % 100 == 0:
                curve.append({"step": step, "recall": round(recall.item(), 4)})

        val_local, val_recall = evaluate(model, eval_data, device)
        results[dist] = {
            "val_recall": round(val_recall, 4),
            "curve": curve,
        }
        print(f"  dist={dist:>3d}  recall={val_recall:.4f}")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    return results


def train_chunk_size_sweep(
    model_spec: ModelSpec,
    chunk_sizes: list[int],
    steps: int = 1200,
    batch_size: int = 16,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
) -> dict[int, dict[str, Any]]:
    """Train models with different KDA chunk sizes."""
    results = {}
    for cs in chunk_sizes:
        enforce_determinism()
        torch.manual_seed(seed)

        # Override chunk size
        spec = ModelSpec(
            vocab_size=model_spec.vocab_size,
            d_model=model_spec.d_model,
            n_layers=model_spec.n_layers,
            n_heads=model_spec.n_heads,
            d_head=model_spec.d_head,
            d_hidden=model_spec.d_hidden,
            attention_pattern=model_spec.attention_pattern,
            chunk_size=cs,
        )
        model = NanoLM(spec).to(device)
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        corpus = CorpusSpec(vocab_size=spec.vocab_size, seed=seed)
        train_data = make_batches(corpus, batch_size, steps, seed)
        eval_data = make_batches(corpus, batch_size, 50, 99991)

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

            if step % 100 == 0:
                curve.append({"step": step, "recall": round(recall.item(), 4)})

        val_local, val_recall = evaluate(model, eval_data, device)
        results[cs] = {
            "val_recall": round(val_recall, 4),
            "curve": curve,
        }
        print(f"  chunk={cs:>3d}  recall={val_recall:.4f}")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    return results


def run_mechanism_demos(
    config_path: Path,
    out_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run both mechanism demonstrations."""
    cfg = yaml.safe_load(config_path.read_text())
    m = cfg["model"]
    t = cfg["train"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build base model spec
    base_spec = ModelSpec(
        vocab_size=64,
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_head=m["d_head"],
        d_hidden=m["d_hidden"],
        attention_pattern=["kda"],
        chunk_size=m.get("chunk_size", 64),
    )

    # ─── Distance Sweep ──────────────────────────────────────────────────────
    print("=== Distance Sweep (needle-in-haystack) ===")
    distances = [16, 32, 64, 96, 128, 160, 192]
    distance_results = {}
    for arm_id, pattern in [("all-mla", ["mla"]), ("kda-3-1", ["kda", "kda", "kda", "mla"]), ("all-kda", ["kda"])]:
        spec = ModelSpec(
            vocab_size=base_spec.vocab_size,
            d_model=base_spec.d_model,
            n_layers=base_spec.n_layers,
            n_heads=base_spec.n_heads,
            d_head=base_spec.d_head,
            d_hidden=base_spec.d_hidden,
            attention_pattern=pattern,
            chunk_size=base_spec.chunk_size,
        )
        print(f"\n  {arm_id}:")
        distance_results[arm_id] = train_distance_sweep(
            spec, distances, steps=t["steps"], batch_size=t["batch_size"],
            lr=t["lr"], seed=0, device=device,
        )

    # ─── Chunk Size Sweep ────────────────────────────────────────────────────
    print("\n=== Chunk Size Sweep ===")
    chunk_sizes = [8, 16, 32, 64, 128]
    chunk_results = train_chunk_size_sweep(
        base_spec, chunk_sizes, steps=t["steps"], batch_size=t["batch_size"],
        lr=t["lr"], seed=0, device=device,
    )

    output = {
        "distance_sweep": distance_results,
        "chunk_size_sweep": chunk_results,
        "summary": {
            "distances_tested": distances,
            "chunk_sizes_tested": chunk_sizes,
            "arms_tested": list(distance_results.keys()),
        },
    }

    (out_dir / "mechanism_results.json").write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = run_mechanism_demos(args.config, args.out, args.device)
    print(json.dumps(output["summary"], indent=2))
