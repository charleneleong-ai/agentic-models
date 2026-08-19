"""Shared nano training loop for the ablations that need one.

Every arm sees byte-identical batches and an identical schedule; only the model differs. Data
is pre-generated once per seed and reused across arms, so a loss delta cannot come from one arm
drawing luckier samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from archlab.data import (
    ChainSpec,
    CorpusSpec,
    DyckSpec,
    batches,
    chain_batches,
    dyck_batches,
)
from archlab.model import ModelSpec, NanoLM, losses


@dataclass
class TrainSpec:
    steps: int = 1500
    batch_size: int = 16
    lr: float = 3e-4
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    eval_batches: int = 8
    seed: int = 0
    device: str = "cpu"


Corpus = CorpusSpec | ChainSpec | DyckSpec


def make_batches(
    corpus: Corpus, batch_size: int, n_batches: int, seed: int
) -> list[tuple[Tensor, Tensor]]:
    """Dispatch on corpus type. Both yield (tokens, target_mask) so the loop is agnostic."""
    if isinstance(corpus, DyckSpec):
        return dyck_batches(corpus, batch_size, n_batches, seed)
    if isinstance(corpus, ChainSpec):
        return chain_batches(corpus, batch_size, n_batches, seed)
    return batches(corpus, batch_size, n_batches, seed)


def lr_at(step: int, spec: TrainSpec) -> float:
    """Linear warmup then cosine decay — the schedule K3 §3.2 found beats WSD when both tuned."""
    warmup = max(1, int(spec.warmup_frac * spec.steps))
    if step < warmup:
        return spec.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, spec.steps - warmup)
    return spec.lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: NanoLM, data: list[tuple[Tensor, Tensor]], device: str) -> tuple[float, float]:
    model.eval()
    local, recall = [], []
    for tokens, mask in data:
        tokens, mask = tokens.to(device), mask.to(device)
        loc, rec = losses(model(tokens), tokens, mask)
        local.append(float(loc))
        recall.append(float(rec))
    model.train()
    return sum(local) / len(local), sum(recall) / len(recall)


def train_arm(model_spec: ModelSpec, train_spec: TrainSpec, corpus: CorpusSpec) -> dict[str, Any]:
    """Train one arm and return its metrics. Deterministic given the seeds."""
    torch.manual_seed(train_spec.seed)
    device = train_spec.device
    model = NanoLM(model_spec).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=train_spec.lr, weight_decay=train_spec.weight_decay
    )

    # Identical data for every arm: same corpus seed, same batch order.
    train_data = make_batches(corpus, train_spec.batch_size, train_spec.steps, train_spec.seed)
    eval_data = make_batches(corpus, train_spec.batch_size, train_spec.eval_batches, 99991)

    peak_sources, curve = 0, []
    for step, (tokens, mask) in enumerate(train_data):
        tokens, mask = tokens.to(device), mask.to(device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, train_spec)

        local, recall = losses(model(tokens), tokens, mask)
        (local + recall).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        peak_sources = max(peak_sources, model.peak_live_sources())
        if step % max(1, train_spec.steps // 10) == 0:
            curve.append(
                {
                    "step": step,
                    "local": round(local.item(), 4),
                    "recall": round(recall.item(), 4),
                }
            )

    val_local, val_recall = evaluate(model, eval_data, device)
    return {
        "val_local_loss": round(val_local, 4),
        "val_recall_loss": round(val_recall, 4),
        "n_params": model.n_params(),
        "peak_live_sources": peak_sources,
        "curve": curve,
    }
