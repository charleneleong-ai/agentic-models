"""Shared nano training loop for the ablations that need one.

Every arm sees byte-identical batches and an identical schedule; only the model differs. Data
is pre-generated once per seed and reused across arms, so a loss delta cannot come from one arm
drawing luckier samples.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from archlab.data import (
    ChainSpec,
    CorpusSpec,
    DyckSpec,
    PermSpec,
    batches,
    chain_batches,
    dyck_batches,
    perm_batches,
)
from archlab.model import ModelSpec, NanoLM, losses
from archlab.mup import mup_param_groups

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


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
    deterministic: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_config: dict[str, Any] = field(default_factory=dict)


Corpus = CorpusSpec | ChainSpec | DyckSpec | PermSpec


def make_batches(
    corpus: Corpus, batch_size: int, n_batches: int, seed: int
) -> list[tuple[Tensor, Tensor]]:
    """Dispatch on corpus type. Both yield (tokens, target_mask) so the loop is agnostic."""
    if isinstance(corpus, DyckSpec):
        return dyck_batches(corpus, batch_size, n_batches, seed)
    if isinstance(corpus, ChainSpec):
        return chain_batches(corpus, batch_size, n_batches, seed)
    if isinstance(corpus, PermSpec):
        return perm_batches(corpus, batch_size, n_batches, seed)
    return batches(corpus, batch_size, n_batches, seed)


def enforce_determinism() -> None:
    """Make runs bit-reproducible across *processes*, not just within one.

    Measured on an A100: the same config and seed, run in three separate processes, spanned
    0.20 in final loss — against 0.038 for three runs inside a single process. Non-deterministic
    CUDA kernels (atomics, per-process algorithm selection) are the cause. With this enabled the
    same three processes agreed to four decimal places.

    On by default because an ablation is a *comparison between runs*. A harness that is only
    reproducible within one process silently invalidates every cross-invocation comparison, and
    the failure is invisible — it looks like an effect. The cost is speed, which is a real
    tradeoff for production training and cheap insurance for a measurement tool.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    if train_spec.deterministic:
        enforce_determinism()
    torch.manual_seed(train_spec.seed)
    device = train_spec.device
    model = NanoLM(model_spec).to(device)

    # wandb logging (optional — skips gracefully if not installed or no auth)
    wb_run = None
    if train_spec.wandb_project and _WANDB_AVAILABLE:
        wb_run = _wandb.init(
            project=train_spec.wandb_project,
            name=train_spec.wandb_run_name,
            config={
                **train_spec.wandb_config,
                "seed": train_spec.seed,
                "device": device,
                "n_params": model.n_params(),
            },
        )

    # Under muP the learning rate is per-parameter-class, so the optimum holds still as width
    # changes and a width ladder measures the mechanism rather than the parametrization.
    params = (
        mup_param_groups(model, model.width_mult, train_spec.lr, train_spec.weight_decay)
        if model_spec.mup
        else model.parameters()
    )
    opt = torch.optim.AdamW(params, lr=train_spec.lr, weight_decay=train_spec.weight_decay)

    # Identical data for every arm: same corpus seed, same batch order.
    train_data = make_batches(corpus, train_spec.batch_size, train_spec.steps, train_spec.seed)
    eval_data = make_batches(corpus, train_spec.batch_size, train_spec.eval_batches, 99991)

    base_lrs = [g["lr"] for g in opt.param_groups]
    peak_sources, curve = 0, []
    for step, (tokens, mask) in enumerate(train_data):
        tokens, mask = tokens.to(device), mask.to(device)
        # Preserve the per-group ratio muP established; the schedule only scales it.
        schedule = lr_at(step, train_spec) / train_spec.lr
        for group, base in zip(opt.param_groups, base_lrs, strict=True):
            group["lr"] = base * schedule

        local, recall = losses(model(tokens), tokens, mask)
        (local + recall).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        peak_sources = max(peak_sources, model.peak_live_sources())

        if wb_run:
            _wandb.log(
                {
                    "train/markov_loss": local.item(),
                    "train/recall_loss": recall.item(),
                    "train/lr": base_lrs[0] * schedule,
                    "train/step": step,
                },
                step=step,
            )

        if step % max(1, train_spec.steps // 10) == 0:
            curve.append(
                {
                    "step": step,
                    "local": round(local.item(), 4),
                    "recall": round(recall.item(), 4),
                }
            )

    val_local, val_recall = evaluate(model, eval_data, device)
    metrics = {
        "val_markov_loss": round(val_local, 4),
        "val_recall_loss": round(val_recall, 4),
        "n_params": model.n_params(),
        "peak_live_sources": peak_sources,
        "curve": curve,
    }

    if wb_run:
        _wandb.log(
            {
                "val/markov_loss": val_local,
                "val/recall_loss": val_recall,
                "val/n_params": model.n_params(),
                "val/peak_live_sources": peak_sources,
            }
        )

    return metrics
