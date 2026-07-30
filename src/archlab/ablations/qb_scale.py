"""Runner for the qb-scale ablation: does QB's edge over sign-SGD widen with expert count?

Routing depends only on scores and bias, so no training is needed — the whole sweep is a
synthetic-router experiment that runs on CPU. See `configs/ablations/qb-scale.yaml`.

Fairness note: sign-SGD's step size is swept and its *best* result reported. QB has no step
size to tune, so giving the baseline a tuning budget QB never gets is the honest comparison.
Pinning one gamma would manufacture the conclusion.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from archlab.moe.quantile_balance import (
    alternating_qb_solve,
    expert_load,
    histogram_quantile_bias,
    load_imbalance,
    quantile_balance_update,
    routing_scores,
    topk_route,
)

BALANCED = 1.05  # "within 5%" of uniform load


@dataclass(frozen=True)
class Cell:
    """One point of the sweep: a synthetic router of a given size and saturation."""

    n_experts: int
    n_active: int
    n_tokens: int
    router_scale: float
    popularity_skew: float
    seed: int = 0


def make_scores(cell: Cell, d_model: int = 64) -> Tensor:
    """Synthetic router with a popularity gradient across experts.

    `router_scale` controls sigmoid saturation, which is the tie axis: at scale 1.0 a
    meaningful fraction of scores pin to exactly 0.0/1.0, and the quantile derivation
    (Appendix C) assumes no ties.
    """
    g = torch.Generator().manual_seed(cell.seed)
    x = torch.randn(cell.n_tokens, d_model, generator=g) * cell.router_scale
    w = torch.randn(cell.n_experts, d_model, generator=g) * cell.router_scale
    skew = torch.linspace(cell.popularity_skew, -cell.popularity_skew, cell.n_experts)
    return routing_scores(x, w + skew.unsqueeze(-1))


def imbalance_of(scores: Tensor, bias: Tensor, k: int, n: int) -> tuple[float, int]:
    routed, _, _ = topk_route(scores, bias, k)
    load = expert_load(routed, n)
    return load_imbalance(load), int((load == 0).sum())


def run_arm(arm: dict[str, Any], scores: Tensor, cell: Cell, steps: int) -> list[float]:
    """Return the imbalance trajectory: index i is the imbalance after i updates."""
    n, k = cell.n_experts, cell.n_active
    bias = torch.zeros(n)
    trajectory = [imbalance_of(scores, bias, k, n)[0]]
    kind = arm["balancer"]

    if kind is None:
        return trajectory * (steps + 1)

    if kind == "alternating_qb_solve":  # one-shot offline reference
        bias = alternating_qb_solve(scores, k, iters=arm.get("iters", 20))
        return trajectory + [imbalance_of(scores, bias, k, n)[0]] * steps

    for _ in range(steps):
        routed, _, alpha = topk_route(scores, bias, k)
        if kind == "sign":
            target = cell.n_tokens * k / n
            error = expert_load(routed, n).float() - target
            bias = bias - arm["gamma"] * torch.sign(error)
        elif kind == "quantile_balance_update":
            bias = quantile_balance_update(scores, alpha, k)
        elif kind == "histogram_quantile_bias":
            bias = histogram_quantile_bias(scores, alpha, k, n_bins=arm.get("n_bins", 1000))
        else:
            raise ValueError(f"unknown balancer: {kind}")
        trajectory.append(imbalance_of(scores, bias, k, n)[0])
    return trajectory


def steps_to_balanced(trajectory: list[float]) -> int | None:
    return next((i for i, v in enumerate(trajectory) if v < BALANCED), None)


def expand_arms(arm: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Sign-SGD carries a list of gammas; every other arm is a single configuration."""
    gammas = arm.get("gamma")
    if isinstance(gammas, list):
        for g in gammas:
            yield {**arm, "gamma": g}
    else:
        yield arm


def evaluate_cell(cell: Cell, arms: list[dict[str, Any]], steps: int) -> list[dict[str, Any]]:
    scores = make_scores(cell)
    ties = float(((scores == 0.0) | (scores == 1.0)).float().mean())
    rows = []

    for arm in arms:
        variants = [(v, run_arm(v, scores, cell, steps)) for v in expand_arms(arm)]
        # Report the arm's best configuration — see the fairness note in the module docstring.
        variant, trajectory = min(variants, key=lambda vt: vt[1][-1])
        _, dead = imbalance_of(scores, torch.zeros(cell.n_experts), cell.n_active, cell.n_experts)

        rows.append(
            {
                "arm": arm["id"],
                "n_experts": cell.n_experts,
                "router_scale": cell.router_scale,
                "load_imbalance": round(trajectory[-1], 4),
                "steps_to_within_5pct": steps_to_balanced(trajectory),
                "dead_expert_count": dead,
                "tie_fraction": round(ties, 4),
                "trajectory": [round(v, 4) for v in trajectory],
                "gamma": variant.get("gamma") if arm["balancer"] == "sign" else None,
                "n_gammas_tried": len(variants) if arm["balancer"] == "sign" else None,
            }
        )
    return rows


def run(config_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(config_path.read_text())
    sweep, arms, steps = cfg["sweep"], cfg["arms"], cfg["steps"]

    results = []
    for n_experts in sweep["n_experts"]:
        for scale in sweep["router_scale"]:
            cell = Cell(
                n_experts=n_experts,
                n_active=sweep["n_active"],
                n_tokens=sweep["n_tokens"],
                router_scale=scale,
                popularity_skew=sweep["popularity_skew"],
            )
            rows = evaluate_cell(cell, arms, steps)
            results.extend(rows)
            best = min(r["load_imbalance"] for r in rows if r["arm"] != "none")
            print(
                f"n={n_experts:4d} scale={scale:<4} ties={rows[0]['tie_fraction']:.3f} best={best:.4f}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    return results
