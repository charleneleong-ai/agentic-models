"""Visualize proxy results — three panels demonstrating KDA doesn't cause laziness."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_proxy_results(results_path: Path, out_dir: Path) -> None:
    """Generate three-panel figure."""
    data = json.loads(results_path.read_text())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ─── Panel 1: Recall Head Convergence Curves ─────────────────────────────
    ax1 = axes[0]
    curves = data["recall_convergence"]
    colors = {"all-mla": "#2196F3", "kda-1-1": "#4CAF50", "kda-3-1": "#FF9800", "kda-7-1": "#F44336", "all-kda": "#9C27B0"}

    for arm, points in curves.items():
        steps = [p["step"] for p in points]
        recalls = [p["recall"] for p in points]
        ax1.plot(steps, recalls, label=arm, color=colors.get(arm, "gray"), alpha=0.7)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Recall Loss")
    ax1.set_title("Recall Head Convergence")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ─── Panel 2: MLA Attention Output Variance ─────────────────────────────
    ax2 = axes[1]
    attn = data["mla_attention_analysis"]["layer_output_variance"]
    layers = sorted(attn.keys())
    variances = [attn[l] for l in layers]

    ax2.bar(layers, variances, color="#2196F3", alpha=0.7)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Output Variance (proxy for attention focus)")
    ax2.set_title("MLA Attention Concentration by Layer")
    ax2.grid(True, alpha=0.3, axis="y")

    # ─── Panel 3: State Compression Efficiency ──────────────────────────────
    ax3 = axes[2]
    eff = data["state_compression_efficiency"]
    arms = list(eff.keys())
    recalls = [eff[a]["avg_recall_loss"] for a in arms]
    bytes_kb = [eff[a]["state_bytes"] / 1024 for a in arms]

    scatter = ax3.scatter(bytes_kb, recalls, s=100, c=[colors.get(a, "gray") for a in arms], alpha=0.8)
    for i, arm in enumerate(arms):
        ax3.annotate(arm, (bytes_kb[i], recalls[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax3.set_xlabel("State Size (KB)")
    ax3.set_ylabel("Avg Recall Loss")
    ax3.set_title("State Compression Efficiency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "proxy_laziness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_dir / 'proxy_laziness.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plot_proxy_results(args.results, args.out)
