"""Visualize follow-up results — cliff boundary and gradient profiling."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_followups(results_path: Path, out_dir: Path) -> None:
    """Generate two-panel follow-up figure."""
    data = json.loads(results_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ─── Panel 1: Chain Length Cliff ─────────────────────────────────────────
    ax1 = axes[0]
    chain_data = data["chain_len_sweep"]
    chain_lens = sorted(int(k) for k in chain_data.keys())
    recalls = [chain_data[str(cl)]["val_recall"] for cl in chain_lens]

    ax1.plot(chain_lens, recalls, "o-", color="#FF6B6B", linewidth=2, markersize=8)
    ax1.axhline(y=1.386, color="gray", linestyle="--", alpha=0.5, label="chance = ln(4)")
    ax1.axvline(x=3, color="orange", linestyle=":", alpha=0.7, label="cliff boundary")

    # Annotate
    for cl, r in zip(chain_lens, recalls):
        ax1.annotate("%.3f" % r, (cl, r), textcoords="offset points", xytext=(0, 10), fontsize=9)

    ax1.set_xlabel("Chain Length", fontsize=12)
    ax1.set_ylabel("Recall Loss", fontsize=12)
    ax1.set_title("Permutation Composition: Where is the Cliff?\n(chain_len=3 reveals the boundary)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotate the mechanism
    ax1.annotate("Memorization\nregime", xy=(1.5, 0.5), fontsize=10, color="green", ha="center")
    ax1.annotate("Algorithm\nrequired", xy=(6, 1.3), fontsize=10, color="red", ha="center")

    # ─── Panel 2: Gradient Profiling at 48L ─────────────────────────────────
    ax2 = axes[1]
    grad_data = data["gradient_profiling"]
    mixings = ["residual", "block", "full"]
    metrics = ["embed_grad_avg", "layer_grad_avg", "depth_grad_avg", "head_grad_avg"]
    metric_labels = ["Embed", "Layers", "Depth Mixer", "Head"]

    x = np.arange(len(mixings))
    width = 0.2
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [grad_data[m][metric] for m in mixings]
        ax2.bar(x + i * width, vals, width, label=label, color=colors[i], alpha=0.8)

    ax2.set_xlabel("Depth Mixing", fontsize=12)
    ax2.set_ylabel("Avg Gradient Norm", fontsize=12)
    ax2.set_title("Gradient Flow at 48 Layers\n(full AttnRes = vanishing gradients)", fontsize=13)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(mixings)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotate the key finding
    ax2.annotate("Full AttnRes:\ngradients vanish", xy=(2, grad_data["full"]["depth_grad_avg"]),
                xytext=(1.5, max(grad_data["full"].values()) * 0.8),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "followup_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved to %s" % (out_dir / "followup_results.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plot_followups(args.results, args.out)
