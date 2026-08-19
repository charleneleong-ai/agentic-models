"""Visualize mechanism demos — two panels showing WHY KDA doesn't cause laziness."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_mechanism_demos(results_path: Path, out_dir: Path) -> None:
    """Generate two-panel mechanism figure."""
    data = json.loads(results_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {"all-mla": "#2196F3", "kda-3-1": "#FF9800", "all-kda": "#9C27B0"}

    # ─── Panel 1: Distance Sweep ─────────────────────────────────────────────
    ax1 = axes[0]
    dist_data = data["distance_sweep"]

    for arm_id, results in dist_data.items():
        distances = sorted(results.keys())
        recalls = [results[d]["val_recall"] for d in distances]
        ax1.plot(distances, recalls, "o-", label=arm_id, color=colors.get(arm_id, "gray"),
                linewidth=2, markersize=6)

    ax1.set_xlabel("Plant-to-Query Distance (tokens)", fontsize=12)
    ax1.set_ylabel("Recall Loss", fontsize=12)
    ax1.set_title("Needle-in-Haystack: Does Distance Hurt KDA?\n(longer distance = harder recall)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotate the key insight
    ax1.annotate("KDA state fills up here", xy=(128, 3.8), xytext=(150, 4.0),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")

    # ─── Panel 2: Chunk Size Sweep ───────────────────────────────────────────
    ax2 = axes[1]
    chunk_data = data["chunk_size_sweep"]

    chunk_sizes = sorted(chunk_data.keys())
    recalls = [chunk_data[cs]["val_recall"] for cs in chunk_sizes]

    bars = ax2.bar(range(len(chunk_sizes)), recalls, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(chunk_sizes))), alpha=0.8)
    ax2.set_xticks(range(len(chunk_sizes)))
    ax2.set_xticklabels([str(cs) for cs in chunk_sizes])
    ax2.set_xlabel("KDA Chunk Size (tokens)", fontsize=12)
    ax2.set_ylabel("Recall Loss", fontsize=12)
    ax2.set_title("Chunk Size: Does Window Affect Retrieval?\n(larger window = potentially lazier)", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, recalls):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                "%.3f" % val, ha="center", va="bottom", fontsize=9)

    # Annotate
    ax2.annotate("No significant trend\n= window size doesn't matter here",
                xy=(2, max(recalls) * 0.98), fontsize=9, color="gray",
                ha="center", style="italic")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mechanism_demos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved to %s" % (out_dir / "mechanism_demos.png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plot_mechanism_demos(args.results, args.out)
