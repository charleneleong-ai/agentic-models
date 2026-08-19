"""Visualize laziness proxy results — two panels showing KDA doesn't delay retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_laziness_proxies(results_path: Path, attn_path: Path, out_dir: Path) -> None:
    """Generate two-panel figure."""
    # Load fine-grained curves
    curves_data = [json.loads(l) for l in results_path.read_text().strip().split("\n") if l]

    # Load attention analysis
    attn_data = json.loads(attn_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        "all-mla": "#2196F3",
        "kda-1-1": "#4CAF50",
        "kda-3-1": "#FF9800",
        "kda-7-1": "#F44336",
        "all-kda": "#9C27B0",
    }

    # ─── Panel 1: Fine-grained Recall Convergence ───────────────────────────
    ax1 = axes[0]

    # Group by arm
    arm_curves: dict[str, list[dict[str, Any]]] = {}
    for row in curves_data:
        arm = row["arm"]
        if arm not in arm_curves:
            arm_curves[arm] = []
        for pt in row["curve"]:
            arm_curves[arm].append({"step": pt["step"], "recall": pt["recall"], "seed": row["seed"]})

    # Average across seeds
    for arm, pts in arm_curves.items():
        # Group by step
        by_step: dict[int, list[float]] = {}
        for pt in pts:
            s = pt["step"]
            if s not in by_step:
                by_step[s] = []
            by_step[s].append(pt["recall"])

        steps = sorted(by_step.keys())
        avg_recall = [np.mean(by_step[s]) for s in steps]
        std_recall = [np.std(by_step[s]) for s in steps]

        ax1.plot(steps, avg_recall, label=arm, color=colors.get(arm, "gray"), linewidth=2)
        ax1.fill_between(steps,
                        [a - s for a, s in zip(avg_recall, std_recall)],
                        [a + s for a, s in zip(avg_recall, std_recall)],
                        color=colors.get(arm, "gray"), alpha=0.15)

    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Recall Loss", fontsize=12)
    ax1.set_title("Recall Head Convergence\n(all arms converge similarly = no laziness)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Mark convergence zone
    ax1.axhline(y=3.8, color='gray', linestyle='--', alpha=0.5, label='convergence ~3.8')
    ax1.text(500, 3.82, 'convergence zone', fontsize=9, color='gray')

    # ─── Panel 2: Attention Mass Distribution ────────────────────────────────
    ax2 = axes[1]

    arms = list(attn_data.keys())
    filler_avgs = []
    non_filler_avgs = []

    for arm in arms:
        filler = attn_data[arm]["filler_attention"]
        non_filler = attn_data[arm]["non_filler_attention"]
        filler_avgs.append(np.mean(list(filler.values())))
        non_filler_avgs.append(np.mean(list(non_filler.values())))

    x = np.arange(len(arms))
    width = 0.35

    bars1 = ax2.bar(x - width/2, filler_avgs, width, label='Filler tokens', color='#FF6B6B', alpha=0.8)
    bars2 = ax2.bar(x + width/2, non_filler_avgs, width, label='Non-filler tokens', color='#4ECDC4', alpha=0.8)

    ax2.set_xlabel("Architecture", fontsize=12)
    ax2.set_ylabel("Average Attention Mass", fontsize=12)
    ax2.set_title("Attention to Filler Tokens\n(equal attention = no laziness)", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(arms, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "laziness_proxies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_dir / 'laziness_proxies.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--attn", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    plot_laziness_proxies(args.curves, args.attn, args.out)
