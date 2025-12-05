import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from toy_dataset import CONTEXT_LEN, NUM_TRAJ, generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize sampled windows from the expert dataset")
    parser.add_argument(
        "--num-traj",
        type=int,
        default=NUM_TRAJ,
        help="Number of expert trajectories to generate before slicing (default: toy_dataset.NUM_TRAJ)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of random context windows to plot (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for selecting windows (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/dataset_overview.png",
        help="Path to save the dataset visualization (default: plots/dataset_overview.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    contexts, targets = generate_dataset(args.num_traj)
    num_contexts = contexts.shape[0]
    num_samples = min(args.num_samples, num_contexts)

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(num_contexts, size=num_samples, replace=False)

    t = np.arange(CONTEXT_LEN)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for idx in sample_idx:
        axes[0].plot(t, contexts[idx], alpha=0.7)
    axes[0].set_ylabel("state x")
    axes[0].set_title(f"Sampled context windows (n={num_samples})")
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(targets, bins=40, color="tab:orange", alpha=0.9)
    axes[1].set_xlabel("target action u")
    axes[1].set_ylabel("count")
    axes[1].set_title("Action distribution")

    fig.suptitle(
        f"Dataset overview: {num_contexts} windows from {args.num_traj} trajectories", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved dataset visualization to {output_path}")


if __name__ == "__main__":
    main()
