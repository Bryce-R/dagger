import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from stop_dataset import (
    EXPERT_DECELERATIONS,
    MAX_TIME,
    SIM_DT,
    STOP_SIGN_POSITION,
    generate_stop_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize sample trajectories from the stop-sign dataset.")
    parser.add_argument("--num-policies", type=int, default=5, help="Number of expert braking profiles to plot (default: 5)")
    parser.add_argument(
        "--output",
        type=str,
        default="plots/stop_dataset.png",
        help="Path to save the dataset visualization (default: plots/stop_dataset.png)",
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=0.0,
        help="Std. dev. of Gaussian noise added to position/velocity/distance (default: 0.0)",
    )
    parser.add_argument(
        "--stop-sign-position",
        type=float,
        default=STOP_SIGN_POSITION,
        help="Location of the stop sign in meters (default: 100.0)",
    )
    parser.add_argument("--dt", type=float, default=SIM_DT, help="Simulation step size (default: 0.1s)")
    parser.add_argument("--max-time", type=float, default=MAX_TIME, help="Maximum simulation time (default: 20s)")
    parser.add_argument("--rng-seed", type=int, default=123, help="Random seed for noise (default: 123)")
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, trajectories = generate_stop_dataset(
        stop_sign_position=args.stop_sign_position,
        dt=args.dt,
        max_time=args.max_time,
        measurement_noise=args.measurement_noise,
        rng_seed=args.rng_seed,
    )

    total_policies = len(EXPERT_DECELERATIONS)
    num_to_plot = min(args.num_policies, total_policies)
    indices = np.linspace(0, total_policies - 1, num_to_plot, dtype=int)
    selected = [trajectories[i] for i in indices]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for traj in selected:
        times = traj["times"]
        axes[0].plot(times, traj["positions"], label=f"a={traj['deceleration']:.1f} m/s^2")
        axes[1].plot(times, traj["velocities"], label=f"a={traj['deceleration']:.1f} m/s^2")

    axes[0].axhline(args.stop_sign_position, color="k", linestyle="--", linewidth=1.0, label="stop sign")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Expert trajectories toward the stop sign")
    axes[0].legend(loc="lower right")

    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Velocities under different braking strengths")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved dataset visualization to {output_path}")


if __name__ == "__main__":
    main()
