import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from model import StopMLPPolicy
from stop_dataset import (
    EXPERT_DECELERATIONS,
    MAX_TIME,
    SIM_DT,
    STOP_SIGN_POSITION,
    rollout_with_policy,
    simulate_stop_policy,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare learned MLP rollouts with all stop-sign experts.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to a checkpoint saved via train_stop.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/stop_rollout.png",
        help="Where to save the rollout comparison plot (default: plots/stop_rollout.png)",
    )
    parser.add_argument(
        "--stop-sign-position",
        type=float,
        default=STOP_SIGN_POSITION,
        help="Location of the stop sign in meters (default: 100.0)",
    )
    parser.add_argument("--dt", type=float, default=SIM_DT, help="Simulation step size (default: 0.1s)")
    parser.add_argument("--max-time", type=float, default=MAX_TIME, help="Maximum simulation time (default: 20s)")
    return parser.parse_args()


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    input_dim = int(checkpoint.get("input_dim", 5))
    hidden_dims = tuple(config.get("hidden_dims", [128, 128]))
    model = StopMLPPolicy(input_dim=input_dim, hidden_sizes=hidden_dims).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def learned_policy_fn(model: StopMLPPolicy):
    def policy(position, velocity, stop_pos, distance, prev_accel):
        state = torch.tensor([[position, velocity, stop_pos, distance, prev_accel]], dtype=torch.float32, device=device)
        with torch.no_grad():
            accel = model(state).cpu().item()
        return min(0.0, float(accel))

    return policy


def plot_rollouts(expert_trajs, learned_traj, output_path: Path, stop_sign_position: float):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    for traj in expert_trajs:
        times = traj["times"]
        axes[0].plot(times, traj["positions"], color="gray", alpha=0.35)
        axes[1].plot(times, traj["velocities"], color="gray", alpha=0.35)
        axes[2].step(times[:-1], traj["actions"], where="post", color="gray", alpha=0.35)

    axes[0].plot(learned_traj["times"], learned_traj["positions"], color="#0072B2", linewidth=2.0, label="Learned MLP")
    axes[1].plot(learned_traj["times"], learned_traj["velocities"], color="#0072B2", linewidth=2.0)
    axes[2].step(
        learned_traj["times"][:-1],
        learned_traj["actions"],
        where="post",
        color="#0072B2",
        linewidth=2.0,
        label="Learned MLP",
    )

    axes[0].axhline(stop_sign_position, color="k", linestyle="--", linewidth=1.0, label="Stop sign")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Position vs. time")
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_title("Velocity vs. time")
    axes[1].grid(True)

    axes[2].set_ylabel("Acceleration (m/s^2)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Braking sequences (negative = braking)")
    axes[2].legend(loc="lower right")
    axes[2].grid(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved rollout comparison to {output_path}")


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = load_model(ckpt_path)
    learned_policy = learned_policy_fn(model)
    learned_traj = rollout_with_policy(
        learned_policy,
        stop_sign_position=args.stop_sign_position,
        dt=args.dt,
        max_time=args.max_time,
    )

    expert_trajs = []
    for decel in EXPERT_DECELERATIONS:
        _, actions, positions, velocities, times = simulate_stop_policy(
            decel,
            stop_sign_position=args.stop_sign_position,
            dt=args.dt,
            max_time=args.max_time,
            measurement_noise=0.0,
        )
        expert_trajs.append(
            {
                "deceleration": decel,
                "positions": positions,
                "velocities": velocities,
                "actions": actions,
                "times": times,
            }
        )

    learned_stop_time = learned_traj["times"][-1]
    learned_final_pos = learned_traj["positions"][-1]
    print(f"Learned policy stopped at x={learned_final_pos:.2f} m (target {args.stop_sign_position:.2f})")
    print(f"Total time before rest: {learned_stop_time:.2f} s")

    output_path = Path(args.output)
    plot_rollouts(expert_trajs, learned_traj, output_path, args.stop_sign_position)


if __name__ == "__main__":
    main()
