import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import TinyTransformerPolicy
from toy_dataset import (
    CONTEXT_LEN,
    DT,
    PROCESS_NOISE_STD,
    TEST_PERTURB_STD,
    U_MAX,
    device,
    expert_policy,
    rng,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare expert vs. learned rollouts")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to a TinyTransformerPolicy checkpoint saved via train.py",
    )
    parser.add_argument("--horizon", type=int, default=50, help="Rollout horizon (default: 50 steps)")
    parser.add_argument(
        "--x0",
        type=float,
        default=None,
        help="Optional initial state override; defaults to rng.normal(0, TEST_PERTURB_STD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rollout.png",
        help="Path to save the rollout comparison plot (default: rollout.png)",
    )
    return parser.parse_args()


def rollout_with_policy(policy_fn, x0, horizon=50):
    xs = [x0]
    us = []
    x = x0
    for _ in range(horizon):
        u = policy_fn(x, xs)
        us.append(u)
        noise = rng.normal(0.0, PROCESS_NOISE_STD)
        x = x + u * DT + noise
        xs.append(x)
    return np.array(xs), np.array(us)


def expert_policy_fn(x, _xs_history):
    return expert_policy(x)


def learned_policy_fn_factory(model, context_len=CONTEXT_LEN):
    def policy(x, xs_history):
        # xs_history includes current x at the end
        x_hist = np.array(xs_history, dtype=np.float32)
        if len(x_hist) < context_len:
            pad = np.zeros(context_len - len(x_hist), dtype=np.float32)
            x_in = np.concatenate([pad, x_hist])
        else:
            x_in = x_hist[-context_len:]
        x_tensor = torch.from_numpy(x_in).view(1, context_len, 1).to(device)
        with torch.no_grad():
            u_pred = model(x_tensor).cpu().item()
        u_pred = float(np.clip(u_pred, -U_MAX, U_MAX))
        return u_pred

    return policy


def load_model(checkpoint_path: Path) -> TinyTransformerPolicy:
    model = TinyTransformerPolicy().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model = load_model(ckpt_path)
    learned_policy_fn = learned_policy_fn_factory(model)

    x0 = args.x0 if args.x0 is not None else rng.normal(0.0, TEST_PERTURB_STD)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Test initial state x0: {x0:.4f}")

    expert_xs, expert_us = rollout_with_policy(expert_policy_fn, x0, horizon=args.horizon)
    learned_xs, learned_us = rollout_with_policy(learned_policy_fn, x0, horizon=args.horizon)

    print(f"Expert max |x|: {np.max(np.abs(expert_xs)):.4f}")
    print(f"Learned max |x|: {np.max(np.abs(learned_xs)):.4f}")
    print(f"Expert avg |u|: {np.mean(np.abs(expert_us)):.4f}")
    print(f"Learned avg |u|: {np.mean(np.abs(learned_us)):.4f}")

    timesteps = np.arange(args.horizon + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(timesteps, expert_xs, label="Expert", color="tab:blue")
    axes[0].plot(timesteps, learned_xs, label="Learned", color="tab:orange")
    axes[0].set_ylabel("state x")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    action_steps = np.arange(args.horizon)
    axes[1].plot(action_steps, expert_us, label="Expert", color="tab:blue")
    axes[1].plot(action_steps, learned_us, label="Learned", color="tab:orange")
    axes[1].set_ylabel("action u")
    axes[1].set_xlabel("timestep")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved rollout plot to {output_path}")


if __name__ == "__main__":
    main()
