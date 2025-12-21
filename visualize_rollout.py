import argparse
from collections import deque
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
from train_latent_bc import LatentPolicy


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
    parser.add_argument(
        "--model-type",
        choices=("transformer", "latent"),
        default="transformer",
        help="Checkpoint type: transformer (train.py) or latent (train_latent_bc.py)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Optional override for latent checkpoints; defaults to value stored in checkpoint",
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_latent_model(
    checkpoint_path: Path,
    device: torch.device,
    history_override: int | None = None,
) -> tuple[LatentPolicy, dict, int, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(
            "Latent checkpoint missing config; please retrain via train_latent_bc.py "
            "which saves the full args dictionary.",
        )
    context_len = int(config.get("context_len", CONTEXT_LEN))
    history_len = int(history_override if history_override is not None else config.get("history_length", 8))
    if history_len < 1:
        raise ValueError("history-length must be >= 1 for latent visualization.")
    latent_type = config.get("latent_type", "continuous")
    latent_dim = int(config.get("latent_dim", 8))
    num_categories = int(config.get("num_latent_categories", 4))
    encoder_hidden = int(config.get("encoder_hidden", 64))
    policy_hidden = int(config.get("policy_hidden", 128))
    action_dim = int(config.get("action_dim", 1))
    hist_feat_dim = context_len + action_dim

    model = LatentPolicy(
        obs_dim=context_len,
        hist_feat_dim=hist_feat_dim,
        latent_type=latent_type,
        latent_dim=latent_dim,
        num_categories=num_categories,
        encoder_hidden=encoder_hidden,
        policy_hidden=policy_hidden,
        action_dim=action_dim,
    ).to(device)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise ValueError("Latent checkpoint missing 'model_state'.")
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, history_len, context_len, action_dim


def latent_policy_fn_factory(
    model: LatentPolicy,
    context_len: int,
    history_len: int,
    action_dim: int,
    device: torch.device,
):
    history_dim = context_len + action_dim
    history_deque = deque(
        [np.zeros(history_dim, dtype=np.float32) for _ in range(history_len)],
        maxlen=history_len,
    )

    def policy(x, xs_history):
        x_hist = np.array(xs_history, dtype=np.float32)
        if len(x_hist) < context_len:
            pad = np.zeros(context_len - len(x_hist), dtype=np.float32)
            obs_vec = np.concatenate([pad, x_hist])
        else:
            obs_vec = x_hist[-context_len:]

        history_arr = np.stack(history_deque, axis=0)
        obs_tensor = torch.from_numpy(obs_vec).unsqueeze(0).to(device)
        history_tensor = torch.from_numpy(history_arr).unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor, _ = model(obs_tensor, history_tensor, training=False)
        action = float(action_tensor.squeeze().cpu().item())
        action = float(np.clip(action, -U_MAX, U_MAX))
        action_vec = np.full((action_dim,), action, dtype=np.float32)
        history_pair = np.concatenate((obs_vec.astype(np.float32), action_vec))
        history_deque.append(history_pair)
        return action

    return policy


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    if args.model_type == "transformer":
        model = load_model(ckpt_path)
        learned_policy_fn = learned_policy_fn_factory(model)
    else:
        latent_model, config, history_len, context_len, action_dim = load_latent_model(
            ckpt_path,
            device,
            args.history_length,
        )
        learned_policy_fn = latent_policy_fn_factory(
            latent_model,
            context_len=context_len,
            history_len=history_len,
            action_dim=action_dim,
            device=device,
        )
        print(
            f"Loaded latent policy: context_len={context_len}, history_len={history_len}, "
            f"type={config.get('latent_type')}",
        )

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
