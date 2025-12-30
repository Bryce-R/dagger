import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import StopMLPPolicy
from stop_dataset import (
    MAX_TIME,
    SIM_DT,
    STOP_SIGN_POSITION,
    StopDataset,
    generate_stop_dataset,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP policy on the stop-sign dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128)")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of samples for validation (default: 0.1)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Hidden layer sizes for the StopMLPPolicy (default: 128 128)",
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
    parser.add_argument("--rng-seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional path to save the trained checkpoint (default: None)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    states, actions, _ = generate_stop_dataset(
        stop_sign_position=args.stop_sign_position,
        dt=args.dt,
        max_time=args.max_time,
        measurement_noise=args.measurement_noise,
        rng_seed=args.rng_seed,
    )

    num_samples = states.shape[0]
    rng = np.random.default_rng(args.rng_seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    split_idx = int((1.0 - args.val_fraction) * num_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_ds = StopDataset(states[train_idx], actions[train_idx])
    val_ds = StopDataset(states[val_idx], actions[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = StopMLPPolicy(input_dim=states.shape[1], hidden_sizes=tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_states, batch_actions in train_loader:
            # print(f"Batch states: {batch_states.shape}")
            # print(f"Batch states: {batch_states[:2, :]}")
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            pred = model(batch_states)
            loss = criterion(pred, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_states.size(0)

        avg_train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                pred = model(batch_states)
                loss = criterion(pred, batch_actions)
                val_loss += loss.item() * batch_states.size(0)

        avg_val_loss = val_loss / len(val_ds)
        print(f"Epoch {epoch:03d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        model_cpu = model.to("cpu")
        torch.save(
            {
                "model_state_dict": model_cpu.state_dict(),
                "config": vars(args),
                "input_dim": states.shape[1],
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
