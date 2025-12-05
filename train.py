import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import TinyTransformerPolicy
from toy_dataset import ImitationDataset, NUM_TRAJ, device, generate_dataset, rng


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TinyTransformerPolicy on the toy dataset.")
    parser.add_argument(
        "--num-traj",
        type=int,
        default=NUM_TRAJ,
        help="Number of expert trajectories to generate for training (default from toy_dataset.NUM_TRAJ)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64)")
    parser.add_argument(
        "--val-batch-size", type=int, default=256, help="Validation batch size for evaluation (default: 256)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional path to save the trained model checkpoint (e.g., checkpoints/policy.pt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- Generate data ----------
    contexts, targets = generate_dataset(args.num_traj)

    # Split train/val
    num_samples = contexts.shape[0]
    idx = np.arange(num_samples)
    rng.shuffle(idx)
    split = int(0.9 * num_samples)
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = ImitationDataset(contexts[train_idx], targets[train_idx])
    val_ds = ImitationDataset(contexts[val_idx], targets[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)

    # ---------- Model ----------
    model = TinyTransformerPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # ---------- Training ----------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x_seq, u in train_loader:
            x_seq = x_seq.to(device)  # (B, T, 1)
            u = u.to(device)  # (B,)

            pred_u = model(x_seq)  # (B,)
            loss = criterion(pred_u, u)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_seq.size(0)

        avg_train_loss = total_loss / len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, u in val_loader:
                x_seq = x_seq.to(device)
                u = u.to(device)
                pred_u = model(x_seq)
                loss = criterion(pred_u, u)
                val_loss += loss.item() * x_seq.size(0)

        avg_val_loss = val_loss / len(val_ds)
        print(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # Always move to CPU for portable checkpoints
        model_cpu = model.to("cpu")
        torch.save(
            {
                "model_state_dict": model_cpu.state_dict(),
                "epochs": args.epochs,
                "num_traj": args.num_traj,
                "batch_size": args.batch_size,
                "val_batch_size": args.val_batch_size,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
