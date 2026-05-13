"""Train a tiny VQ-VAE on synthetic straight/left/right trajectories.

This script is intentionally self-contained. It builds three base trajectory
families, adds noise to make a small dataset, and trains a temporal VQ-VAE that
reconstructs 2D relative trajectories through a discrete codebook.

Example:
  python tutorial/train_dummy_trajectory_vqvae.py --epochs 200
  python tutorial/train_dummy_trajectory_vqvae.py --device mps
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SyntheticTrajectoryConfig:
  num_per_class: int = 1000
  num_steps: int = 50
  noise_std: float = 0.06
  dt: float = 0.1


def make_base_trajectory(kind: str, num_steps: int, dt: float) -> torch.Tensor:
  """Returns one [T, 2] trajectory in an ego-centric local frame."""
  t = torch.arange(num_steps, dtype=torch.float32) * dt

  if kind == "straight":
    x = 6.0 * t
    y = torch.zeros_like(t)
  elif kind == "left":
    theta = torch.linspace(0.0, math.pi / 2.2, num_steps)
    radius = 12.0
    x = radius * torch.sin(theta)
    y = radius * (1.0 - torch.cos(theta))
  elif kind == "right":
    theta = torch.linspace(0.0, math.pi / 2.2, num_steps)
    radius = 12.0
    x = radius * torch.sin(theta)
    y = -radius * (1.0 - torch.cos(theta))
  else:
    raise ValueError(f"Unknown trajectory kind: {kind}")

  return torch.stack([x, y], dim=-1)


def make_noisy_trajectory_dataset(
    config: SyntheticTrajectoryConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Builds normalized displacement trajectories and integer class labels."""
  trajectories = []
  labels = []
  kinds = ("straight", "left", "right")

  for label, kind in enumerate(kinds):
    base = make_base_trajectory(kind, config.num_steps, config.dt)
    for _ in range(config.num_per_class):
      speed_scale = torch.empty(1).uniform_(0.85, 1.15)
      lateral_bias = torch.empty(1).normal_(0.0, config.noise_std)
      noise = torch.randn_like(base) * config.noise_std
      noisy = base * speed_scale + noise
      noisy[:, 1] += lateral_bias
      noisy = noisy - noisy[:1]

      # Learn motion increments instead of absolute positions. This is usually
      # easier for trajectory models and mirrors common forecasting pipelines.
      deltas = torch.zeros_like(noisy)
      deltas[1:] = noisy[1:] - noisy[:-1]
      trajectories.append(deltas)
      labels.append(label)

  x = torch.stack(trajectories)
  y = torch.tensor(labels, dtype=torch.long)

  mean = x.mean(dim=(0, 1), keepdim=True)
  std = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
  x = (x - mean) / std
  return x, y, mean, std


class VectorQuantizer(nn.Module):
  """Nearest-neighbor vector quantizer with straight-through gradients."""

  def __init__(self, num_codes: int, code_dim: int, commitment_weight: float):
    super().__init__()
    self.codebook = nn.Embedding(num_codes, code_dim)
    self.commitment_weight = commitment_weight
    self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

  def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # z: [B, C, T] -> flatten vectors as [B*T, C].
    z_bt = z.permute(0, 2, 1).contiguous()
    flat_z = z_bt.view(-1, z_bt.shape[-1])

    distances = (
        flat_z.pow(2).sum(dim=1, keepdim=True)
        - 2.0 * flat_z @ self.codebook.weight.t()
        + self.codebook.weight.pow(2).sum(dim=1)
    )
    indices = distances.argmin(dim=1)
    quantized = self.codebook(indices).view_as(z_bt)

    codebook_loss = F.mse_loss(quantized, z_bt.detach())
    commitment_loss = F.mse_loss(z_bt, quantized.detach())
    vq_loss = codebook_loss + self.commitment_weight * commitment_loss

    quantized = z_bt + (quantized - z_bt).detach()
    quantized = quantized.permute(0, 2, 1).contiguous()
    indices = indices.view(z.shape[0], z.shape[2])
    return quantized, vq_loss, indices


class TrajectoryVQVAE(nn.Module):
  """Small temporal convolutional VQ-VAE for [B, T, 2] trajectories."""

  def __init__(
      self,
      input_dim: int = 2,
      hidden_dim: int = 64,
      code_dim: int = 32,
      num_codes: int = 32,
      commitment_weight: float = 0.25,
  ):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, code_dim, kernel_size=3, padding=1),
    )
    self.quantizer = VectorQuantizer(num_codes, code_dim, commitment_weight)
    self.decoder = nn.Sequential(
        nn.Conv1d(code_dim, hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose1d(
            hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1
        ),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, input_dim, kernel_size=5, padding=2),
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = self.encoder(x.permute(0, 2, 1))
    quantized, vq_loss, indices = self.quantizer(z)
    recon = self.decoder(quantized).permute(0, 2, 1)
    recon = recon[:, : x.shape[1]]
    return recon, vq_loss, indices


def codebook_perplexity(indices: torch.Tensor, num_codes: int) -> float:
  counts = torch.bincount(indices.reshape(-1), minlength=num_codes).float()
  probs = counts / counts.sum().clamp_min(1.0)
  entropy = -(probs * (probs + 1e-10).log()).sum()
  return float(torch.exp(entropy))


def deltas_to_positions(deltas: torch.Tensor) -> torch.Tensor:
  return torch.cumsum(deltas, dim=1)


def save_reconstruction_plot(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    epoch: int,
) -> None:
  """Saves target/reconstruction plots for one straight, left, and right sample."""
  import matplotlib.pyplot as plt

  model.eval()
  output_dir.mkdir(parents=True, exist_ok=True)
  class_names = ("straight", "left", "right")
  sample_indices = [int((labels == class_id).nonzero()[0]) for class_id in range(3)]
  sample_x = x[sample_indices].to(device)

  with torch.no_grad():
    recon, _, token_indices = model(sample_x)

  mean = mean.to(device)
  std = std.to(device)
  target_deltas = sample_x * std + mean
  recon_deltas = recon * std + mean
  target_xy = deltas_to_positions(target_deltas).cpu()
  recon_xy = deltas_to_positions(recon_deltas).cpu()

  fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
  for axis, class_name, target, pred, tokens in zip(
      axes, class_names, target_xy, recon_xy, token_indices.cpu()
  ):
    axis.plot(target[:, 0], target[:, 1], label="target", linewidth=2)
    axis.plot(pred[:, 0], pred[:, 1], label="recon", linewidth=2, linestyle="--")
    axis.scatter(target[0, 0], target[0, 1], s=20, marker="o", color="black")
    axis.set_title(f"{class_name}: {len(torch.unique(tokens))} codes")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
  axes[0].legend(loc="best")
  fig.suptitle(f"Trajectory VQ-VAE reconstruction, epoch {epoch}")
  fig.tight_layout()
  fig.savefig(output_dir / f"reconstruction_epoch_{epoch:04d}.png", dpi=150)
  plt.close(fig)


def select_device(device_arg: str) -> torch.device:
  """Selects a training device, preferring Apple MPS in auto mode."""
  if device_arg == "auto":
    if torch.backends.mps.is_available():
      return torch.device("mps")
    if torch.cuda.is_available():
      return torch.device("cuda")
    return torch.device("cpu")

  if device_arg == "mps" and not torch.backends.mps.is_available():
    raise RuntimeError(
        "MPS was requested, but torch.backends.mps.is_available() is false."
    )
  if device_arg == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
  return torch.device(device_arg)


def train(args: argparse.Namespace) -> None:
  torch.manual_seed(args.seed)
  device = select_device(args.device)

  config = SyntheticTrajectoryConfig(
      num_per_class=args.num_per_class,
      num_steps=args.num_steps,
      noise_std=args.noise_std,
  )
  x, labels, mean, std = make_noisy_trajectory_dataset(config)
  dataset = TensorDataset(x, labels)
  loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

  model = TrajectoryVQVAE(
      hidden_dim=args.hidden_dim,
      code_dim=args.code_dim,
      num_codes=args.num_codes,
      commitment_weight=args.commitment_weight,
  ).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

  print(f"Training on {len(dataset)} synthetic trajectories with device={device}")
  for epoch in range(1, args.epochs + 1):
    model.train()
    total_recon = 0.0
    total_vq = 0.0
    total_batches = 0
    last_indices = None

    for batch_x, _ in loader:
      batch_x = batch_x.to(device)
      recon, vq_loss, indices = model(batch_x)
      recon_loss = F.smooth_l1_loss(recon, batch_x)
      loss = recon_loss + vq_loss

      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      total_recon += float(recon_loss.detach())
      total_vq += float(vq_loss.detach())
      total_batches += 1
      last_indices = indices.detach().cpu()

    if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
      perplexity = codebook_perplexity(last_indices, args.num_codes)
      print(
          f"epoch={epoch:04d} "
          f"recon={total_recon / total_batches:.5f} "
          f"vq={total_vq / total_batches:.5f} "
          f"last_batch_perplexity={perplexity:.2f}"
      )

    if args.plot_every > 0 and (
        epoch == 1 or epoch % args.plot_every == 0 or epoch == args.epochs
    ):
      save_reconstruction_plot(
          model=model,
          x=x,
          labels=labels,
          mean=mean,
          std=std,
          device=device,
          output_dir=Path(args.plot_dir),
          epoch=epoch,
      )

  model.eval()
  with torch.no_grad():
    sample_x = x[:9].to(device)
    recon, _, indices = model(sample_x)
    mse = F.mse_loss(recon, sample_x).item()
    used_codes = torch.unique(indices.cpu()).tolist()

  print(f"Final sample MSE: {mse:.5f}")
  print(f"Codes used by first 9 samples: {used_codes}")
  print("First sample discrete tokens:")
  print(indices[0].cpu().tolist())


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Train a VQ-VAE on noisy straight/left/right dummy trajectories."
  )
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--num-per-class", type=int, default=1000)
  parser.add_argument("--num-steps", type=int, default=50)
  parser.add_argument("--noise-std", type=float, default=0.06)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--hidden-dim", type=int, default=64)
  parser.add_argument("--code-dim", type=int, default=32)
  parser.add_argument("--num-codes", type=int, default=32)
  parser.add_argument("--commitment-weight", type=float, default=0.25)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--log-every", type=int, default=10)
  parser.add_argument(
      "--plot-every",
      type=int,
      default=10,
      help="Save target/reconstruction plots every N epochs. Use 0 to disable.",
  )
  parser.add_argument("--plot-dir", default="trajectory_vqvae_plots")
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument(
      "--device",
      choices=("auto", "mps", "cuda", "cpu"),
      default="auto",
      help="Training device. auto prefers MPS, then CUDA, then CPU.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  train(parse_args())
