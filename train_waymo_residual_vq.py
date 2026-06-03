"""Train residual VQ tokenizers for Waymo trajectory reconstruction.

Two variants are supported:

* ``xy``: the residual VQ stack encodes the full local XY sequence.
* ``velocity``: the residual VQ stack encodes the full local VX/VY sequence,
  then reconstructs XY by integrating decoded velocity from the initial pose.

Each residual step uses its own vector-quantized codebook and MLP
encoder/decoder. Plots overlay GT, final reconstruction, and all prefix
reconstructions after codebook 1..N.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate_waymo_xy_binning_baseline import (
    fit_uniform_bins,
    quantize_xy_to_centers,
    split_dataset,
)
from train_waymo_accel_fsq import physical_trajectory_tensors
from train_waymo_mlp_fsq import WAYMO_STEPS_PER_SECOND, extract_waymo_trajectories
from train_waymo_residual_fsq import format_second_errors, p99_errors_by_second, write_csv


class VectorQuantizer(nn.Module):
  def __init__(self, num_codes: int, code_dim: int):
    super().__init__()
    self.codebook = nn.Embedding(num_codes, code_dim)
    nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

  def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distances = (
        latent.square().sum(dim=1, keepdim=True)
        - 2.0 * latent @ self.codebook.weight.t()
        + self.codebook.weight.square().sum(dim=1).view(1, -1)
    )
    indices = distances.argmin(dim=1)
    quantized = self.codebook(indices)
    codebook_loss = F.mse_loss(quantized, latent.detach())
    commitment_loss = F.mse_loss(latent, quantized.detach())
    quantized = latent + (quantized - latent).detach()
    return quantized, codebook_loss + 0.25 * commitment_loss, indices


class ResidualVQTokenizer(nn.Module):
  def __init__(
      self,
      num_steps: int,
      target_dim: int,
      hidden_dim: int,
      code_dim: int,
      num_codes: int,
      num_tokens: int,
  ):
    super().__init__()
    self.num_steps = num_steps
    self.target_dim = target_dim
    self.num_tokens = num_tokens
    input_dim = num_steps * target_dim * 3
    self.encoders = nn.ModuleList()
    self.quantizers = nn.ModuleList()
    self.decoders = nn.ModuleList()
    for _ in range(num_tokens):
      self.encoders.append(
          nn.Sequential(
              nn.Linear(input_dim, hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, code_dim),
          )
      )
      self.quantizers.append(VectorQuantizer(num_codes, code_dim))
      self.decoders.append(
          nn.Sequential(
              nn.Linear(code_dim, hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, hidden_dim),
              nn.ReLU(),
              nn.Linear(hidden_dim, num_steps * target_dim),
          )
      )

  def forward(
      self,
      target_norm: torch.Tensor,
      max_tokens: int | None = None,
  ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    if max_tokens is None:
      max_tokens = self.num_tokens
    recon = torch.zeros_like(target_norm)
    recons = []
    indices = []
    vq_loss = target_norm.sum() * 0.0
    for token_index in range(max_tokens):
      residual = target_norm - recon
      features = torch.cat([target_norm, recon, residual], dim=-1)
      latent = self.encoders[token_index](features.reshape(features.shape[0], -1))
      latent = torch.tanh(latent)
      quantized, token_vq_loss, token_indices = self.quantizers[token_index](latent)
      correction = self.decoders[token_index](quantized).view_as(target_norm)
      recon = recon + correction
      recons.append(recon)
      indices.append(token_indices)
      vq_loss = vq_loss + token_vq_loss
    return recons, vq_loss, indices


def velocity_to_xy(velocity: torch.Tensor) -> torch.Tensor:
  dt = 1.0 / WAYMO_STEPS_PER_SECOND
  xy = torch.zeros_like(velocity)
  for step in range(1, velocity.shape[1]):
    xy[:, step] = xy[:, step - 1] + 0.5 * (
        velocity[:, step - 1] + velocity[:, step]
    ) * dt
  return xy


def normalize_target(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  mean = target.mean(dim=(0, 1), keepdim=True)
  std = target.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
  return (target - mean) / std, mean, std


def denormalize(target_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
  return target_norm * std.to(target_norm.device) + mean.to(target_norm.device)


def target_to_xy(
    target_norm: torch.Tensor,
    target_space: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
  physical = denormalize(target_norm, mean, std)
  if target_space == "xy":
    xy = physical
    return xy - xy[:, :1]
  if target_space == "velocity":
    return velocity_to_xy(physical)
  raise ValueError(f"unknown target space: {target_space}")


def xy_metrics(recon_xy: torch.Tensor, target_xy: torch.Tensor) -> dict[str, float | str]:
  abs_error = (recon_xy - target_xy).abs()
  first_three_steps = min(3 * WAYMO_STEPS_PER_SECOND, target_xy.shape[1])
  max_first3s_p99 = max(
      float(torch.quantile(abs_error[:, step].reshape(-1), 0.99))
      for step in range(first_three_steps)
  )
  return {
      "xy_mse": float(F.mse_loss(recon_xy, target_xy)),
      "xy_mae": float(abs_error.mean()),
      "max_error": float(abs_error.max()),
      "p90_error": float(torch.quantile(abs_error.reshape(-1), 0.90)),
      "p99_error": float(torch.quantile(abs_error.reshape(-1), 0.99)),
      "max_first3s_p99": max_first3s_p99,
      "p99_by_second": format_second_errors(p99_errors_by_second(recon_xy, target_xy)),
  }


def residual_vq_loss(
    recons: list[torch.Tensor],
    target_norm: torch.Tensor,
    target_xy: torch.Tensor,
    target_space: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_loss_weight: float,
    xy_loss_weight: float,
) -> torch.Tensor:
  loss = target_norm.sum() * 0.0
  for recon in recons:
    if target_loss_weight > 0.0:
      loss = loss + float(target_loss_weight) * F.mse_loss(recon, target_norm)
    if xy_loss_weight > 0.0:
      recon_xy = target_to_xy(recon, target_space, mean, std)
      loss = loss + float(xy_loss_weight) * F.mse_loss(recon_xy, target_xy.to(recon.device))
  return loss / max(1, len(recons))


def evaluate_model(
    model: ResidualVQTokenizer,
    target_norm: torch.Tensor,
    target_xy: torch.Tensor,
    target_space: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, float | str]]:
  prefix_parts = [[] for _ in range(model.num_tokens)]
  model.eval()
  with torch.no_grad():
    loader = DataLoader(TensorDataset(target_norm), batch_size=batch_size)
    for (batch_target,) in loader:
      batch_target = batch_target.to(device)
      recons, _, _ = model(batch_target)
      for index, recon in enumerate(recons):
        prefix_parts[index].append(target_to_xy(recon.cpu(), target_space, mean, std))
  prefix_xy = [torch.cat(parts) for parts in prefix_parts]
  return prefix_xy, xy_metrics(prefix_xy[-1], target_xy)


def plot_prefix_reconstructions(
    target_xy: torch.Tensor,
    prefix_xy: list[torch.Tensor],
    output_path: Path,
    title: str,
    sample_indices: list[int],
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  output_path.parent.mkdir(parents=True, exist_ok=True)
  num_cols = min(3, len(sample_indices))
  num_rows = math.ceil(len(sample_indices) / num_cols)
  fig, axes = plt.subplots(
      num_rows,
      num_cols,
      figsize=(5.2 * num_cols, 4.8 * num_rows),
      squeeze=False,
  )
  cmap = plt.get_cmap("viridis")
  for axis, sample_index in zip(axes.ravel(), sample_indices):
    target = target_xy[sample_index]
    axis.plot(target[:, 0], target[:, 1], "o-", color="black", linewidth=2.2, label="GT")
    for prefix_index, recon_xy in enumerate(prefix_xy):
      color = cmap((prefix_index + 1) / len(prefix_xy))
      label = f"codebook {prefix_index + 1}" if prefix_index in (0, len(prefix_xy) - 1) else None
      alpha = 0.25 if prefix_index < len(prefix_xy) - 1 else 0.95
      linewidth = 1.0 if prefix_index < len(prefix_xy) - 1 else 2.0
      axis.plot(
          recon_xy[sample_index, :, 0],
          recon_xy[sample_index, :, 1],
          "x--",
          color=color,
          alpha=alpha,
          linewidth=linewidth,
          label=label,
      )
    final_error = float((prefix_xy[-1][sample_index] - target).square().mean())
    axis.set_title(f"sample={sample_index} final_mse={final_error:.4f}")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
  for axis in axes.ravel()[len(sample_indices):]:
    axis.axis("off")
  handles, labels = axes.ravel()[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=3)
  fig.suptitle(title, y=0.98)
  fig.tight_layout(rect=(0, 0.05, 1, 0.94))
  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def train_variant(
    target_space: str,
    train_target_norm: torch.Tensor,
    val_target_norm: torch.Tensor,
    train_xy: torch.Tensor,
    val_xy: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float | str]:
  model = ResidualVQTokenizer(
      num_steps=train_target_norm.shape[1],
      target_dim=train_target_norm.shape[2],
      hidden_dim=args.hidden_dim,
      code_dim=args.code_dim,
      num_codes=args.num_codes,
      num_tokens=args.num_tokens,
  ).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  loader = DataLoader(
      TensorDataset(train_target_norm, train_xy),
      batch_size=args.batch_size,
      shuffle=True,
      generator=torch.Generator().manual_seed(args.seed),
  )
  best_state = None
  best_val_mse = float("inf")
  for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch_target, batch_xy in loader:
      batch_target = batch_target.to(device)
      batch_xy = batch_xy.to(device)
      recons, vq_loss, _ = model(batch_target)
      loss = residual_vq_loss(
          recons,
          batch_target,
          batch_xy,
          target_space,
          target_mean.to(device),
          target_std.to(device),
          args.target_loss_weight,
          args.xy_loss_weight,
      ) + args.vq_loss_weight * vq_loss
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      if args.grad_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
      optimizer.step()
      total_loss += float(loss.detach())
      batches += 1
    if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
      prefix_xy, metrics = evaluate_model(
          model,
          val_target_norm,
          val_xy,
          target_space,
          target_mean,
          target_std,
          args.batch_size,
          device,
      )
      print(
          f"target={target_space} epoch={epoch:04d} "
          f"train_loss={total_loss / batches:.6f} "
          f"val_xy_mse={metrics['xy_mse']:.6f} "
          f"max_first3s_p99={metrics['max_first3s_p99']:.6f}"
      )
      if float(metrics["xy_mse"]) < best_val_mse:
        best_val_mse = float(metrics["xy_mse"])
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

  if best_state is not None:
    model.load_state_dict(best_state)
  prefix_xy, metrics = evaluate_model(
      model,
      val_target_norm,
      val_xy,
      target_space,
      target_mean,
      target_std,
      args.batch_size,
      device,
  )
  per_example_mse = (prefix_xy[-1] - val_xy).square().mean(dim=(1, 2))
  first_indices = list(range(min(args.num_plot_samples, len(val_xy))))
  worst_indices = torch.topk(
      per_example_mse, k=min(args.num_plot_samples, len(val_xy))
  ).indices.tolist()
  plot_prefix_reconstructions(
      val_xy,
      prefix_xy,
      output_dir / f"{target_space}_first_samples_prefix_recon.png",
      f"Residual VQ {target_space}: first validation samples",
      first_indices,
  )
  plot_prefix_reconstructions(
      val_xy,
      prefix_xy,
      output_dir / f"{target_space}_worst_samples_prefix_recon.png",
      f"Residual VQ {target_space}: worst validation samples",
      worst_indices,
  )
  torch.save(
      {
          "model_state_dict": model.state_dict(),
          "args": vars(args),
          "target_space": target_space,
          "target_mean": target_mean,
          "target_std": target_std,
      },
      output_dir / f"{target_space}_best_model.pt",
  )
  return {"method": f"residual_vq_{target_space}", **metrics}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train Waymo residual VQ baselines.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--target-spaces", nargs="+", choices=("xy", "velocity"), default=["xy", "velocity"])
  parser.add_argument("--epochs", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--hidden-dim", type=int, default=256)
  parser.add_argument("--code-dim", type=int, default=32)
  parser.add_argument("--num-codes", type=int, default=512)
  parser.add_argument("--num-tokens", type=int, default=10)
  parser.add_argument("--target-loss-weight", type=float, default=0.1)
  parser.add_argument("--xy-loss-weight", type=float, default=1.0)
  parser.add_argument("--vq-loss-weight", type=float, default=0.1)
  parser.add_argument("--grad-clip-norm", type=float, default=1.0)
  parser.add_argument("--baseline-bins", type=int, default=71)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--log-every", type=int, default=100)
  parser.add_argument("--num-plot-samples", type=int, default=6)
  parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
  parser.add_argument("--output-dir", default="waymo_residual_vq")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  x, labels, mean, std, stats = extract_waymo_trajectories(
      tfrecord_paths=[Path(path) for path in args.tfrecord],
      num_steps=args.num_steps,
      max_trajectories=args.max_trajectories,
      include_all_valid_tracks=args.include_all_valid_tracks,
      object_types=set(args.object_type),
      include_yaw=True,
      include_all_states=True,
      decode_absolute_positions=True,
  )
  train_x, _, val_x, _ = split_dataset(x, labels, args.val_fraction, args.seed)
  train_xy, train_velocity = physical_trajectory_tensors(train_x, mean, std)
  val_xy, val_velocity = physical_trajectory_tensors(val_x, mean, std)
  train_xy_target_norm, xy_mean, xy_std = normalize_target(train_xy)
  val_xy_target_norm = (val_xy - xy_mean) / xy_std
  train_velocity_norm, velocity_mean, velocity_std = normalize_target(train_velocity)
  val_velocity_norm = (val_velocity - velocity_mean) / velocity_std

  print(f"Parsed stats: {stats}")
  print(f"Split: train={len(train_x)} val={len(val_x)}")
  print(
      f"RVQ config: tokens={args.num_tokens} codebook_size={args.num_codes} "
      f"code_dim={args.code_dim}"
  )

  rows: list[dict[str, float | str]] = []
  min_xy, width_xy = fit_uniform_bins(train_xy, args.baseline_bins, per_timestep=False)
  baseline_recon = quantize_xy_to_centers(val_xy, min_xy, width_xy, args.baseline_bins)
  baseline_metrics = xy_metrics(baseline_recon, val_xy)
  baseline_row = {
      "method": f"naive_xy_bins_{args.baseline_bins}x{args.baseline_bins}",
      **baseline_metrics,
  }
  rows.append(baseline_row)
  print(
      f"baseline bins={args.baseline_bins} xy_mse={baseline_metrics['xy_mse']:.6f} "
      f"max_first3s_p99={baseline_metrics['max_first3s_p99']:.6f} "
      f"p99={baseline_metrics['p99_by_second']}"
  )

  if "xy" in args.target_spaces:
    rows.append(
        train_variant(
            "xy",
            train_xy_target_norm,
            val_xy_target_norm,
            train_xy,
            val_xy,
            xy_mean,
            xy_std,
            args,
            device,
            output_dir,
        )
    )
  if "velocity" in args.target_spaces:
    rows.append(
        train_variant(
            "velocity",
            train_velocity_norm,
            val_velocity_norm,
            train_xy,
            val_xy,
            velocity_mean,
            velocity_std,
            args,
            device,
            output_dir,
        )
    )

  write_csv(rows, output_dir / "metrics.csv")
  print(f"Saved metrics: {output_dir / 'metrics.csv'}")
  for row in rows:
    print(row)


if __name__ == "__main__":
  main()
