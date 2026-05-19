"""Train an iterative residual FSQ tokenizer for Waymo trajectories.

Each token predicts a full-sequence residual correction. At evaluation time,
the tokenizer stops per example when reconstruction error falls below a meter
threshold, so simple trajectories can use fewer tokens.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate_waymo_xy_binning_baseline import split_dataset
from train_waymo_mlp_fsq import (
    WAYMO_STEPS_PER_SECOND,
    FiniteScalarQuantizer,
    extract_waymo_trajectories,
    xy_to_positions,
)


def physical_xy_positions(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
  return xy_to_positions(x * std + mean, decode_absolute_positions=True)


class ResidualFSQTokenizer(nn.Module):
  """Recurrent residual autoencoder with one FSQ token per correction step."""

  def __init__(
      self,
      num_steps: int,
      hidden_dim: int,
      code_dim: int,
      fsq_levels: list[int],
      fsq_input_scale: float,
      xy_mean: torch.Tensor,
      xy_std: torch.Tensor,
  ):
    super().__init__()
    self.num_steps = num_steps
    self.register_buffer("xy_mean", xy_mean.view(1, 1, 2))
    self.register_buffer("xy_std", xy_std.view(1, 1, 2).clamp_min(1e-6))
    input_dim = num_steps * 6
    self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, code_dim),
    )
    self.decoder = nn.Sequential(
        nn.Linear(code_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_steps * 2),
    )
    self.quantizer = FiniteScalarQuantizer(fsq_levels, code_dim, fsq_input_scale)

  def normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
    return (xy - self.xy_mean) / self.xy_std

  def correction_from_token(
      self,
      target_xy: torch.Tensor,
      recon_xy: torch.Tensor,
      quantize_strength: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_xy = target_xy - recon_xy
    features = torch.cat(
        [
            self.normalize_xy(target_xy),
            self.normalize_xy(recon_xy),
            residual_xy / self.xy_std,
        ],
        dim=-1,
    )
    latent = self.encoder(features.reshape(features.shape[0], -1))
    latent = latent.unsqueeze(-1)
    quantized, vq_loss, indices = self.quantizer(latent, quantize_strength)
    correction_norm = self.decoder(quantized.squeeze(-1)).view(
        target_xy.shape[0], self.num_steps, 2
    )
    correction_xy = correction_norm * self.xy_std
    return correction_xy, vq_loss, indices.squeeze(1)

  def forward(
      self,
      target_xy: torch.Tensor,
      max_tokens: int,
      quantize_strength: float = 1.0,
  ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    recon_xy = torch.zeros_like(target_xy)
    reconstructions = []
    index_history = []
    total_vq = target_xy.sum() * 0.0
    for _ in range(max_tokens):
      correction_xy, vq_loss, indices = self.correction_from_token(
          target_xy, recon_xy, quantize_strength
      )
      recon_xy = recon_xy + correction_xy
      reconstructions.append(recon_xy)
      index_history.append(indices)
      total_vq = total_vq + vq_loss
    return reconstructions, total_vq, index_history


def quantize_strength_for_epoch(
    epoch: int,
    warmup_epochs: int,
    anneal_epochs: int,
) -> float:
  if epoch <= warmup_epochs:
    return 0.0
  if anneal_epochs <= 0:
    return 1.0
  return min(1.0, (epoch - warmup_epochs) / anneal_epochs)


def reconstruction_loss(
    reconstructions: list[torch.Tensor],
    target_xy: torch.Tensor,
    loss_type: str,
    final_weight: float,
) -> torch.Tensor:
  losses = []
  for recon in reconstructions:
    if loss_type == "mse":
      losses.append(F.mse_loss(recon, target_xy))
    elif loss_type == "smooth_l1":
      losses.append(F.smooth_l1_loss(recon, target_xy))
    else:
      raise ValueError(f"unknown loss type: {loss_type}")
  if final_weight != 1.0:
    losses[-1] = losses[-1] * final_weight
  return torch.stack(losses).mean()


def stop_token_counts(
    reconstructions: list[torch.Tensor],
    target_xy: torch.Tensor,
    threshold: float,
    horizon_steps: int | None,
) -> torch.Tensor:
  if horizon_steps is None:
    horizon_steps = target_xy.shape[1]
  horizon_steps = max(1, min(horizon_steps, target_xy.shape[1]))
  counts = torch.full(
      (target_xy.shape[0],),
      len(reconstructions),
      dtype=torch.long,
      device=target_xy.device,
  )
  unresolved = torch.ones_like(counts, dtype=torch.bool)
  for token_index, recon in enumerate(reconstructions, start=1):
    max_error = (recon[:, :horizon_steps] - target_xy[:, :horizon_steps]).abs().amax(
        dim=(1, 2)
    )
    done = unresolved & (max_error <= threshold)
    counts[done] = token_index
    unresolved = unresolved & ~done
  return counts


def p99_errors_by_second(
    recon_xy: torch.Tensor,
    target_xy: torch.Tensor,
) -> dict[int, float]:
  abs_error = (recon_xy - target_xy).abs()
  max_seconds = math.ceil(target_xy.shape[1] / WAYMO_STEPS_PER_SECOND)
  errors = {}
  for second in range(1, max_seconds + 1):
    step = min(second * WAYMO_STEPS_PER_SECOND - 1, target_xy.shape[1] - 1)
    errors[second] = float(torch.quantile(abs_error[:, step].reshape(-1), 0.99))
  return errors


def max_first_seconds_p99(
    recon_xy: torch.Tensor,
    target_xy: torch.Tensor,
    seconds: int,
) -> float:
  steps = min(seconds * WAYMO_STEPS_PER_SECOND, target_xy.shape[1])
  abs_error = (recon_xy[:, :steps] - target_xy[:, :steps]).abs()
  values = [
      float(torch.quantile(abs_error[:, step].reshape(-1), 0.99))
      for step in range(steps)
  ]
  return max(values)


def format_second_errors(errors: dict[int, float]) -> str:
  return " ".join(f"{second}s={error:.6f}" for second, error in errors.items())


def evaluate_model(
    model: ResidualFSQTokenizer,
    xy: torch.Tensor,
    max_tokens: int,
    batch_size: int,
    threshold: float,
    stop_horizon_steps: int | None,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
  rows = []
  recon_parts = [[] for _ in range(max_tokens)]
  token_count_parts = []
  model.eval()
  with torch.no_grad():
    for (batch_xy,) in DataLoader(TensorDataset(xy), batch_size=batch_size):
      batch_xy = batch_xy.to(device)
      recons, _, _ = model(batch_xy, max_tokens=max_tokens)
      counts = stop_token_counts(recons, batch_xy, threshold, stop_horizon_steps)
      token_count_parts.append(counts.cpu())
      for token_index, recon in enumerate(recons):
        recon_parts[token_index].append(recon.cpu())
  target_xy = xy.cpu()
  token_counts = torch.cat(token_count_parts)
  for token_index in range(max_tokens):
    recon_xy = torch.cat(recon_parts[token_index])
    p99_by_second = p99_errors_by_second(recon_xy, target_xy)
    rows.append(
        {
            "tokens": token_index + 1,
            "xy_mse": float(F.mse_loss(recon_xy, target_xy)),
            "xy_mae": float((recon_xy - target_xy).abs().mean()),
            "max_error": float((recon_xy - target_xy).abs().max()),
            "p99_error": float(torch.quantile((recon_xy - target_xy).abs().reshape(-1), 0.99)),
            "max_first3s_p99": max_first_seconds_p99(recon_xy, target_xy, 3),
            "p99_by_second": format_second_errors(p99_by_second),
        }
    )
  rows.append(
      {
          "tokens": "stop",
          "xy_mse": "",
          "xy_mae": "",
          "max_error": "",
          "p99_error": "",
          "max_first3s_p99": "",
          "p99_by_second": "",
          "avg_stop_tokens": float(token_counts.float().mean()),
          "p50_stop_tokens": float(torch.quantile(token_counts.float(), 0.50)),
          "p90_stop_tokens": float(torch.quantile(token_counts.float(), 0.90)),
          "stop_success_rate": float((token_counts < max_tokens).float().mean()),
      }
  )
  return rows


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train Waymo residual FSQ tokenizer.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--include-all-states", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--epochs", type=int, default=1000)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--hidden-dim", type=int, default=256)
  parser.add_argument("--code-dim", type=int, default=8)
  parser.add_argument("--fsq-levels", type=int, nargs="+", default=[64])
  parser.add_argument("--fsq-input-scale", type=float, default=0.1)
  parser.add_argument("--max-tokens", type=int, default=4)
  parser.add_argument("--loss-type", choices=("smooth_l1", "mse"), default="smooth_l1")
  parser.add_argument("--final-loss-weight", type=float, default=1.0)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--quantize-warmup-epochs", type=int, default=0)
  parser.add_argument("--quantize-anneal-epochs", type=int, default=0)
  parser.add_argument("--stop-threshold", type=float, default=0.01)
  parser.add_argument(
      "--stop-horizon-steps",
      type=int,
      default=0,
      help="0 means use full sequence for the stop threshold.",
  )
  parser.add_argument("--log-every", type=int, default=100)
  parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
  parser.add_argument("--output-dir", default="waymo_residual_fsq")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  x, labels, mean, std, stats = extract_waymo_trajectories(
      tfrecord_paths=[Path(path) for path in args.tfrecord],
      num_steps=args.num_steps,
      max_trajectories=args.max_trajectories,
      include_all_valid_tracks=args.include_all_valid_tracks,
      object_types=set(args.object_type),
      include_yaw=False,
      include_all_states=args.include_all_states,
      decode_absolute_positions=True,
  )
  train_x, _, val_x, _ = split_dataset(x, labels, args.val_fraction, args.seed)
  train_xy = physical_xy_positions(train_x, mean, std)
  val_xy = physical_xy_positions(val_x, mean, std)
  xy_mean = train_xy.mean(dim=(0, 1))
  xy_std = train_xy.std(dim=(0, 1)).clamp_min(1e-6)
  print(f"Parsed stats: {stats}")
  print(f"Split: train={len(train_xy)} val={len(val_xy)}")

  model = ResidualFSQTokenizer(
      num_steps=args.num_steps,
      hidden_dim=args.hidden_dim,
      code_dim=args.code_dim,
      fsq_levels=args.fsq_levels,
      fsq_input_scale=args.fsq_input_scale,
      xy_mean=xy_mean,
      xy_std=xy_std,
  ).to(device)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )
  loader = DataLoader(
      TensorDataset(train_xy),
      batch_size=args.batch_size,
      shuffle=True,
      generator=torch.Generator().manual_seed(args.seed),
  )
  best_val_loss = float("inf")
  best_state = None
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  for epoch in range(1, args.epochs + 1):
    model.train()
    quantize_strength = quantize_strength_for_epoch(
        epoch, args.quantize_warmup_epochs, args.quantize_anneal_epochs
    )
    total_loss = 0.0
    batches = 0
    for (batch_xy,) in loader:
      batch_xy = batch_xy.to(device)
      recons, vq_loss, _ = model(
          batch_xy,
          max_tokens=args.max_tokens,
          quantize_strength=quantize_strength,
      )
      loss = reconstruction_loss(
          recons, batch_xy, args.loss_type, args.final_loss_weight
      ) + vq_loss
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      total_loss += float(loss.detach())
      batches += 1
    if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
      model.eval()
      with torch.no_grad():
        val_recons, _, _ = model(
            val_xy.to(device),
            max_tokens=args.max_tokens,
            quantize_strength=1.0,
        )
        val_loss = float(F.mse_loss(val_recons[-1].cpu(), val_xy))
        stop_counts = stop_token_counts(
            val_recons,
            val_xy.to(device),
            args.stop_threshold,
            args.stop_horizon_steps or None,
        )
      print(
          f"epoch={epoch:04d} train_loss={total_loss / batches:.6f} "
          f"val_final_mse={val_loss:.6f} "
          f"avg_stop_tokens={float(stop_counts.float().mean()):.3f} "
          f"stop_success_rate={float((stop_counts < args.max_tokens).float().mean()):.3f}"
      )
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

  if best_state is not None:
    model.load_state_dict(best_state)
  rows = evaluate_model(
      model,
      val_xy,
      max_tokens=args.max_tokens,
      batch_size=args.batch_size,
      threshold=args.stop_threshold,
      stop_horizon_steps=args.stop_horizon_steps or None,
      device=device,
  )
  for row in rows:
    print(row)
  csv_path = output_dir / "metrics.csv"
  write_csv(rows, csv_path)
  torch.save(
      {
          "model_state_dict": model.state_dict(),
          "args": vars(args),
          "xy_mean": xy_mean,
          "xy_std": xy_std,
      },
      output_dir / "best_model.pt",
  )
  print(f"Saved metrics: {csv_path}")
  print(f"Saved model: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
  main()
