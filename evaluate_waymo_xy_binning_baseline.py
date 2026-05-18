"""Evaluate naive absolute-XY binning against Waymo MLP-FSQ reconstruction.

This baseline has no learned model. It extracts the same local Waymo trajectories
as the MLP-FSQ trainer, fits uniform x/y bins on the training split only, then
reconstructs validation positions with bin centers.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from train_waymo_mlp_fsq import (
    WAYMO_STEPS_PER_SECOND,
    TrajectoryMLPFSQ,
    absolute_position_loss,
    evaluate_reconstruction,
    extract_waymo_trajectories,
    max_abs_position_error_by_second,
    xy_to_positions,
)


def split_dataset(
    x: torch.Tensor,
    labels: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  generator = torch.Generator().manual_seed(seed)
  permutation = torch.randperm(len(x), generator=generator)
  val_size = int(round(len(x) * val_fraction))
  if val_fraction > 0.0:
    val_size = max(1, min(len(x) - 1, val_size))
  train_indices = permutation[val_size:]
  val_indices = permutation[:val_size]
  return x[train_indices], labels[train_indices], x[val_indices], labels[val_indices]


def physical_xy_positions(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    decode_absolute_positions: bool,
) -> torch.Tensor:
  physical = x * std + mean
  return xy_to_positions(physical, decode_absolute_positions)


def fit_uniform_bins(
    train_xy: torch.Tensor,
    num_bins: int,
    per_timestep: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  if per_timestep:
    min_xy = train_xy.amin(dim=0)
    max_xy = train_xy.amax(dim=0)
  else:
    min_xy = train_xy.reshape(-1, 2).amin(dim=0).view(1, 2)
    max_xy = train_xy.reshape(-1, 2).amax(dim=0).view(1, 2)
  width = (max_xy - min_xy).clamp_min(1e-6) / num_bins
  return min_xy, width


def quantize_xy_to_centers(
    xy: torch.Tensor,
    min_xy: torch.Tensor,
    width: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
  min_xy = min_xy.to(xy.device)
  width = width.to(xy.device)
  indices = torch.floor((xy - min_xy) / width).long().clamp(0, num_bins - 1)
  return min_xy + (indices.float() + 0.5) * width


def max_abs_xy_error_by_second(
    recon_xy: torch.Tensor,
    target_xy: torch.Tensor,
    steps_per_second: int = WAYMO_STEPS_PER_SECOND,
) -> dict[int, float]:
  num_steps = target_xy.shape[1]
  max_seconds = math.ceil(num_steps / steps_per_second)
  errors = {}
  abs_error = (recon_xy - target_xy).abs()
  for second in range(1, max_seconds + 1):
    index = min(second * steps_per_second - 1, num_steps - 1)
    errors[second] = float(abs_error[:, index].max())
  return errors


def p99_abs_xy_error_by_second(
    recon_xy: torch.Tensor,
    target_xy: torch.Tensor,
    steps_per_second: int = WAYMO_STEPS_PER_SECOND,
) -> dict[int, float]:
  num_steps = target_xy.shape[1]
  max_seconds = math.ceil(num_steps / steps_per_second)
  errors = {}
  abs_error = (recon_xy - target_xy).abs()
  for second in range(1, max_seconds + 1):
    index = min(second * steps_per_second - 1, num_steps - 1)
    errors[second] = float(torch.quantile(abs_error[:, index].reshape(-1), 0.99))
  return errors


def format_second_errors(errors: dict[int, float]) -> str:
  return " ".join(f"{second}s={error:.6f}" for second, error in errors.items())


def xy_metrics(recon_xy: torch.Tensor, target_xy: torch.Tensor) -> dict[str, float | str]:
  second_errors = max_abs_xy_error_by_second(recon_xy, target_xy)
  p99_second_errors = p99_abs_xy_error_by_second(recon_xy, target_xy)
  return {
      "pos_loss": float(F.mse_loss(recon_xy, target_xy)),
      "xy_mse": float(F.mse_loss(recon_xy, target_xy)),
      "xy_mae": float((recon_xy - target_xy).abs().mean()),
      "max_error_by_second": format_second_errors(second_errors),
      "p99_error_by_second": format_second_errors(p99_second_errors),
  }


def load_fsq_model(
    checkpoint_path: Path,
    input_dim: int,
    device: torch.device,
) -> tuple[TrajectoryMLPFSQ, dict, int | str]:
  checkpoint = torch.load(checkpoint_path, map_location=device)
  checkpoint_args = checkpoint["args"]
  model = TrajectoryMLPFSQ(
      num_steps=checkpoint_args["num_steps"],
      input_dim=input_dim,
      hidden_dim=checkpoint_args["hidden_dim"],
      code_dim=checkpoint_args["code_dim"],
      mlp_latent_tokens=checkpoint_args["mlp_latent_tokens"],
      mlp_dropout=checkpoint_args["mlp_dropout"],
      fsq_levels=checkpoint_args["fsq_levels"],
      fsq_input_scale=checkpoint_args["fsq_input_scale"],
  ).to(device)
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  return model, checkpoint_args, checkpoint.get("epoch", "unknown")


def evaluate_fsq_checkpoint(
    checkpoint_path: Path,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
    decode_absolute_positions: bool,
) -> list[dict[str, str | float]]:
  model, _, checkpoint_epoch = load_fsq_model(
      checkpoint_path, train_x.shape[-1], device
  )
  rows = []
  for split_name, split_x in [("train", train_x), ("val", val_x)]:
    metrics = evaluate_reconstruction(
        model=model,
        x=split_x,
        mean=mean,
        std=std,
        device=device,
        batch_size=batch_size,
        decode_absolute_positions=decode_absolute_positions,
    )
    second_errors = max_abs_position_error_by_second(
        model=model,
        x=split_x,
        mean=mean,
        std=std,
        device=device,
        batch_size=batch_size,
        decode_absolute_positions=decode_absolute_positions,
    )
    fsq_recon_xy, fsq_target_xy = reconstruct_xy_for_model(
        model=model,
        x=split_x,
        mean=mean,
        std=std,
        batch_size=batch_size,
        device=device,
        decode_absolute_positions=decode_absolute_positions,
    )
    p99_second_errors = p99_abs_xy_error_by_second(fsq_recon_xy, fsq_target_xy)
    rows.append(
        {
            "method": "mlp_fsq_best_val",
            "split": split_name,
            "num_bins": "",
            "per_timestep": "",
            "pos_loss": metrics["pos_loss"],
            "xy_mse": "",
            "xy_mae": "",
            "max_error_by_second": format_second_errors(second_errors),
            "p99_error_by_second": format_second_errors(p99_second_errors),
            "notes": f"checkpoint_epoch={checkpoint_epoch}",
        }
    )
  return rows


def reconstruct_xy_for_model(
    model: TrajectoryMLPFSQ,
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    batch_size: int,
    device: torch.device,
    decode_absolute_positions: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  recon_xy_parts = []
  target_xy_parts = []
  mean = mean.to(device)
  std = std.to(device)
  model.eval()
  with torch.no_grad():
    for start in range(0, len(x), batch_size):
      batch_x = x[start : start + batch_size].to(device)
      recon, _, _ = model(batch_x)
      recon_xy_parts.append(
          xy_to_positions(recon * std + mean, decode_absolute_positions).cpu()
      )
      target_xy_parts.append(
          xy_to_positions(batch_x * std + mean, decode_absolute_positions).cpu()
      )
  return torch.cat(recon_xy_parts), torch.cat(target_xy_parts)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate naive Waymo XY binning.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--include-all-states", action="store_true")
  parser.add_argument("--include-yaw", action="store_true")
  parser.add_argument("--decode-absolute-positions", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--xy-bins", type=int, nargs="+", default=[16, 32, 64, 128, 256])
  parser.add_argument(
      "--per-timestep",
      action="store_true",
      help="Fit separate x/y bin ranges at each future timestep.",
  )
  parser.add_argument("--output-csv", default="waymo_xy_binning_baseline.csv")
  parser.add_argument("--fsq-checkpoint", default=None)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  x, labels, mean, std, stats = extract_waymo_trajectories(
      tfrecord_paths=[Path(path) for path in args.tfrecord],
      num_steps=args.num_steps,
      max_trajectories=args.max_trajectories,
      include_all_valid_tracks=args.include_all_valid_tracks,
      object_types=set(args.object_type),
      include_yaw=args.include_yaw,
      include_all_states=args.include_all_states,
      decode_absolute_positions=args.decode_absolute_positions,
  )
  train_x, _, val_x, _ = split_dataset(x, labels, args.val_fraction, args.seed)
  train_xy = physical_xy_positions(train_x, mean, std, args.decode_absolute_positions)
  val_xy = physical_xy_positions(val_x, mean, std, args.decode_absolute_positions)

  print(f"Parsed stats: {stats}")
  print(
      f"Split: train={len(train_x)} val={len(val_x)} "
      f"val_fraction={args.val_fraction:.3f}"
  )

  rows: list[dict[str, str | float]] = []
  for num_bins in args.xy_bins:
    min_xy, width = fit_uniform_bins(train_xy, num_bins, args.per_timestep)
    for split_name, target_xy in [("train", train_xy), ("val", val_xy)]:
      recon_xy = quantize_xy_to_centers(target_xy, min_xy, width, num_bins)
      metrics = xy_metrics(recon_xy, target_xy)
      row = {
          "method": "xy_uniform_bins",
          "split": split_name,
          "num_bins": num_bins,
          "per_timestep": args.per_timestep,
          **metrics,
          "notes": "absolute_xy_centers",
      }
      rows.append(row)
      print(
          f"xy_bins={num_bins:04d} split={split_name} "
          f"pos_loss={metrics['pos_loss']:.6f} "
          f"xy_mse={metrics['xy_mse']:.6f} "
          f"xy_mae={metrics['xy_mae']:.6f} "
          f"max {metrics['max_error_by_second']} "
          f"p99 {metrics['p99_error_by_second']}"
      )

  if args.fsq_checkpoint:
    device = torch.device(args.device)
    fsq_rows = evaluate_fsq_checkpoint(
        checkpoint_path=Path(args.fsq_checkpoint),
        train_x=train_x,
        val_x=val_x,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        device=device,
        decode_absolute_positions=args.decode_absolute_positions,
    )
    rows.extend(fsq_rows)
    for row in fsq_rows:
      print(
          f"method={row['method']} split={row['split']} "
          f"pos_loss={float(row['pos_loss']):.6f} "
          f"max {row['max_error_by_second']} "
          f"p99 {row['p99_error_by_second']}"
      )

  output_csv = Path(args.output_csv)
  output_csv.parent.mkdir(parents=True, exist_ok=True)
  with output_csv.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
  main()
