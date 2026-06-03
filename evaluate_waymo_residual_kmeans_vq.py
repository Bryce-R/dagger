"""Evaluate residual k-means VQ codebooks for Waymo trajectories.

This is an oracle encoder for the RVQ representation: every residual stage picks
the nearest full-sequence residual code from a learned k-means codebook. It
isolates whether full-sequence residual codebooks can beat naive XY bins before
we try to learn the encoder/decoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_waymo_xy_binning_baseline import (
    fit_uniform_bins,
    quantize_xy_to_centers,
    split_dataset,
)
from train_waymo_accel_fsq import physical_trajectory_tensors
from train_waymo_mlp_fsq import extract_waymo_trajectories
from train_waymo_residual_fsq import write_csv
from train_waymo_residual_vq import (
    normalize_target,
    plot_prefix_reconstructions,
    target_to_xy,
    velocity_to_xy,
    xy_metrics,
)


def fit_kmeans(
    points: torch.Tensor,
    num_codes: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  if len(points) < num_codes:
    raise ValueError(f"num_codes={num_codes} exceeds number of points={len(points)}")
  centers = points[torch.randperm(len(points), generator=generator)[:num_codes]].clone()
  for _ in range(iterations):
    distances = torch.cdist(points, centers)
    assignments = distances.argmin(dim=1)
    next_centers = centers.clone()
    for code in range(num_codes):
      mask = assignments == code
      if mask.any():
        next_centers[code] = points[mask].mean(dim=0)
    if torch.allclose(next_centers, centers):
      break
    centers = next_centers
  return centers


def nearest_codes(
    residuals: torch.Tensor,
    codebook: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  indices = []
  values = []
  for start in range(0, len(residuals), batch_size):
    batch = residuals[start : start + batch_size]
    distances = torch.cdist(batch, codebook)
    batch_indices = distances.argmin(dim=1)
    indices.append(batch_indices)
    values.append(codebook[batch_indices])
  return torch.cat(values), torch.cat(indices)


def fit_residual_codebooks(
    train_target_norm: torch.Tensor,
    num_codes: int,
    num_tokens: int,
    kmeans_iters: int,
    seed: int,
) -> list[torch.Tensor]:
  flat_target = train_target_norm.reshape(len(train_target_norm), -1)
  residual = flat_target.clone()
  codebooks = []
  for token_index in range(num_tokens):
    codebook = fit_kmeans(
        residual,
        num_codes=num_codes,
        iterations=kmeans_iters,
        seed=seed + token_index,
    )
    encoded, _ = nearest_codes(residual, codebook, batch_size=2048)
    residual = residual - encoded
    codebooks.append(codebook)
    train_mse = float(residual.square().mean())
    print(f"kmeans_token={token_index + 1:02d} remaining_train_norm_mse={train_mse:.6f}")
  return codebooks


def encode_residual_vq(
    target_norm: torch.Tensor,
    codebooks: list[torch.Tensor],
    batch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
  flat_target = target_norm.reshape(len(target_norm), -1)
  residual = flat_target.clone()
  recon = torch.zeros_like(flat_target)
  prefix_recons = []
  index_history = []
  for codebook in codebooks:
    encoded, indices = nearest_codes(residual, codebook, batch_size=batch_size)
    recon = recon + encoded
    residual = flat_target - recon
    prefix_recons.append(recon.view_as(target_norm))
    index_history.append(indices)
  return prefix_recons, index_history


def evaluate_variant(
    target_space: str,
    train_target_norm: torch.Tensor,
    val_target_norm: torch.Tensor,
    val_xy: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, float | str]:
  print(f"Fitting residual k-means VQ target={target_space}")
  codebooks = fit_residual_codebooks(
      train_target_norm,
      num_codes=args.num_codes,
      num_tokens=args.num_tokens,
      kmeans_iters=args.kmeans_iters,
      seed=args.seed,
  )
  prefix_norm, indices = encode_residual_vq(val_target_norm, codebooks, args.batch_size)
  prefix_xy = [
      target_to_xy(prefix, target_space, target_mean, target_std)
      for prefix in prefix_norm
  ]
  metrics = xy_metrics(prefix_xy[-1], val_xy)
  per_example_mse = (prefix_xy[-1] - val_xy).square().mean(dim=(1, 2))
  first_indices = list(range(min(args.num_plot_samples, len(val_xy))))
  worst_indices = torch.topk(
      per_example_mse, k=min(args.num_plot_samples, len(val_xy))
  ).indices.tolist()
  plot_prefix_reconstructions(
      val_xy,
      prefix_xy,
      output_dir / f"{target_space}_kmeans_first_samples_prefix_recon.png",
      f"Residual k-means VQ {target_space}: first validation samples",
      first_indices,
  )
  plot_prefix_reconstructions(
      val_xy,
      prefix_xy,
      output_dir / f"{target_space}_kmeans_worst_samples_prefix_recon.png",
      f"Residual k-means VQ {target_space}: worst validation samples",
      worst_indices,
  )
  used_per_codebook = [
      int(torch.unique(token_indices).numel())
      for token_indices in indices
  ]
  return {
      "method": f"residual_kmeans_vq_{target_space}",
      "num_tokens": args.num_tokens,
      "num_codes": args.num_codes,
      "avg_used_codes": float(torch.tensor(used_per_codebook, dtype=torch.float32).mean()),
      "used_codes_by_token": " ".join(str(value) for value in used_per_codebook),
      **metrics,
  }


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate Waymo residual k-means VQ.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--target-spaces", nargs="+", choices=("xy", "velocity"), default=["xy", "velocity"])
  parser.add_argument("--num-codes", type=int, default=512)
  parser.add_argument("--num-tokens", type=int, default=10)
  parser.add_argument("--kmeans-iters", type=int, default=50)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--baseline-bins", type=int, default=71)
  parser.add_argument("--num-plot-samples", type=int, default=6)
  parser.add_argument("--output-dir", default="waymo_residual_kmeans_vq")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
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
  train_xy_norm, xy_mean, xy_std = normalize_target(train_xy)
  val_xy_norm = (val_xy - xy_mean) / xy_std
  train_velocity_norm, velocity_mean, velocity_std = normalize_target(train_velocity)
  val_velocity_norm = (val_velocity - velocity_mean) / velocity_std

  rows: list[dict[str, float | str]] = []
  min_xy, width_xy = fit_uniform_bins(train_xy, args.baseline_bins, per_timestep=False)
  baseline_recon = quantize_xy_to_centers(val_xy, min_xy, width_xy, args.baseline_bins)
  rows.append(
      {
          "method": f"naive_xy_bins_{args.baseline_bins}x{args.baseline_bins}",
          "num_tokens": args.num_steps,
          "num_codes": args.baseline_bins * args.baseline_bins,
          **xy_metrics(baseline_recon, val_xy),
      }
  )

  print(f"Parsed stats: {stats}")
  print(f"Split: train={len(train_xy)} val={len(val_xy)}")
  print(f"Baseline xy_mse={rows[-1]['xy_mse']:.6f}")

  if "xy" in args.target_spaces:
    rows.append(
        evaluate_variant(
            "xy",
            train_xy_norm,
            val_xy_norm,
            val_xy,
            xy_mean,
            xy_std,
            args,
            output_dir,
        )
    )
  if "velocity" in args.target_spaces:
    rows.append(
        evaluate_variant(
            "velocity",
            train_velocity_norm,
            val_velocity_norm,
            val_xy,
            velocity_mean,
            velocity_std,
            args,
            output_dir,
        )
    )

  write_csv(rows, output_dir / "metrics.csv")
  for row in rows:
    print(row)
  print(f"Saved metrics: {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
  main()
