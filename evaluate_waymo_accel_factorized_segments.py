"""Evaluate small-vocab factorized acceleration segment tokens.

The token structure is classification-friendly without a flat Cartesian product:

  primitive_id: one of K learned acceleration primitives
  duration_id:  one of the possible segment durations

The decoder holds acceleration constant for the duration, conditioned on the
current decoded position/velocity. This tests whether a small primitive vocab
plus a separate duration head can preserve most of the compression benefit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate_waymo_accel_horizon_metrics import horizon_metrics
from evaluate_waymo_accel_segment_tokens import (
    decode_constant_accel_segment,
    run_length_encode_ids,
    segment_token_string,
)
from evaluate_waymo_xy_binning_baseline import split_dataset
from train_waymo_accel_fsq import (
    physical_trajectory_tensors,
    velocity_to_acceleration,
    xy_metrics,
)
from train_waymo_mlp_fsq import extract_waymo_trajectories
from train_waymo_residual_fsq import write_csv


def fit_kmeans(
    points: torch.Tensor,
    num_clusters: int,
    iterations: int,
    seed: int,
) -> torch.Tensor:
  generator = torch.Generator().manual_seed(seed)
  initial = torch.randperm(len(points), generator=generator)[:num_clusters]
  centers = points[initial].clone()
  for _ in range(iterations):
    distances = torch.cdist(points, centers)
    assignments = distances.argmin(dim=1)
    next_centers = centers.clone()
    for cluster in range(num_clusters):
      mask = assignments == cluster
      if mask.any():
        next_centers[cluster] = points[mask].mean(dim=0)
    if torch.allclose(next_centers, centers):
      break
    centers = next_centers
  return centers


def nearest_codebook(
    accel: torch.Tensor,
    codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  flat = accel.reshape(-1, 2)
  distances = torch.cdist(flat, codebook)
  ids = distances.argmin(dim=1)
  centers = codebook[ids].view_as(accel)
  return centers, ids.view(accel.shape[:-1])


def greedy_codebook_segments(
    target_xy: torch.Tensor,
    target_velocity: torch.Tensor,
    target_accel: torch.Tensor,
    codebook: torch.Tensor,
    xy_threshold: float,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
  num_steps = target_xy.shape[0]
  recon_xy = torch.empty_like(target_xy)
  recon_xy[0] = target_xy[0]
  tokens: list[tuple[int, int]] = []
  start = 1
  current_xy = target_xy[0].clone()
  current_velocity = target_velocity[0].clone()
  while start < num_steps:
    best = None
    for end in range(num_steps - 1, start - 1, -1):
      segment_accel = target_accel[start : end + 1].mean(dim=0)
      accel_center, primitive_id = nearest_codebook(segment_accel.view(1, 2), codebook)
      candidate_xy, candidate_velocity = decode_constant_accel_segment(
          current_xy,
          current_velocity,
          accel_center[0],
          end - start + 2,
      )
      max_error = float((candidate_xy[1:] - target_xy[start : end + 1]).abs().max())
      if max_error <= xy_threshold or end == start:
        best = (end, int(primitive_id[0]), candidate_xy, candidate_velocity)
        break
    assert best is not None
    end, primitive_id, candidate_xy, candidate_velocity = best
    recon_xy[start : end + 1] = candidate_xy[1:]
    tokens.append((primitive_id, end - start + 1))
    current_xy = candidate_xy[-1]
    current_velocity = candidate_velocity
    start = end + 1
  return recon_xy, tokens


def evaluate_codebook(
    val_xy: torch.Tensor,
    val_velocity: torch.Tensor,
    val_accel: torch.Tensor,
    codebook: torch.Tensor,
    xy_threshold: float,
    method: str,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], list[list[tuple[int, int]]], torch.Tensor]:
  recon_parts = []
  token_sequences = []
  for index in range(len(val_xy)):
    recon_xy, tokens = greedy_codebook_segments(
        val_xy[index],
        val_velocity[index],
        val_accel[index],
        codebook,
        xy_threshold,
    )
    recon_parts.append(recon_xy)
    token_sequences.append(tokens)
  recon_xy = torch.stack(recon_parts)
  token_counts = torch.tensor([len(tokens) for tokens in token_sequences], dtype=torch.float32)
  max_duration = val_xy.shape[1] - 1
  metrics = xy_metrics(recon_xy, val_xy)
  row = {
      "method": method,
      "primitive_vocab": len(codebook),
      "duration_vocab": max_duration,
      "factorized_logits": len(codebook) + max_duration,
      "flat_pair_vocab": len(codebook) * max_duration,
      "xy_threshold": xy_threshold,
      "fixed_interval_tokens_before": max_duration,
      "avg_segments_after": float(token_counts.mean()),
      "avg_two_token_factorized_length": float(token_counts.mean() * 2.0),
      "p50_segments_after": float(torch.quantile(token_counts, 0.50)),
      "p90_segments_after": float(torch.quantile(token_counts, 0.90)),
      "interval_compression_ratio": float(max_duration / token_counts.mean().clamp_min(1.0)),
      **metrics,
  }
  return row, horizon_metrics(recon_xy, val_xy, method), token_sequences, recon_xy


def print_examples(
    val_accel: torch.Tensor,
    codebook: torch.Tensor,
    token_sequences: list[list[tuple[int, int]]],
    prefix: str,
    num_examples: int,
) -> None:
  _, primitive_ids = nearest_codebook(val_accel[:, 1:], codebook)
  for index in range(min(num_examples, len(token_sequences))):
    before_ids = [int(token_id) for token_id in primitive_ids[index].tolist()]
    print(
        f"{prefix} sample={index} before={','.join(str(x) for x in before_ids)} "
        f"before_rle={run_length_encode_ids(before_ids)} "
        f"after={segment_token_string(token_sequences[index])}"
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate factorized acceleration segment tokens.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--primitive-vocabs", type=int, nargs="+", default=[16, 32, 64, 128, 256])
  parser.add_argument("--xy-thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
  parser.add_argument("--kmeans-iters", type=int, default=50)
  parser.add_argument("--print-token-examples", type=int, default=8)
  parser.add_argument("--example-primitive-vocab", type=int, default=128)
  parser.add_argument("--example-threshold", type=float, default=0.3)
  parser.add_argument("--output-csv", default="waymo_accel_factorized_segments.csv")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
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
  train_accel = velocity_to_acceleration(train_velocity)
  val_accel = velocity_to_acceleration(val_velocity)
  train_interval_accel = train_accel[:, 1:].reshape(-1, 2)

  rows: list[dict[str, float | int | str]] = []
  horizon_rows: list[dict[str, float | int | str]] = []
  example: tuple[str, torch.Tensor, list[list[tuple[int, int]]]] | None = None
  print(f"Parsed stats: {stats}")
  print(f"Split: train={len(train_xy)} val={len(val_xy)}")
  print(f"fixed_interval_tokens_before={val_xy.shape[1] - 1}")

  for primitive_vocab in args.primitive_vocabs:
    codebook = fit_kmeans(
        train_interval_accel,
        num_clusters=primitive_vocab,
        iterations=args.kmeans_iters,
        seed=args.seed + primitive_vocab,
    )
    for threshold in args.xy_thresholds:
      method = f"factorized_k{primitive_vocab}_xythr_{threshold:g}"
      row, per_horizon, token_sequences, _ = evaluate_codebook(
          val_xy,
          val_velocity,
          val_accel,
          codebook,
          threshold,
          method,
      )
      rows.append(row)
      horizon_rows.extend(per_horizon)
      if primitive_vocab == args.example_primitive_vocab and abs(threshold - args.example_threshold) < 1e-9:
        example = (method, codebook, token_sequences)
      print(
          f"{method} factorized_logits={row['factorized_logits']} "
          f"flat_pair_vocab={row['flat_pair_vocab']} "
          f"avg_segments={row['avg_segments_after']:.3f} "
          f"xy_mse={row['xy_mse']:.6f} "
          f"max_first3s_p99={row['max_first3s_p99']:.6f} "
          f"p99={row['p99_by_second']}"
      )

  if example is None and rows:
    primitive_vocab = args.primitive_vocabs[-1]
    codebook = fit_kmeans(
        train_interval_accel,
        num_clusters=primitive_vocab,
        iterations=args.kmeans_iters,
        seed=args.seed + primitive_vocab,
    )
    method = f"factorized_k{primitive_vocab}_xythr_{args.xy_thresholds[-1]:g}"
    _, _, token_sequences, _ = evaluate_codebook(
        val_xy,
        val_velocity,
        val_accel,
        codebook,
        args.xy_thresholds[-1],
        method,
    )
    example = (method, codebook, token_sequences)

  if example is not None and args.print_token_examples > 0:
    method, codebook, token_sequences = example
    print_examples(val_accel, codebook, token_sequences, method, args.print_token_examples)

  write_csv(rows, Path(args.output_csv))
  horizon_csv = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_horizon.csv")
  write_csv(horizon_rows, horizon_csv)
  print(f"Saved CSV: {args.output_csv}")
  print(f"Saved horizon CSV: {horizon_csv}")


if __name__ == "__main__":
  main()
