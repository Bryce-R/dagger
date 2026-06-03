"""Evaluate quantized acceleration duration tokens on Waymo futures.

This tests a classification-friendly tokenizer: each token is a quantized local
acceleration bin plus a duration. Consecutive timesteps are merged when one
constant acceleration token can reconstruct the segment well enough.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_waymo_accel_binning import fit_uniform_bins
from evaluate_waymo_accel_horizon_metrics import horizon_metrics
from evaluate_waymo_xy_binning_baseline import split_dataset
from train_waymo_accel_fsq import (
    physical_trajectory_tensors,
    velocity_to_acceleration,
    xy_metrics,
)
from train_waymo_mlp_fsq import WAYMO_STEPS_PER_SECOND, extract_waymo_trajectories
from train_waymo_residual_fsq import write_csv


def quantize_accel_with_indices(
    accel: torch.Tensor,
    min_accel: torch.Tensor,
    width: torch.Tensor,
    num_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  indices = torch.floor((accel - min_accel) / width).long().clamp(0, num_bins - 1)
  centers = min_accel + (indices.float() + 0.5) * width
  joint_ids = indices[..., 0] * num_bins + indices[..., 1]
  return centers, joint_ids


def decode_constant_accel_segment(
    start_xy: torch.Tensor,
    start_velocity: torch.Tensor,
    acceleration: torch.Tensor,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  dt = 1.0 / WAYMO_STEPS_PER_SECOND
  xy = torch.empty((length, 2), dtype=start_xy.dtype)
  velocity = start_velocity.clone()
  xy[0] = start_xy
  for local_step in range(1, length):
    next_velocity = velocity + acceleration * dt
    xy[local_step] = xy[local_step - 1] + 0.5 * (velocity + next_velocity) * dt
    velocity = next_velocity
  return xy, velocity


def run_length_encode_ids(ids: list[int]) -> str:
  if not ids:
    return ""
  parts = []
  current = ids[0]
  count = 1
  for token_id in ids[1:]:
    if token_id == current:
      count += 1
    else:
      parts.append(f"{current}*{count}" if count > 1 else str(current))
      current = token_id
      count = 1
  parts.append(f"{current}*{count}" if count > 1 else str(current))
  return " ".join(parts)


def segment_token_string(tokens: list[tuple[int, int]]) -> str:
  return " ".join(
      f"{token_id}*{duration}" if duration > 1 else str(token_id)
      for token_id, duration in tokens
  )


def greedy_xy_threshold_segments(
    target_xy: torch.Tensor,
    target_velocity: torch.Tensor,
    target_accel: torch.Tensor,
    min_accel: torch.Tensor,
    width: torch.Tensor,
    num_bins: int,
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
      accel_center, joint_id = quantize_accel_with_indices(
          segment_accel.view(1, 2), min_accel, width, num_bins
      )
      candidate_xy, candidate_velocity = decode_constant_accel_segment(
          current_xy,
          current_velocity,
          accel_center[0],
          end - start + 2,
      )
      max_error = float((candidate_xy[1:] - target_xy[start : end + 1]).abs().max())
      if max_error <= xy_threshold or end == start:
        best = (end, accel_center[0], int(joint_id[0]), candidate_xy, candidate_velocity)
        break
    assert best is not None
    end, accel_center, joint_id, candidate_xy, candidate_velocity = best
    recon_xy[start : end + 1] = candidate_xy[1:]
    tokens.append((joint_id, end - start + 1))
    current_xy = candidate_xy[-1]
    current_velocity = candidate_velocity
    start = end + 1
  return recon_xy, tokens


def greedy_accel_similarity_segments(
    target_xy: torch.Tensor,
    target_velocity: torch.Tensor,
    target_accel: torch.Tensor,
    min_accel: torch.Tensor,
    width: torch.Tensor,
    num_bins: int,
    accel_threshold: float,
    xy_threshold: float | None,
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
      segment_accel_values = target_accel[start : end + 1]
      segment_accel = segment_accel_values.mean(dim=0)
      max_accel_delta = float((segment_accel_values - segment_accel).norm(dim=1).max())
      if max_accel_delta > accel_threshold and end > start:
        continue
      accel_center, joint_id = quantize_accel_with_indices(
          segment_accel.view(1, 2), min_accel, width, num_bins
      )
      candidate_xy, candidate_velocity = decode_constant_accel_segment(
          current_xy,
          current_velocity,
          accel_center[0],
          end - start + 2,
      )
      max_xy_error = float((candidate_xy[1:] - target_xy[start : end + 1]).abs().max())
      if xy_threshold is not None and max_xy_error > xy_threshold and end > start:
        continue
      best = (end, int(joint_id[0]), candidate_xy, candidate_velocity)
      break
    assert best is not None
    end, joint_id, candidate_xy, candidate_velocity = best
    recon_xy[start : end + 1] = candidate_xy[1:]
    tokens.append((joint_id, end - start + 1))
    current_xy = candidate_xy[-1]
    current_velocity = candidate_velocity
    start = end + 1
  return recon_xy, tokens


def evaluate_tokenizer(
    method: str,
    val_xy: torch.Tensor,
    val_velocity: torch.Tensor,
    val_accel: torch.Tensor,
    min_accel: torch.Tensor,
    width: torch.Tensor,
    num_bins: int,
    mode: str,
    threshold: float,
    xy_guard_threshold: float | None,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], list[list[tuple[int, int]]], torch.Tensor]:
  recon_parts = []
  token_sequences = []
  for index in range(len(val_xy)):
    if mode == "xy":
      recon_xy, tokens = greedy_xy_threshold_segments(
          val_xy[index],
          val_velocity[index],
          val_accel[index],
          min_accel,
          width,
          num_bins,
          threshold,
      )
    elif mode == "accel":
      recon_xy, tokens = greedy_accel_similarity_segments(
          val_xy[index],
          val_velocity[index],
          val_accel[index],
          min_accel,
          width,
          num_bins,
          threshold,
          xy_guard_threshold,
      )
    else:
      raise ValueError(f"unknown mode: {mode}")
    recon_parts.append(recon_xy)
    token_sequences.append(tokens)
  recon_xy_all = torch.stack(recon_parts)
  token_counts = torch.tensor([len(tokens) for tokens in token_sequences], dtype=torch.float32)
  metrics = xy_metrics(recon_xy_all, val_xy)
  row = {
      "method": method,
      "mode": mode,
      "threshold": threshold,
      "xy_guard_threshold": "" if xy_guard_threshold is None else xy_guard_threshold,
      "avg_tokens_before": float(torch.tensor(val_xy.shape[1], dtype=torch.float32)),
      "avg_tokens_after": float(token_counts.mean()),
      "p50_tokens_after": float(torch.quantile(token_counts, 0.50)),
      "p90_tokens_after": float(torch.quantile(token_counts, 0.90)),
      "compression_ratio": float(val_xy.shape[1] / token_counts.mean().clamp_min(1.0)),
      **metrics,
  }
  horizon_rows = horizon_metrics(recon_xy_all, val_xy, method)
  return row, horizon_rows, token_sequences, recon_xy_all


def print_token_examples(
    joint_ids: torch.Tensor,
    token_sequences: list[list[tuple[int, int]]],
    prefix: str,
    num_examples: int,
) -> None:
  for index in range(min(num_examples, len(token_sequences))):
    before_ids = [int(token_id) for token_id in joint_ids[index, 1:].tolist()]
    before = ",".join(str(token_id) for token_id in before_ids)
    before_rle = run_length_encode_ids(before_ids)
    after = segment_token_string(token_sequences[index])
    print(
        f"{prefix} sample={index} before={before} "
        f"before_rle={before_rle} after={after}"
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate quantized acceleration duration tokens.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--accel-bins", type=int, default=71)
  parser.add_argument("--xy-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
  parser.add_argument("--accel-thresholds", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
  parser.add_argument("--accel-xy-guard-threshold", type=float, default=0.3)
  parser.add_argument("--print-token-examples", type=int, default=8)
  parser.add_argument("--output-csv", default="waymo_accel_segment_tokens.csv")
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
  min_accel, width = fit_uniform_bins(train_accel, args.accel_bins, False)
  _, val_joint_ids = quantize_accel_with_indices(val_accel, min_accel, width, args.accel_bins)

  rows: list[dict[str, float | int | str]] = []
  horizon_rows: list[dict[str, float | int | str]] = []
  example_sequences: tuple[str, list[list[tuple[int, int]]]] | None = None
  print(f"Parsed stats: {stats}")
  print(f"Split: train={len(train_xy)} val={len(val_xy)}")
  print(f"joint_accel_vocab={args.accel_bins * args.accel_bins}")

  for threshold in args.xy_thresholds:
    method = f"accel_duration_xy_threshold_{threshold:g}"
    row, per_horizon, token_sequences, _ = evaluate_tokenizer(
        method,
        val_xy,
        val_velocity,
        val_accel,
        min_accel,
        width,
        args.accel_bins,
        mode="xy",
        threshold=threshold,
        xy_guard_threshold=None,
    )
    rows.append(row)
    horizon_rows.extend(per_horizon)
    if example_sequences is None or math.isclose(threshold, 0.2):
      example_sequences = (method, token_sequences)
    print(
        f"{method} avg_tokens_before={row['avg_tokens_before']:.3f} "
        f"avg_tokens_after={row['avg_tokens_after']:.3f} "
        f"xy_mse={row['xy_mse']:.6f} max_first3s_p99={row['max_first3s_p99']:.6f} "
        f"p99={row['p99_by_second']}"
    )

  for threshold in args.accel_thresholds:
    method = f"accel_duration_accel_threshold_{threshold:g}_xyguard_{args.accel_xy_guard_threshold:g}"
    row, per_horizon, token_sequences, _ = evaluate_tokenizer(
        method,
        val_xy,
        val_velocity,
        val_accel,
        min_accel,
        width,
        args.accel_bins,
        mode="accel",
        threshold=threshold,
        xy_guard_threshold=args.accel_xy_guard_threshold,
    )
    rows.append(row)
    horizon_rows.extend(per_horizon)
    print(
        f"{method} avg_tokens_before={row['avg_tokens_before']:.3f} "
        f"avg_tokens_after={row['avg_tokens_after']:.3f} "
        f"xy_mse={row['xy_mse']:.6f} max_first3s_p99={row['max_first3s_p99']:.6f} "
        f"p99={row['p99_by_second']}"
    )

  if example_sequences is not None and args.print_token_examples > 0:
    prefix, token_sequences = example_sequences
    print_token_examples(
        val_joint_ids,
        token_sequences,
        prefix=prefix,
        num_examples=args.print_token_examples,
    )

  write_csv(rows, Path(args.output_csv))
  horizon_csv = Path(args.output_csv).with_name(Path(args.output_csv).stem + "_horizon.csv")
  write_csv(horizon_rows, horizon_csv)
  print(f"Saved CSV: {args.output_csv}")
  print(f"Saved horizon CSV: {horizon_csv}")


if __name__ == "__main__":
  main()
