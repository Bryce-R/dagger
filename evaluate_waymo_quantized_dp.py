"""Evaluate quantized piecewise-DP trajectory tokens against residual FSQ.

This keeps DP's temporal segmentation oracle, but replaces continuous segment
polynomial coefficients with scalar-uniform quantized coefficients. The point is
to separate "DP is using continuous precision" from "DP is using better temporal
chunking and duration tokens."
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_waymo_piecewise_dp import (
    design_matrix,
    dynamic_program_segments,
    physical_xy_positions,
    reconstruction_metrics,
    write_csv,
)
from evaluate_waymo_xy_binning_baseline import split_dataset
from train_waymo_mlp_fsq import extract_waymo_trajectories


def segment_spans(boundaries: torch.Tensor) -> list[list[tuple[int, int]]]:
  """Converts boundary indicators into half-open segment spans."""
  all_spans = []
  num_steps = boundaries.shape[1] + 1
  for row in boundaries:
    starts = [0]
    starts.extend(index + 1 for index in torch.nonzero(row > 0.0).flatten().tolist())
    starts.append(num_steps)
    all_spans.append(
        [(starts[index], starts[index + 1]) for index in range(len(starts) - 1)]
    )
  return all_spans


def fit_segment_coefficients(segment_xy: torch.Tensor, degree: int) -> torch.Tensor:
  """Returns padded polynomial coefficients with shape [degree + 1, 2]."""
  matrix = design_matrix(segment_xy.shape[0], degree).to(segment_xy.device)
  pseudo_inverse = torch.linalg.pinv(matrix)
  coeff = torch.einsum("kl,ld->kd", pseudo_inverse, segment_xy)
  padded = torch.zeros(degree + 1, 2, dtype=segment_xy.dtype, device=segment_xy.device)
  padded[: coeff.shape[0]] = coeff
  return padded


def reconstruct_coefficients(coeff: torch.Tensor, length: int, degree: int) -> torch.Tensor:
  matrix = design_matrix(length, degree).to(coeff.device)
  return torch.einsum("lk,kd->ld", matrix, coeff[: matrix.shape[1]])


def collect_coefficients_by_duration(
    xy: torch.Tensor,
    spans: list[list[tuple[int, int]]],
    degree: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
  all_coeffs = []
  by_duration: dict[int, list[torch.Tensor]] = {}
  for example_index, example_spans in enumerate(spans):
    for start, end in example_spans:
      coeff = fit_segment_coefficients(xy[example_index, start:end], degree).cpu()
      all_coeffs.append(coeff)
      by_duration.setdefault(end - start, []).append(coeff)
  return torch.stack(all_coeffs), {
      duration: torch.stack(coeffs) for duration, coeffs in by_duration.items()
  }


def fit_ranges(coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  return coeffs.amin(dim=0), coeffs.amax(dim=0)


def quantize_uniform(
    coeff: torch.Tensor,
    coeff_min: torch.Tensor,
    coeff_max: torch.Tensor,
    levels: int,
) -> torch.Tensor:
  if levels < 2:
    return (coeff_min + coeff_max) * 0.5
  step = (coeff_max - coeff_min).clamp_min(1e-9) / float(levels - 1)
  index = torch.round((coeff - coeff_min) / step).clamp(0, levels - 1)
  return coeff_min + index * step


def reconstruct_quantized_dp(
    xy: torch.Tensor,
    spans: list[list[tuple[int, int]]],
    degree: int,
    levels: int,
    global_range: tuple[torch.Tensor, torch.Tensor],
    duration_ranges: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
) -> torch.Tensor:
  recon = torch.empty_like(xy)
  global_min, global_max = global_range
  for example_index, example_spans in enumerate(spans):
    for start, end in example_spans:
      duration = end - start
      coeff_min, coeff_max = global_min, global_max
      if duration_ranges is not None and duration in duration_ranges:
        coeff_min, coeff_max = duration_ranges[duration]
      coeff = fit_segment_coefficients(xy[example_index, start:end], degree).cpu()
      quantized = quantize_uniform(coeff, coeff_min, coeff_max, levels).to(xy.device)
      recon[example_index, start:end] = reconstruct_coefficients(
          quantized, duration, degree
      )
  return recon


def read_residual_fsq_rows(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open(newline="") as file:
    return list(csv.DictReader(file))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Evaluate quantized Waymo DP tokens.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--include-all-states", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--degree", type=int, default=2)
  parser.add_argument("--threshold", type=float, default=0.2)
  parser.add_argument(
      "--levels",
      type=int,
      nargs="+",
      default=[4, 8, 16, 32, 64, 128, 256],
  )
  parser.add_argument(
      "--residual-fsq-csv",
      default="waymo_residual_fsq_tok8_cd16_fsq64_thr001_e1000_b512/metrics.csv",
  )
  parser.add_argument("--output-csv", default="waymo_quantized_dp_vs_residual_fsq.csv")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
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
  print(f"Parsed stats: {stats}")
  print(f"Split: train={len(train_xy)} val={len(val_xy)}")

  train_recon, train_boundaries, train_token_counts = dynamic_program_segments(
      train_xy, degree=args.degree, max_error_threshold=args.threshold
  )
  val_recon, val_boundaries, val_token_counts = dynamic_program_segments(
      val_xy, degree=args.degree, max_error_threshold=args.threshold
  )
  del train_recon

  train_spans = segment_spans(train_boundaries)
  val_spans = segment_spans(val_boundaries)
  train_coeffs, train_coeffs_by_duration = collect_coefficients_by_duration(
      train_xy, train_spans, args.degree
  )
  global_range = fit_ranges(train_coeffs)
  duration_ranges = {
      duration: fit_ranges(coeffs)
      for duration, coeffs in train_coeffs_by_duration.items()
  }

  rows: list[dict[str, float | int | str]] = []
  oracle_metrics = reconstruction_metrics(val_recon, val_xy)
  rows.append(
      {
          "mode": "dp_continuous",
          "degree": args.degree,
          "threshold": args.threshold,
          "levels": "continuous",
          "avg_tokens": float(val_token_counts.float().mean()),
          "p50_tokens": float(torch.quantile(val_token_counts.float(), 0.50)),
          "p90_tokens": float(torch.quantile(val_token_counts.float(), 0.90)),
          **oracle_metrics,
      }
  )

  for levels in args.levels:
    for quantizer_name, ranges in (
        ("global_coeff", None),
        ("duration_coeff", duration_ranges),
    ):
      recon = reconstruct_quantized_dp(
          val_xy,
          val_spans,
          args.degree,
          levels,
          global_range,
          ranges,
      )
      metrics = reconstruction_metrics(recon, val_xy)
      rows.append(
          {
              "mode": f"dp_quantized_{quantizer_name}",
              "degree": args.degree,
              "threshold": args.threshold,
              "levels": levels,
              "avg_tokens": float(val_token_counts.float().mean()),
              "p50_tokens": float(torch.quantile(val_token_counts.float(), 0.50)),
              "p90_tokens": float(torch.quantile(val_token_counts.float(), 0.90)),
              "token_duration_vocab": args.num_steps,
              "coeff_scalars_per_token": (args.degree + 1) * 2,
              **metrics,
          }
      )

  for residual_row in read_residual_fsq_rows(Path(args.residual_fsq_csv)):
    if not residual_row.get("xy_mse"):
      continue
    rows.append(
        {
            "mode": "residual_fsq",
            "degree": "",
            "threshold": "",
            "levels": "",
            "avg_tokens": residual_row["tokens"],
            "p50_tokens": "",
            "p90_tokens": "",
            "xy_mse": residual_row["xy_mse"],
            "xy_mae": residual_row["xy_mae"],
            "max_error": residual_row["max_error"],
            "p99_error": residual_row["p99_error"],
            "max_first3s_p99": residual_row["max_first3s_p99"],
            "p99_by_second": residual_row["p99_by_second"],
        }
    )

  for row in rows:
    print(
        f"mode={row['mode']} levels={row['levels']} avg_tokens={row['avg_tokens']} "
        f"mse={float(row['xy_mse']):.6f} "
        f"max_first3s_p99={float(row['max_first3s_p99']):.6f} "
        f"p99_by_second={row['p99_by_second']}"
    )
  output_csv = Path(args.output_csv)
  write_csv(rows, output_csv)
  print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
  main()
