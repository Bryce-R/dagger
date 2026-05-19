"""Evaluate variable-length piecewise trajectory tokens on Waymo futures.

The oracle tokenizer uses dynamic programming to split a fixed-length trajectory
into the fewest motion-primitive segments that satisfy an error threshold, or
to minimize squared reconstruction error plus a per-token penalty.
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
    extract_waymo_trajectories,
    xy_to_positions,
)


def physical_xy_positions(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
  return xy_to_positions(x * std + mean, decode_absolute_positions=True)


def design_matrix(length: int, degree: int) -> torch.Tensor:
  effective_degree = min(degree, length - 1)
  if length == 1:
    t = torch.zeros(1)
  else:
    t = torch.linspace(0.0, 1.0, length)
  columns = [torch.ones_like(t)]
  for power in range(1, effective_degree + 1):
    columns.append(t.pow(power))
  return torch.stack(columns, dim=1)


def fit_segment_batch(segment_xy: torch.Tensor, degree: int) -> torch.Tensor:
  """Fits one polynomial segment for each batch item."""
  length = segment_xy.shape[1]
  if length <= 1:
    return segment_xy.clone()
  matrix = design_matrix(length, degree).to(segment_xy.device)
  pseudo_inverse = torch.linalg.pinv(matrix)
  coeff = torch.einsum("kl,nld->nkd", pseudo_inverse, segment_xy)
  return torch.einsum("lk,nkd->nld", matrix, coeff)


def precompute_segments(
    xy: torch.Tensor,
    degree: int,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor]]:
  """Returns reconstructions and per-example errors for all half-open spans."""
  num_steps = xy.shape[1]
  reconstructions = {}
  sum_squared_errors = {}
  max_abs_errors = {}
  for start in range(num_steps):
    for end in range(start + 1, num_steps + 1):
      recon = fit_segment_batch(xy[:, start:end], degree)
      error = recon - xy[:, start:end]
      reconstructions[(start, end)] = recon.cpu()
      sum_squared_errors[(start, end)] = error.square().sum(dim=(1, 2)).cpu()
      max_abs_errors[(start, end)] = error.abs().amax(dim=(1, 2)).cpu()
  return reconstructions, sum_squared_errors, max_abs_errors


def dynamic_program_segments(
    xy: torch.Tensor,
    degree: int,
    token_penalty: float | None = None,
    max_error_threshold: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Segments each trajectory and reconstructs it with fitted primitives."""
  if (token_penalty is None) == (max_error_threshold is None):
    raise ValueError("Set exactly one of token_penalty or max_error_threshold.")

  recon_by_span, sse_by_span, max_by_span = precompute_segments(xy, degree)
  num_examples, num_steps, _ = xy.shape
  inf = torch.tensor(float("inf"))
  dp = torch.full((num_examples, num_steps + 1), float("inf"))
  back = torch.full((num_examples, num_steps + 1), -1, dtype=torch.long)
  dp[:, 0] = 0.0

  for end in range(1, num_steps + 1):
    best = torch.full((num_examples,), float("inf"))
    best_start = torch.full((num_examples,), -1, dtype=torch.long)
    for start in range(end):
      if token_penalty is not None:
        segment_cost = sse_by_span[(start, end)] + token_penalty
      else:
        valid = max_by_span[(start, end)] <= float(max_error_threshold)
        segment_cost = torch.where(valid, torch.ones(num_examples), inf)
      candidate = dp[:, start] + segment_cost
      better = candidate < best
      best = torch.where(better, candidate, best)
      best_start = torch.where(
          better,
          torch.full_like(best_start, start),
          best_start,
      )
    dp[:, end] = best
    back[:, end] = best_start

  recon = torch.empty_like(xy)
  boundaries = torch.zeros((num_examples, num_steps - 1), dtype=torch.float32)
  token_counts = torch.zeros(num_examples, dtype=torch.long)
  for index in range(num_examples):
    end = num_steps
    while end > 0:
      start = int(back[index, end])
      if start < 0:
        raise RuntimeError("DP failed to find a valid segmentation.")
      recon[index, start:end] = recon_by_span[(start, end)][index]
      if start > 0:
        boundaries[index, start - 1] = 1.0
      token_counts[index] += 1
      end = start
  return recon, boundaries, token_counts


def format_second_errors(errors: dict[int, float]) -> str:
  return " ".join(f"{second}s={error:.6f}" for second, error in errors.items())


def reconstruction_metrics(
    recon_xy: torch.Tensor,
    target_xy: torch.Tensor,
) -> dict[str, float | str]:
  abs_error = (recon_xy - target_xy).abs()
  num_steps = target_xy.shape[1]
  max_seconds = math.ceil(num_steps / WAYMO_STEPS_PER_SECOND)
  p99_by_second = {}
  max_by_second = {}
  for second in range(1, max_seconds + 1):
    step = min(second * WAYMO_STEPS_PER_SECOND - 1, num_steps - 1)
    step_error = abs_error[:, step].reshape(-1)
    p99_by_second[second] = float(torch.quantile(step_error, 0.99))
    max_by_second[second] = float(step_error.max())
  first_three_steps = min(3 * WAYMO_STEPS_PER_SECOND, num_steps)
  p99_by_step = [
      float(torch.quantile(abs_error[:, step].reshape(-1), 0.99))
      for step in range(first_three_steps)
  ]
  return {
      "xy_mse": float(F.mse_loss(recon_xy, target_xy)),
      "xy_mae": float(abs_error.mean()),
      "max_error": float(abs_error.max()),
      "p99_error": float(torch.quantile(abs_error.reshape(-1), 0.99)),
      "max_first3s_p99": max(p99_by_step),
      "p99_by_second": format_second_errors(p99_by_second),
      "max_by_second": format_second_errors(max_by_second),
      "p99_first3s_steps": " ".join(f"{value:.6f}" for value in p99_by_step),
  }


def evaluate_dp_sweep(
    xy: torch.Tensor,
    degrees: list[int],
    token_penalties: list[float],
    thresholds: list[float],
) -> tuple[list[dict[str, float | int | str]], dict[tuple[int, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
  rows = []
  threshold_cache = {}
  for degree in degrees:
    for threshold in thresholds:
      recon, boundaries, token_counts = dynamic_program_segments(
          xy, degree=degree, max_error_threshold=threshold
      )
      metrics = reconstruction_metrics(recon, xy)
      rows.append(
          {
              "mode": "threshold",
              "degree": degree,
              "value": threshold,
              "avg_tokens": float(token_counts.float().mean()),
              "p50_tokens": float(torch.quantile(token_counts.float(), 0.50)),
              "p90_tokens": float(torch.quantile(token_counts.float(), 0.90)),
              **metrics,
          }
      )
      threshold_cache[(degree, threshold)] = (recon, boundaries, token_counts)
    for penalty in token_penalties:
      recon, _, token_counts = dynamic_program_segments(
          xy, degree=degree, token_penalty=penalty
      )
      metrics = reconstruction_metrics(recon, xy)
      rows.append(
          {
              "mode": "penalty",
              "degree": degree,
              "value": penalty,
              "avg_tokens": float(token_counts.float().mean()),
              "p50_tokens": float(torch.quantile(token_counts.float(), 0.50)),
              "p90_tokens": float(torch.quantile(token_counts.float(), 0.90)),
              **metrics,
          }
      )
  return rows, threshold_cache


class BoundaryMLP(nn.Module):
  def __init__(self, num_steps: int, input_dim: int, hidden_dim: int):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(num_steps * input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_steps - 1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x.reshape(x.shape[0], -1))


def reconstruct_from_boundaries(
    xy: torch.Tensor,
    boundaries: torch.Tensor,
    degree: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  num_examples, num_steps, _ = xy.shape
  recon = torch.empty_like(xy)
  token_counts = torch.zeros(num_examples, dtype=torch.long)
  for index in range(num_examples):
    starts = [0]
    for boundary_index in torch.nonzero(boundaries[index] > 0.0).flatten().tolist():
      starts.append(boundary_index + 1)
    starts.append(num_steps)
    token_counts[index] = len(starts) - 1
    for segment_index in range(len(starts) - 1):
      start = starts[segment_index]
      end = starts[segment_index + 1]
      recon[index, start:end] = fit_segment_batch(
          xy[index : index + 1, start:end], degree
      )[0]
  return recon, token_counts


def train_boundary_model(
    train_x: torch.Tensor,
    train_boundaries: torch.Tensor,
    val_x: torch.Tensor,
    val_boundaries: torch.Tensor,
    val_xy: torch.Tensor,
    degree: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    prediction_thresholds: list[float],
) -> list[dict[str, float | str]]:
  model = BoundaryMLP(train_x.shape[1], train_x.shape[2], hidden_dim).to(device)
  positives = train_boundaries.sum()
  negatives = train_boundaries.numel() - positives
  pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
  loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  loader = DataLoader(
      TensorDataset(train_x, train_boundaries),
      batch_size=batch_size,
      shuffle=True,
      generator=torch.Generator().manual_seed(7),
  )
  for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    batches = 0
    for batch_x, batch_y in loader:
      batch_x = batch_x.to(device)
      batch_y = batch_y.to(device)
      logits = model(batch_x)
      loss = loss_fn(logits, batch_y)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      total_loss += float(loss.detach())
      batches += 1
    if epoch == 1 or epoch % max(1, epochs // 4) == 0 or epoch == epochs:
      print(f"learn_epoch={epoch:04d} boundary_bce={total_loss / batches:.6f}")

  model.eval()
  with torch.no_grad():
    probs = torch.sigmoid(model(val_x.to(device))).cpu()

  rows = []
  for threshold in prediction_thresholds:
    predicted = (probs >= threshold).float()
    recon, token_counts = reconstruct_from_boundaries(val_xy, predicted, degree)
    metrics = reconstruction_metrics(recon, val_xy)
    boundary_accuracy = float((predicted == val_boundaries).float().mean())
    rows.append(
        {
            "mode": "learned_boundary",
            "degree": degree,
            "value": threshold,
            "avg_tokens": float(token_counts.float().mean()),
            "p50_tokens": float(torch.quantile(token_counts.float(), 0.50)),
            "p90_tokens": float(torch.quantile(token_counts.float(), 0.90)),
            "boundary_accuracy": boundary_accuracy,
            **metrics,
        }
    )
  return rows


def plot_reconstructions(
    target_xy: torch.Tensor,
    recon_xy: torch.Tensor,
    boundaries: torch.Tensor,
    token_counts: torch.Tensor,
    output_path: Path,
    title: str,
    sample_indices: list[int],
) -> None:
  """Saves target/reconstruction XY plots with segment boundaries."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  if not sample_indices:
    return
  output_path.parent.mkdir(parents=True, exist_ok=True)
  num_cols = min(4, len(sample_indices))
  num_rows = math.ceil(len(sample_indices) / num_cols)
  fig, axes = plt.subplots(
      num_rows,
      num_cols,
      figsize=(4.0 * num_cols, 4.0 * num_rows),
      squeeze=False,
  )
  for axis, sample_index in zip(axes.ravel(), sample_indices):
    target = target_xy[sample_index]
    recon = recon_xy[sample_index]
    axis.plot(target[:, 0], target[:, 1], "o-", color="tab:blue", label="target")
    axis.plot(recon[:, 0], recon[:, 1], "x--", color="tab:orange", label="piecewise")
    boundary_steps = [
        boundary_index + 1
        for boundary_index in torch.nonzero(boundaries[sample_index] > 0.0).flatten().tolist()
    ]
    if boundary_steps:
      boundary_points = target[boundary_steps]
      axis.scatter(
          boundary_points[:, 0],
          boundary_points[:, 1],
          s=80,
          facecolors="none",
          edgecolors="tab:red",
          linewidths=1.8,
          label="boundary",
      )
    max_error = float((target - recon).abs().max())
    axis.set_title(
        f"sample={sample_index} tokens={int(token_counts[sample_index])} "
        f"max_err={max_error:.3f}m"
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
  for axis in axes.ravel()[len(sample_indices):]:
    axis.axis("off")
  handles, labels = axes.ravel()[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=3)
  fig.suptitle(title.replace("_", " "), y=0.98)
  fig.tight_layout(rect=(0, 0.05, 1, 0.94))
  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def save_dp_plots(
    target_xy: torch.Tensor,
    recon_xy: torch.Tensor,
    boundaries: torch.Tensor,
    token_counts: torch.Tensor,
    output_dir: Path,
    prefix: str,
    num_samples: int,
) -> list[Path]:
  first_count = min(num_samples, len(target_xy))
  first_indices = list(range(first_count))
  first_three_steps = min(3 * WAYMO_STEPS_PER_SECOND, target_xy.shape[1])
  first_three_error = (target_xy[:, :first_three_steps] - recon_xy[:, :first_three_steps]).abs()
  per_example_error = first_three_error.amax(dim=(1, 2))
  worst_indices = torch.topk(
      per_example_error,
      k=min(num_samples, len(target_xy)),
  ).indices.tolist()
  first_path = output_dir / f"{prefix}_first_samples.png"
  worst_path = output_dir / f"{prefix}_worst_first3s.png"
  plot_reconstructions(
      target_xy,
      recon_xy,
      boundaries,
      token_counts,
      first_path,
      f"{prefix}: first validation samples",
      first_indices,
  )
  plot_reconstructions(
      target_xy,
      recon_xy,
      boundaries,
      token_counts,
      worst_path,
      f"{prefix}: worst first-3s validation samples",
      worst_indices,
  )
  return [first_path, worst_path]


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
  parser = argparse.ArgumentParser(description="Evaluate Waymo piecewise DP tokens.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=10)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--include-all-states", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
  parser.add_argument(
      "--token-penalties",
      type=float,
      nargs="+",
      default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
  )
  parser.add_argument(
      "--thresholds",
      type=float,
      nargs="+",
      default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
  )
  parser.add_argument("--output-csv", default="waymo_piecewise_dp.csv")
  parser.add_argument("--plot-dir", default=None)
  parser.add_argument("--num-plot-samples", type=int, default=8)
  parser.add_argument("--plot-degree", type=int, default=2)
  parser.add_argument("--plot-threshold", type=float, default=0.2)
  parser.add_argument("--learn-boundaries", action="store_true")
  parser.add_argument("--learn-degree", type=int, default=2)
  parser.add_argument("--learn-threshold", type=float, default=0.2)
  parser.add_argument("--learn-hidden-dim", type=int, default=128)
  parser.add_argument("--learn-epochs", type=int, default=200)
  parser.add_argument("--learn-batch-size", type=int, default=256)
  parser.add_argument("--learn-lr", type=float, default=1e-3)
  parser.add_argument(
      "--learn-prediction-thresholds",
      type=float,
      nargs="+",
      default=[0.3, 0.5, 0.7],
  )
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
      include_yaw=False,
      include_all_states=args.include_all_states,
      decode_absolute_positions=True,
  )
  train_x, _, val_x, _ = split_dataset(x, labels, args.val_fraction, args.seed)
  train_xy = physical_xy_positions(train_x, mean, std)
  val_xy = physical_xy_positions(val_x, mean, std)
  print(f"Parsed stats: {stats}")
  print(f"Split: train={len(train_x)} val={len(val_x)}")

  rows, threshold_cache = evaluate_dp_sweep(
      val_xy,
      degrees=args.degrees,
      token_penalties=args.token_penalties,
      thresholds=args.thresholds,
  )
  for row in rows:
    print(
        f"mode={row['mode']} degree={row['degree']} value={row['value']} "
        f"avg_tokens={row['avg_tokens']:.3f} "
        f"max_first3s_p99={row['max_first3s_p99']:.6f} "
        f"p99_by_second={row['p99_by_second']}"
    )

  if args.plot_dir:
    plot_key = (args.plot_degree, args.plot_threshold)
    if plot_key in threshold_cache:
      plot_recon, plot_boundaries, plot_token_counts = threshold_cache[plot_key]
    else:
      plot_recon, plot_boundaries, plot_token_counts = dynamic_program_segments(
          val_xy, degree=args.plot_degree, max_error_threshold=args.plot_threshold
      )
    plot_paths = save_dp_plots(
        target_xy=val_xy,
        recon_xy=plot_recon,
        boundaries=plot_boundaries,
        token_counts=plot_token_counts,
        output_dir=Path(args.plot_dir),
        prefix=f"dp_degree{args.plot_degree}_threshold{args.plot_threshold:g}",
        num_samples=args.num_plot_samples,
    )
    for plot_path in plot_paths:
      print(f"Saved plot: {plot_path}")

  if args.learn_boundaries:
    target_key = (args.learn_degree, args.learn_threshold)
    if target_key not in threshold_cache:
      _, train_boundaries, _ = dynamic_program_segments(
          train_xy, degree=args.learn_degree, max_error_threshold=args.learn_threshold
      )
      _, val_boundaries, _ = dynamic_program_segments(
          val_xy, degree=args.learn_degree, max_error_threshold=args.learn_threshold
      )
    else:
      _, val_boundaries, _ = threshold_cache[target_key]
      _, train_boundaries, _ = dynamic_program_segments(
          train_xy, degree=args.learn_degree, max_error_threshold=args.learn_threshold
      )
    learned_rows = train_boundary_model(
        train_x=train_x,
        train_boundaries=train_boundaries,
        val_x=val_x,
        val_boundaries=val_boundaries,
        val_xy=val_xy,
        degree=args.learn_degree,
        hidden_dim=args.learn_hidden_dim,
        epochs=args.learn_epochs,
        batch_size=args.learn_batch_size,
        lr=args.learn_lr,
        device=torch.device(args.device),
        prediction_thresholds=args.learn_prediction_thresholds,
    )
    rows.extend(learned_rows)
    for row in learned_rows:
      print(
          f"mode={row['mode']} degree={row['degree']} threshold={row['value']} "
          f"avg_tokens={row['avg_tokens']:.3f} "
          f"boundary_accuracy={row['boundary_accuracy']:.4f} "
          f"max_first3s_p99={row['max_first3s_p99']:.6f} "
          f"p99_by_second={row['p99_by_second']}"
      )

  output_csv = Path(args.output_csv)
  write_csv(rows, output_csv)
  print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
  main()
