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


def select_target_features(x: torch.Tensor, target_space: str) -> torch.Tensor:
  if target_space == "xy":
    return x[..., :2]
  if target_space == "xy_yaw_velocity":
    if x.shape[-1] < 6:
      raise ValueError("--target-space xy_yaw_velocity requires --include-all-states.")
    return x[..., [0, 1, 2, 4, 5]]
  raise ValueError(f"unknown target space: {target_space}")


def target_features_to_xy(
    target: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
  physical = target * feature_std.view(1, 1, -1) + feature_mean.view(1, 1, -1)
  return physical[..., :2]


class ResidualFSQTokenizer(nn.Module):
  """Recurrent residual autoencoder with one FSQ token per correction step."""

  def __init__(
      self,
      num_steps: int,
      target_dim: int,
      hidden_dim: int,
      code_dim: int,
      fsq_levels: list[int],
      fsq_input_scale: float,
      feature_mean: torch.Tensor,
      feature_std: torch.Tensor,
  ):
    super().__init__()
    self.num_steps = num_steps
    self.target_dim = target_dim
    self.register_buffer("feature_mean", feature_mean.view(1, 1, target_dim))
    self.register_buffer("feature_std", feature_std.view(1, 1, target_dim).clamp_min(1e-6))
    input_dim = num_steps * target_dim * 3
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
        nn.Linear(hidden_dim, num_steps * target_dim),
    )
    self.quantizer = FiniteScalarQuantizer(fsq_levels, code_dim, fsq_input_scale)

  def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
    return (features - self.feature_mean) / self.feature_std

  def correction_from_token(
      self,
      target: torch.Tensor,
      recon: torch.Tensor,
      quantize_strength: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = target - recon
    features = torch.cat(
        [
            self.normalize_features(target),
            self.normalize_features(recon),
            residual / self.feature_std,
        ],
        dim=-1,
    )
    latent = self.encoder(features.reshape(features.shape[0], -1))
    latent = latent.unsqueeze(-1)
    quantized, vq_loss, indices = self.quantizer(latent, quantize_strength)
    correction_norm = self.decoder(quantized.squeeze(-1)).view(
        target.shape[0], self.num_steps, self.target_dim
    )
    correction = correction_norm * self.feature_std
    return correction, vq_loss, indices.squeeze(1)

  def forward(
      self,
      target: torch.Tensor,
      max_tokens: int,
      quantize_strength: float = 1.0,
  ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    recon = torch.zeros_like(target)
    reconstructions = []
    index_history = []
    total_vq = target.sum() * 0.0
    for _ in range(max_tokens):
      correction_xy, vq_loss, indices = self.correction_from_token(
          target, recon, quantize_strength
      )
      recon = recon + correction_xy
      reconstructions.append(recon)
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


def stop_token_counts_by_xy(
    reconstructions: list[torch.Tensor],
    target_xy: torch.Tensor,
    threshold: float,
    horizon_steps: int | None,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
  recon_xy = [
      target_features_to_xy(recon, feature_mean.to(recon.device), feature_std.to(recon.device))
      for recon in reconstructions
  ]
  return stop_token_counts(recon_xy, target_xy, threshold, horizon_steps)


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
    target: torch.Tensor,
    target_xy: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
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
    loader = DataLoader(TensorDataset(target, target_xy), batch_size=batch_size)
    for batch_target, batch_target_xy in loader:
      batch_target = batch_target.to(device)
      batch_target_xy = batch_target_xy.to(device)
      recons, _, _ = model(batch_target, max_tokens=max_tokens)
      counts = stop_token_counts_by_xy(
          recons,
          batch_target_xy,
          threshold,
          stop_horizon_steps,
          feature_mean,
          feature_std,
      )
      token_count_parts.append(counts.cpu())
      for token_index, recon in enumerate(recons):
        recon_parts[token_index].append(
            target_features_to_xy(
                recon.cpu(),
                feature_mean.cpu(),
                feature_std.cpu(),
            )
        )
  target_xy = target_xy.cpu()
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


def select_stopped_reconstructions(
    reconstructions: list[torch.Tensor],
    token_counts: torch.Tensor,
) -> torch.Tensor:
  stopped = torch.empty_like(reconstructions[-1])
  for token_index, recon in enumerate(reconstructions, start=1):
    mask = token_counts == token_index
    if mask.any():
      stopped[mask] = recon[mask]
  return stopped


def plot_reconstructions(
    target_xy: torch.Tensor,
    recon_xy: torch.Tensor,
    token_counts: torch.Tensor,
    output_path: Path,
    title: str,
    sample_indices: list[int],
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  if not sample_indices:
    return
  output_path.parent.mkdir(parents=True, exist_ok=True)
  num_cols = min(3, len(sample_indices))
  num_rows = math.ceil(len(sample_indices) / num_cols)
  fig, axes = plt.subplots(
      num_rows,
      num_cols,
      figsize=(5.6 * num_cols, 5.2 * num_rows),
      squeeze=False,
  )
  for axis, sample_index in zip(axes.ravel(), sample_indices):
    target = target_xy[sample_index]
    recon = recon_xy[sample_index]
    axis.plot(target[:, 0], target[:, 1], "o-", color="tab:blue", label="target")
    axis.plot(recon[:, 0], recon[:, 1], "x--", color="tab:orange", label="residual FSQ")
    max_error = float((target - recon).abs().max())
    mse = float(F.mse_loss(recon, target))
    axis.set_title(
        f"sample={sample_index} tokens={int(token_counts[sample_index])} "
        f"max={max_error:.4f}m mse={mse:.6f}"
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
  for axis in axes.ravel()[len(sample_indices):]:
    axis.axis("off")
  handles, labels = axes.ravel()[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=2)
  fig.suptitle(title, y=0.98)
  fig.tight_layout(rect=(0, 0.05, 1, 0.94))
  fig.savefig(output_path, dpi=180)
  plt.close(fig)


def save_reconstruction_plots(
    model: ResidualFSQTokenizer,
    target: torch.Tensor,
    target_xy: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    max_tokens: int,
    threshold: float,
    stop_horizon_steps: int | None,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
    split_name: str,
    epoch: int | None = None,
) -> list[Path]:
  recon_parts = [[] for _ in range(max_tokens)]
  token_count_parts = []
  model.eval()
  with torch.no_grad():
    for batch_target, batch_xy in DataLoader(
        TensorDataset(target, target_xy), batch_size=batch_size
    ):
      batch_target = batch_target.to(device)
      batch_xy = batch_xy.to(device)
      recons, _, _ = model(batch_target, max_tokens=max_tokens)
      token_count_parts.append(
          stop_token_counts_by_xy(
              recons,
              batch_xy,
              threshold,
              stop_horizon_steps,
              feature_mean,
              feature_std,
          ).cpu()
      )
      for token_index, recon in enumerate(recons):
        recon_parts[token_index].append(
            target_features_to_xy(recon.cpu(), feature_mean.cpu(), feature_std.cpu())
        )

  reconstructions = [torch.cat(parts) for parts in recon_parts]
  token_counts = torch.cat(token_count_parts)
  stopped_recon = select_stopped_reconstructions(reconstructions, token_counts)
  output_dir.mkdir(parents=True, exist_ok=True)

  file_epoch = f"_epoch_{epoch:04d}" if epoch is not None else ""
  first_indices = list(range(min(num_samples, len(target_xy))))
  first_path = output_dir / f"reconstruction_{split_name}{file_epoch}_first_samples.png"
  plot_reconstructions(
      target_xy,
      stopped_recon,
      token_counts,
      first_path,
      f"Residual FSQ {split_name} stopped reconstruction, threshold={threshold:g}m",
      first_indices,
  )

  per_example_error = (stopped_recon - target_xy).abs().amax(dim=(1, 2))
  worst_indices = torch.topk(
      per_example_error,
      k=min(num_samples, len(target_xy)),
  ).indices.tolist()
  worst_path = output_dir / f"reconstruction_{split_name}{file_epoch}_worst_samples.png"
  plot_reconstructions(
      target_xy,
      stopped_recon,
      token_counts,
      worst_path,
      f"Residual FSQ {split_name} worst stopped reconstructions, threshold={threshold:g}m",
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
  parser.add_argument(
      "--target-space",
      choices=("xy", "xy_yaw_velocity"),
      default="xy",
      help="Feature space reconstructed by residual tokens.",
  )
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
  parser.add_argument("--plot-dir", default=None)
  parser.add_argument("--num-plot-samples", type=int, default=5)
  parser.add_argument(
      "--plot-every",
      type=int,
      default=100,
      help="Save train reconstruction plots every N epochs; 0 disables periodic plots.",
  )
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
  train_physical = train_x * std + mean
  val_physical = val_x * std + mean
  train_xy = train_physical[..., :2]
  val_xy = val_physical[..., :2]
  feature_indices = [0, 1] if args.target_space == "xy" else [0, 1, 2, 4, 5]
  physical_feature_mean = mean[..., feature_indices].squeeze(0)
  physical_feature_std = std[..., feature_indices].squeeze(0).clamp_min(1e-6)
  train_target = select_target_features(train_x, args.target_space)
  val_target = select_target_features(val_x, args.target_space)
  model_feature_mean = torch.zeros(train_target.shape[-1])
  model_feature_std = torch.ones(train_target.shape[-1])
  print(f"Parsed stats: {stats}")
  print(
      f"Split: train={len(train_target)} val={len(val_target)} "
      f"target_space={args.target_space}"
  )

  model = ResidualFSQTokenizer(
      num_steps=args.num_steps,
      target_dim=train_target.shape[-1],
      hidden_dim=args.hidden_dim,
      code_dim=args.code_dim,
      fsq_levels=args.fsq_levels,
      fsq_input_scale=args.fsq_input_scale,
      feature_mean=model_feature_mean,
      feature_std=model_feature_std,
  ).to(device)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )
  loader = DataLoader(
      TensorDataset(train_target),
      batch_size=args.batch_size,
      shuffle=True,
      generator=torch.Generator().manual_seed(args.seed),
  )
  best_val_loss = float("inf")
  best_state = None
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  plot_dir = Path(args.plot_dir) if args.plot_dir else output_dir
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
            val_target.to(device),
            max_tokens=args.max_tokens,
            quantize_strength=1.0,
        )
        val_loss = float(F.mse_loss(val_recons[-1].cpu(), val_target))
        stop_counts = stop_token_counts_by_xy(
            val_recons,
            val_xy.to(device),
            args.stop_threshold,
            args.stop_horizon_steps or None,
            physical_feature_mean,
            physical_feature_std,
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
    if args.plot_every > 0 and epoch % args.plot_every == 0:
      plot_paths = save_reconstruction_plots(
          model,
          train_target,
          train_xy,
          physical_feature_mean,
          physical_feature_std,
          max_tokens=args.max_tokens,
          threshold=args.stop_threshold,
          stop_horizon_steps=args.stop_horizon_steps or None,
          batch_size=args.batch_size,
          device=device,
          output_dir=plot_dir,
          num_samples=args.num_plot_samples,
          split_name="train",
          epoch=epoch,
      )
      for plot_path in plot_paths:
        print(f"Saved plot: {plot_path}")

  if best_state is not None:
    model.load_state_dict(best_state)
  rows = evaluate_model(
      model,
      val_target,
      val_xy,
      physical_feature_mean,
      physical_feature_std,
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
          "feature_mean": physical_feature_mean,
          "feature_std": physical_feature_std,
          "xy_mean": physical_feature_mean[:2],
          "xy_std": physical_feature_std[:2],
      },
      output_dir / "best_model.pt",
  )
  print(f"Saved metrics: {csv_path}")
  print(f"Saved model: {output_dir / 'best_model.pt'}")
  plot_paths = save_reconstruction_plots(
      model,
      val_target,
      val_xy,
      physical_feature_mean,
      physical_feature_std,
      max_tokens=args.max_tokens,
      threshold=args.stop_threshold,
      stop_horizon_steps=args.stop_horizon_steps or None,
      batch_size=args.batch_size,
      device=device,
      output_dir=plot_dir,
      num_samples=args.num_plot_samples,
      split_name="val",
  )
  for plot_path in plot_paths:
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
  main()
