"""Train a small trajectory VQ-VAE on Waymo Motion Scenario TFRecords.

This script reads serialized Scenario protos directly from uncompressed TFRecord
shards, extracts valid 2D future trajectories for tracks_to_predict, converts
them into each agent's local frame, and trains a temporal VQ-VAE on normalized
motion deltas.

Example:
  # From the dagger repo root:
  python train_waymo_trajectory_vqvae.py \
    --tfrecord /path/to/uncompressed_scenario_training_training.tfrecord-00000-of-01000 \
    --device mps \
    --epochs 20

  # Long Waymo vehicle VQ-VAE run from the dagger repo root, using the TFRecord
  # stored in the local waymo-open-dataset checkout:
  env MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp \
    python train_waymo_trajectory_vqvae.py \
      --tfrecord /Users/pengkai/Code/waymo-open-dataset/uncompressed_scenario_training_training.tfrecord-00000-of-01000 \
      --epochs 50000 \
      --batch-size 512 \
      --plot-every 5000 \
      --log-every 1000 \
      --plot-dir waymo_vehicle_trajectory_vqvae_plots_posw0_e50000_b512_dagger

  # If waymo_open_dataset is not installed, point this at a local checkout:
  WAYMO_OPEN_DATASET_ROOT=/Users/pengkai/Code/waymo-open-dataset \
    python train_waymo_trajectory_vqvae.py --tfrecord /path/to/scenario.tfrecord
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


TRACK_TYPE_NAMES = {
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}


def ensure_waymo_scenario_pb2():
  """Imports Scenario proto bindings, generating temporary ones if needed."""
  try:
    return importlib.import_module("waymo_open_dataset.protos.scenario_pb2")
  except ImportError:
    repo_root = Path(
        os.environ.get(
            "WAYMO_OPEN_DATASET_ROOT",
            Path(__file__).resolve().parents[1] / "waymo-open-dataset",
        )
    )
    if not (repo_root / "src" / "waymo_open_dataset" / "protos" / "scenario.proto").exists():
      raise FileNotFoundError(
          "Could not import waymo_open_dataset.protos.scenario_pb2 and could not "
          "find a Waymo Open Dataset checkout. Install waymo-open-dataset-tf-* "
          "or set WAYMO_OPEN_DATASET_ROOT to the local waymo-open-dataset repo."
      )
    proto_out = Path(tempfile.mkdtemp(prefix="waymo_pb2_"))
    proto_files = [
        "waymo_open_dataset/protos/scenario.proto",
        "waymo_open_dataset/protos/map.proto",
        "waymo_open_dataset/protos/vector.proto",
        "waymo_open_dataset/protos/camera_tokens.proto",
        "waymo_open_dataset/protos/compressed_lidar.proto",
        "waymo_open_dataset/dataset.proto",
        "waymo_open_dataset/label.proto",
        "waymo_open_dataset/protos/keypoint.proto",
        "waymo_open_dataset/protos/box.proto",
    ]
    subprocess.run(
        [
            "protoc",
            f"-I{repo_root / 'src'}",
            f"--python_out={proto_out}",
            *[str(repo_root / "src" / path) for path in proto_files],
        ],
        check=True,
    )
    sys.path.insert(0, str(proto_out))
    return importlib.import_module("waymo_open_dataset.protos.scenario_pb2")


def iter_tfrecord_records(path: Path):
  """Yields raw records from an uncompressed TFRecord file."""
  with path.open("rb") as file:
    while True:
      length_bytes = file.read(8)
      if not length_bytes:
        break
      if len(length_bytes) != 8:
        raise ValueError(f"Truncated TFRecord length header in {path}")
      length = struct.unpack("<Q", length_bytes)[0]
      file.read(4)  # length CRC
      data = file.read(length)
      file.read(4)  # data CRC
      if len(data) != length:
        raise ValueError(f"Truncated TFRecord payload in {path}")
      yield data


def rotate_to_local(xy: torch.Tensor, heading: float) -> torch.Tensor:
  """Rotates global xy offsets into the agent frame at the current timestep."""
  cos_h = math.cos(-heading)
  sin_h = math.sin(-heading)
  rot = torch.tensor(
      [[cos_h, -sin_h], [sin_h, cos_h]], dtype=torch.float32
  )
  return xy @ rot.T


def extract_waymo_trajectories(
    tfrecord_paths: list[Path],
    num_steps: int,
    max_trajectories: int | None,
    include_all_valid_tracks: bool,
    object_types: set[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
  """Returns normalized delta trajectories and labels from Waymo scenarios."""
  scenario_pb2 = ensure_waymo_scenario_pb2()
  trajectories = []
  labels = []
  stats = {
      "scenarios": 0,
      "candidate_tracks": 0,
      "valid_trajectories": 0,
  }

  for tfrecord_path in tfrecord_paths:
    for raw_record in iter_tfrecord_records(tfrecord_path):
      scenario = scenario_pb2.Scenario()
      scenario.ParseFromString(raw_record)
      stats["scenarios"] += 1

      if include_all_valid_tracks:
        track_indices = range(len(scenario.tracks))
      else:
        track_indices = [p.track_index for p in scenario.tracks_to_predict]

      start = scenario.current_time_index
      end = start + num_steps
      for track_index in track_indices:
        stats["candidate_tracks"] += 1
        track = scenario.tracks[track_index]
        if track.object_type not in object_types:
          continue
        states = track.states[start:end]
        if len(states) != num_steps or not all(state.valid for state in states):
          continue

        xy = torch.tensor(
            [(state.center_x, state.center_y) for state in states],
            dtype=torch.float32,
        )
        xy = xy - xy[:1]
        xy = rotate_to_local(xy, states[0].heading)
        deltas = torch.zeros_like(xy)
        deltas[1:] = xy[1:] - xy[:-1]
        trajectories.append(deltas)
        labels.append(track.object_type)
        stats["valid_trajectories"] += 1

        if max_trajectories is not None and len(trajectories) >= max_trajectories:
          return normalize_trajectories(trajectories, labels, stats)

  return normalize_trajectories(trajectories, labels, stats)


def normalize_trajectories(
    trajectories: list[torch.Tensor],
    labels: list[int],
    stats: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
  if not trajectories:
    raise ValueError("No valid trajectories were extracted.")
  x = torch.stack(trajectories)
  y = torch.tensor(labels, dtype=torch.long)
  mean = x.mean(dim=(0, 1), keepdim=True)
  std = x.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
  x = (x - mean) / std
  return x, y, mean, std, stats


class VectorQuantizer(nn.Module):
  """Nearest-neighbor vector quantizer with straight-through gradients."""

  def __init__(self, num_codes: int, code_dim: int, loss_mode: str):
    super().__init__()
    self.codebook = nn.Embedding(num_codes, code_dim)
    self.loss_mode = loss_mode
    self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

  def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_bt = z.permute(0, 2, 1).contiguous()
    flat_z = z_bt.view(-1, z_bt.shape[-1])
    distances = (
        flat_z.pow(2).sum(dim=1, keepdim=True)
        - 2.0 * flat_z @ self.codebook.weight.t()
        + self.codebook.weight.pow(2).sum(dim=1)
    )
    indices = distances.argmin(dim=1)
    quantized = self.codebook(indices).view_as(z_bt)

    if self.loss_mode == "single":
      vq_loss = F.mse_loss(quantized, z_bt)
    elif self.loss_mode == "split":
      codebook_loss = F.mse_loss(quantized, z_bt.detach())
      commitment_loss = F.mse_loss(z_bt, quantized.detach())
      vq_loss = codebook_loss + 0.25 * commitment_loss
    else:
      raise ValueError(f"Unknown VQ loss mode: {self.loss_mode}")

    quantized = z_bt + (quantized - z_bt).detach()
    quantized = quantized.permute(0, 2, 1).contiguous()
    return quantized, vq_loss, indices.view(z.shape[0], z.shape[2])


class FiniteScalarQuantizer(nn.Module):
  """Straight-through finite scalar quantizer for channel-wise latent bins."""

  def __init__(self, levels: list[int], code_dim: int):
    super().__init__()
    if len(levels) == 1:
      levels = levels * code_dim
    if len(levels) != code_dim:
      raise ValueError(
          f"FSQ levels length ({len(levels)}) must be 1 or match code_dim ({code_dim})."
      )
    if any(level < 2 for level in levels):
      raise ValueError("All FSQ levels must be >= 2.")
    levels_tensor = torch.tensor(levels, dtype=torch.float32)
    self.register_buffer("levels", levels_tensor)
    self.register_buffer("scale", levels_tensor - 1.0)

  @property
  def num_bins(self) -> int:
    return int(self.levels.max().item())

  def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_bt = z.permute(0, 2, 1).contiguous()
    scale = self.scale.view(1, 1, -1)
    bounded = (torch.tanh(z_bt) + 1.0) * 0.5 * scale
    indices = torch.round(bounded).long()
    quantized = (indices.float() / scale) * 2.0 - 1.0
    quantized = z_bt + (quantized - z_bt).detach()
    zero_loss = z.sum() * 0.0
    return quantized.permute(0, 2, 1).contiguous(), zero_loss, indices


class TrajectoryVQVAE(nn.Module):
  """Small temporal convolutional quantized autoencoder for [B, T, 2] trajectories."""

  def __init__(
      self,
      hidden_dim: int,
      code_dim: int,
      num_codes: int,
      quantizer: str,
      fsq_levels: list[int],
      vq_loss_mode: str,
  ):
    super().__init__()
    self.encoder = nn.Sequential(
        nn.Conv1d(2, hidden_dim, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, code_dim, kernel_size=3, padding=1),
    )
    self.quantizer_type = quantizer
    if quantizer == "vq":
      self.quantizer = VectorQuantizer(num_codes, code_dim, vq_loss_mode)
      self.quantizer_bins = num_codes
    elif quantizer == "fsq":
      self.quantizer = FiniteScalarQuantizer(fsq_levels, code_dim)
      self.quantizer_bins = self.quantizer.num_bins
    else:
      raise ValueError(f"Unknown quantizer: {quantizer}")
    self.decoder = nn.Sequential(
        nn.Conv1d(code_dim, hidden_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(hidden_dim, 2, kernel_size=5, padding=2),
    )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = self.encoder(x.permute(0, 2, 1))
    quantized, vq_loss, indices = self.quantizer(z)
    recon = self.decoder(quantized).permute(0, 2, 1)
    return recon[:, : x.shape[1]], vq_loss, indices


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
) -> Path:
  """Saves target/reconstruction plots for representative training samples."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  model.eval()
  output_dir.mkdir(parents=True, exist_ok=True)
  num_samples = min(3, len(x))
  sample_indices = list(range(num_samples))
  sample_names = [
      f"{TRACK_TYPE_NAMES.get(int(labels[index]), str(int(labels[index])))} {index}"
      for index in sample_indices
  ]

  sample_x = x[sample_indices].to(device)
  with torch.no_grad():
    recon, _, token_indices = model(sample_x)

  mean = mean.to(device)
  std = std.to(device)
  target_xy = deltas_to_positions(sample_x * std + mean).cpu()
  recon_xy = deltas_to_positions(recon * std + mean).cpu()

  fig, axes = plt.subplots(
      1, len(sample_indices), figsize=(4 * len(sample_indices), 4), squeeze=False
  )
  for axis, name, target, pred, tokens in zip(
      axes[0], sample_names, target_xy, recon_xy, token_indices.cpu()
  ):
    axis.plot(target[:, 0], target[:, 1], label="target", linewidth=2)
    axis.plot(pred[:, 0], pred[:, 1], label="recon", linewidth=2, linestyle="--")
    axis.scatter(target[0, 0], target[0, 1], s=20, marker="o", color="black")
    axis.set_title(f"{name}: {len(torch.unique(tokens))} codes")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
  axes[0, 0].legend(loc="best")
  fig.suptitle(f"Waymo trajectory VQ-VAE reconstruction, epoch {epoch}")
  fig.tight_layout()
  output_path = output_dir / f"reconstruction_epoch_{epoch:04d}.png"
  fig.savefig(output_path, dpi=150)
  plt.close(fig)
  return output_path


def save_reconstruction_pdf(plot_paths: list[Path], pdf_path: Path) -> None:
  """Writes saved reconstruction plots to a PDF, newest epoch first."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.image as mpimg
  import matplotlib.pyplot as plt
  from matplotlib.backends.backend_pdf import PdfPages

  pdf_path.parent.mkdir(parents=True, exist_ok=True)
  with PdfPages(pdf_path) as pdf_pages:
    for plot_path in reversed(plot_paths):
      image = mpimg.imread(plot_path)
      height, width = image.shape[:2]
      fig_width = 12.0
      fig_height = fig_width * height / width
      fig, axis = plt.subplots(figsize=(fig_width, fig_height))
      axis.imshow(image)
      axis.axis("off")
      fig.tight_layout(pad=0)
      pdf_pages.savefig(fig)
      plt.close(fig)


def save_metrics_csv(metrics: list[dict[str, float]], output_path: Path) -> None:
  """Writes per-epoch metrics to CSV."""
  if not metrics:
    return
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(metrics[0].keys()))
    writer.writeheader()
    writer.writerows(metrics)


def save_metrics_plot(metrics: list[dict[str, float]], output_paths: list[Path]) -> None:
  """Plots training metrics over epochs."""
  if not metrics:
    return

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  epochs = [row["epoch"] for row in metrics]
  fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
  plots = [
      ("recon", "Delta reconstruction loss"),
      ("pos_loss", "Absolute position loss"),
      ("vq", "Quantizer loss"),
      ("perplexity", "Last-batch bin perplexity"),
  ]
  for axis, (key, title) in zip(axes.ravel(), plots):
    axis.plot(epochs, [row[key] for row in metrics], linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.grid(True, alpha=0.3)
  if any(row["vq"] > 0.0 for row in metrics):
    axes[1, 0].set_yscale("log")
  axes[1, 1].set_ylim(bottom=0)
  fig.tight_layout()
  for output_path in output_paths:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
  plt.close(fig)


def absolute_position_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
  mean = mean.to(target.device)
  std = std.to(target.device)
  recon_xy = deltas_to_positions(recon * std + mean)
  target_xy = deltas_to_positions(target * std + mean)
  return F.smooth_l1_loss(recon_xy, target_xy)


def codebook_perplexity(indices: torch.Tensor, num_codes: int) -> float:
  counts = torch.bincount(indices.reshape(-1), minlength=num_codes).float()
  probs = counts / counts.sum().clamp_min(1.0)
  entropy = -(probs * (probs + 1e-10).log()).sum()
  return float(torch.exp(entropy))


def select_device(device_arg: str) -> torch.device:
  if device_arg == "auto":
    if torch.backends.mps.is_available():
      return torch.device("mps")
    if torch.cuda.is_available():
      return torch.device("cuda")
    return torch.device("cpu")
  return torch.device(device_arg)


def train(args: argparse.Namespace) -> None:
  torch.manual_seed(args.seed)
  device = select_device(args.device)
  x, labels, mean, std, stats = extract_waymo_trajectories(
      tfrecord_paths=[Path(path) for path in args.tfrecord],
      num_steps=args.num_steps,
      max_trajectories=args.max_trajectories,
      include_all_valid_tracks=args.include_all_valid_tracks,
      object_types=set(args.object_type),
  )
  label_counts = {
      TRACK_TYPE_NAMES.get(int(label), str(int(label))): int((labels == label).sum())
      for label in torch.unique(labels)
  }
  print(f"Parsed stats: {stats}")
  print(f"Trajectory tensor: {tuple(x.shape)} label_counts={label_counts}")

  dataset = TensorDataset(x, labels)
  loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
  model = TrajectoryVQVAE(
      hidden_dim=args.hidden_dim,
      code_dim=args.code_dim,
      num_codes=args.num_codes,
      quantizer=args.quantizer,
      fsq_levels=args.fsq_levels,
      vq_loss_mode=args.vq_loss_mode,
  ).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
  saved_plot_paths = []
  pdf_path = None
  if args.plot_every > 0:
    pdf_path = Path(args.plot_pdf) if args.plot_pdf else Path(args.plot_dir) / "reconstructions.pdf"
  metrics = []

  for epoch in range(1, args.epochs + 1):
    model.train()
    total_recon = 0.0
    total_pos = 0.0
    total_vq = 0.0
    batches = 0
    last_indices = None
    for batch_x, _ in loader:
      batch_x = batch_x.to(device)
      recon, vq_loss, indices = model(batch_x)
      recon_loss = F.smooth_l1_loss(recon, batch_x)
      pos_loss = absolute_position_loss(recon, batch_x, mean, std)
      loss = recon_loss + args.position_loss_weight * pos_loss + vq_loss
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()

      total_recon += float(recon_loss.detach())
      total_pos += float(pos_loss.detach())
      total_vq += float(vq_loss.detach())
      batches += 1
      last_indices = indices.detach().cpu()

    perplexity = codebook_perplexity(last_indices, model.quantizer_bins)
    epoch_metrics = {
        "epoch": epoch,
        "recon": total_recon / batches,
        "pos_loss": total_pos / batches,
        "vq": total_vq / batches,
        "perplexity": perplexity,
    }
    metrics.append(epoch_metrics)

    if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
      print(
          f"epoch={epoch:04d} "
          f"recon={epoch_metrics['recon']:.5f} "
          f"pos_loss={epoch_metrics['pos_loss']:.5f} "
          f"vq={epoch_metrics['vq']:.5f} "
          f"last_batch_perplexity={perplexity:.2f}"
      )
    if args.plot_every > 0 and (
        epoch == 1 or epoch % args.plot_every == 0 or epoch == args.epochs
    ):
      saved_plot_paths.append(
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
      )

  if pdf_path is not None and saved_plot_paths:
    save_reconstruction_pdf(saved_plot_paths, pdf_path)
    print(f"Saved reconstruction PDF: {pdf_path}")
  metrics_csv = Path(args.metrics_csv) if args.metrics_csv else Path(args.plot_dir) / "metrics.csv"
  metrics_plot = (
      Path(args.metrics_plot)
      if args.metrics_plot
      else Path(args.plot_dir) / "metrics.png"
  )
  metrics_plot_pdf = metrics_plot.with_suffix(".pdf")
  save_metrics_csv(metrics, metrics_csv)
  save_metrics_plot(metrics, [metrics_plot, metrics_plot_pdf])
  print(f"Saved metrics CSV: {metrics_csv}")
  print(f"Saved metrics plot: {metrics_plot}")
  print(f"Saved metrics plot PDF: {metrics_plot_pdf}")

  model.eval()
  with torch.no_grad():
    sample_x = x[: min(32, len(x))].to(device)
    recon, _, indices = model(sample_x)
    mse = F.mse_loss(recon, sample_x).item()
    used_codes = torch.unique(indices.cpu()).tolist()
  print(f"Final sample MSE: {mse:.5f}")
  print(f"Codes used by first {len(sample_x)} samples: {used_codes}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Train a VQ-VAE on Waymo Motion future trajectories."
  )
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--num-steps", type=int, default=50)
  parser.add_argument("--max-trajectories", type=int, default=None)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument(
      "--object-type",
      action="append",
      type=int,
      default=[1],
      help="Object type to include. Defaults to vehicle=1; repeat to include more.",
  )
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--hidden-dim", type=int, default=64)
  parser.add_argument("--code-dim", type=int, default=32)
  parser.add_argument("--num-codes", type=int, default=64)
  parser.add_argument("--quantizer", choices=("vq", "fsq"), default="vq")
  parser.add_argument(
      "--fsq-levels",
      type=int,
      nargs="+",
      default=[8],
      help=(
          "Scalar quantization levels for FSQ. A single value is repeated for "
          "all code_dim channels; otherwise the list must match code_dim."
      ),
  )
  parser.add_argument("--vq-loss-mode", choices=("single", "split"), default="single")
  parser.add_argument("--position-loss-weight", type=float, default=0.0)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--log-every", type=int, default=5)
  parser.add_argument(
      "--plot-every",
      type=int,
      default=5,
      help="Save reconstruction plots every N epochs. Use 0 to disable.",
  )
  parser.add_argument("--plot-dir", default="waymo_trajectory_vqvae_plots")
  parser.add_argument(
      "--plot-pdf",
      default=None,
      help=(
          "Multipage PDF path for reconstruction plots. Defaults to "
          "<plot-dir>/reconstructions.pdf; pages are newest epoch first."
      ),
  )
  parser.add_argument(
      "--metrics-csv",
      default=None,
      help="Path for per-epoch metrics CSV. Defaults to <plot-dir>/metrics.csv.",
  )
  parser.add_argument(
      "--metrics-plot",
      default=None,
      help="Path for metric curve PNG. Defaults to <plot-dir>/metrics.png.",
  )
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
  return parser.parse_args()


if __name__ == "__main__":
  train(parse_args())
