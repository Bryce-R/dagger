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
WAYMO_STEPS_PER_SECOND = 10


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


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
  """Wraps radians to [-pi, pi]."""
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def extract_waymo_trajectories(
    tfrecord_paths: list[Path],
    num_steps: int,
    max_trajectories: int | None,
    include_all_valid_tracks: bool,
    object_types: set[int],
    include_yaw: bool,
    include_all_states: bool,
    decode_absolute_positions: bool,
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
        if decode_absolute_positions:
          deltas = xy
        if include_yaw or include_all_states:
          headings = torch.tensor([state.heading for state in states], dtype=torch.float32)
          local_yaw = wrap_angle(headings - headings[:1])
          yaw_deltas = torch.zeros(num_steps, 1, dtype=torch.float32)
          yaw_deltas[1:, 0] = wrap_angle(local_yaw[1:] - local_yaw[:-1])
          deltas = torch.cat([deltas, yaw_deltas], dim=1)
        if include_all_states:
          z = torch.tensor([[state.center_z] for state in states], dtype=torch.float32)
          z = z - z[:1]
          z_deltas = torch.zeros_like(z)
          z_deltas[1:] = z[1:] - z[:-1]
          velocity = torch.tensor(
              [(state.velocity_x, state.velocity_y) for state in states],
              dtype=torch.float32,
          )
          velocity = rotate_to_local(velocity, states[0].heading)
          state_features = torch.cat([z_deltas, velocity], dim=1)
          deltas = torch.cat([deltas, state_features], dim=1)
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

  def __init__(self, levels: list[int], code_dim: int, input_scale: float):
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
    self.log_input_scale = nn.Parameter(
        torch.full((code_dim,), math.log(input_scale), dtype=torch.float32)
    )
    self.input_offset = nn.Parameter(torch.zeros(code_dim, dtype=torch.float32))

  @property
  def num_bins(self) -> int:
    return int(self.levels.max().item())

  def project(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_bt = z.permute(0, 2, 1).contiguous()
    scale = self.scale.view(1, 1, -1)
    input_scale = self.log_input_scale.exp().view(1, 1, -1)
    input_offset = self.input_offset.view(1, 1, -1)
    adapted = z_bt * input_scale + input_offset
    continuous = torch.tanh(adapted)
    bounded = (continuous + 1.0) * 0.5 * scale
    return adapted, continuous, bounded

  def forward(
      self, z: torch.Tensor, quantize_strength: float = 1.0
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_bt = z.permute(0, 2, 1).contiguous()
    scale = self.scale.view(1, 1, -1)
    _, continuous, bounded = self.project(z)
    indices = torch.round(bounded).long()
    quantized = (indices.float() / scale) * 2.0 - 1.0
    quantized = continuous + (quantized - continuous).detach()
    if quantize_strength < 1.0:
      quantized = (1.0 - quantize_strength) * continuous + quantize_strength * quantized
    zero_loss = z.sum() * 0.0
    return quantized.permute(0, 2, 1).contiguous(), zero_loss, indices


class IdentityQuantizer(nn.Module):
  """No-op quantizer used to measure the autoencoder reconstruction floor."""

  def forward(
      self, z: torch.Tensor, quantize_strength: float = 1.0
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.zeros(
        z.shape[0], z.shape[2], dtype=torch.long, device=z.device
    )
    return z, z.sum() * 0.0, indices


class MLPEncoder(nn.Module):
  """Flattened fixed-horizon trajectory encoder for [B, T, D] inputs."""

  def __init__(
      self,
      num_steps: int,
      input_dim: int,
      hidden_dim: int,
      code_dim: int,
      latent_tokens: int,
      dropout: float,
  ):
    super().__init__()
    self.num_steps = num_steps
    self.input_dim = input_dim
    self.code_dim = code_dim
    self.latent_tokens = latent_tokens
    self.net = nn.Sequential(
        nn.Linear(num_steps * input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, code_dim * latent_tokens),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(x.shape[0], self.num_steps * self.input_dim)
    z = self.net(flat)
    return z.view(x.shape[0], self.latent_tokens, self.code_dim).permute(0, 2, 1)


class MLPDecoder(nn.Module):
  """Flattened fixed-horizon trajectory decoder for quantized latents."""

  def __init__(
      self,
      num_steps: int,
      output_dim: int,
      hidden_dim: int,
      code_dim: int,
      latent_tokens: int,
      dropout: float,
  ):
    super().__init__()
    self.num_steps = num_steps
    self.output_dim = output_dim
    self.net = nn.Sequential(
        nn.Linear(code_dim * latent_tokens, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_steps * output_dim),
    )

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    flat = z.permute(0, 2, 1).contiguous().view(z.shape[0], -1)
    return self.net(flat).view(z.shape[0], self.num_steps, self.output_dim)


class TrajectoryVQVAE(nn.Module):
  """Quantized autoencoder for [B, T, 2] trajectories."""

  def __init__(
      self,
      architecture: str,
      num_steps: int,
      input_dim: int,
      hidden_dim: int,
      code_dim: int,
      mlp_latent_tokens: int,
      mlp_dropout: float,
      num_codes: int,
      quantizer: str,
      fsq_levels: list[int],
      fsq_input_scale: float,
      vq_loss_mode: str,
  ):
    super().__init__()
    self.architecture = architecture
    if architecture == "conv":
      self.encoder = nn.Sequential(
          nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
          nn.ReLU(),
          nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv1d(hidden_dim, code_dim, kernel_size=3, padding=1),
      )
      self.decoder = nn.Sequential(
          nn.Conv1d(code_dim, hidden_dim, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv1d(hidden_dim, input_dim, kernel_size=5, padding=2),
      )
    elif architecture == "mlp":
      self.encoder = MLPEncoder(
          num_steps, input_dim, hidden_dim, code_dim, mlp_latent_tokens, mlp_dropout
      )
      self.decoder = MLPDecoder(
          num_steps, input_dim, hidden_dim, code_dim, mlp_latent_tokens, mlp_dropout
      )
    else:
      raise ValueError(f"Unknown architecture: {architecture}")
    self.quantizer_type = quantizer
    if quantizer == "vq":
      self.quantizer = VectorQuantizer(num_codes, code_dim, vq_loss_mode)
      self.quantizer_bins = num_codes
    elif quantizer == "fsq":
      self.quantizer = FiniteScalarQuantizer(
          fsq_levels, code_dim, fsq_input_scale
      )
      self.quantizer_bins = self.quantizer.num_bins
    elif quantizer == "none":
      self.quantizer = IdentityQuantizer()
      self.quantizer_bins = 1
    else:
      raise ValueError(f"Unknown quantizer: {quantizer}")

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    if self.architecture == "conv":
      return self.encoder(x.permute(0, 2, 1))
    return self.encoder(x)

  def decode(self, z: torch.Tensor, num_steps: int) -> torch.Tensor:
    if self.architecture == "conv":
      recon = self.decoder(z).permute(0, 2, 1)
      return recon[:, :num_steps]
    return self.decoder(z)

  def forward(
      self, x: torch.Tensor, quantize_strength: float = 1.0
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = self.encode(x)
    if self.quantizer_type == "fsq":
      quantized, vq_loss, indices = self.quantizer(z, quantize_strength)
    else:
      quantized, vq_loss, indices = self.quantizer(z)
    recon = self.decode(quantized, x.shape[1])
    return recon, vq_loss, indices


def deltas_to_positions(deltas: torch.Tensor) -> torch.Tensor:
  return torch.cumsum(deltas[..., :2], dim=1)


def xy_to_positions(xy_values: torch.Tensor, decode_absolute_positions: bool) -> torch.Tensor:
  if decode_absolute_positions:
    return xy_values[..., :2]
  return deltas_to_positions(xy_values)


def deltas_to_yaw(deltas: torch.Tensor) -> torch.Tensor:
  if deltas.shape[-1] < 3:
    return torch.empty(*deltas.shape[:2], 0, device=deltas.device)
  return wrap_angle(torch.cumsum(deltas[..., 2:3], dim=1))


def save_reconstruction_plot(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    quantize_strength: float = 1.0,
    num_plot_samples: int = 10,
    decode_absolute_positions: bool = False,
) -> Path:
  """Saves target/reconstruction plots for representative training samples."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  model.eval()
  output_dir.mkdir(parents=True, exist_ok=True)
  num_samples = min(num_plot_samples, len(x))
  sample_indices = list(range(num_samples))
  sample_names = [
      f"{TRACK_TYPE_NAMES.get(int(labels[index]), str(int(labels[index])))} {index}"
      for index in sample_indices
  ]

  sample_x = x[sample_indices].to(device)
  with torch.no_grad():
    recon, _, token_indices = model(sample_x, quantize_strength=quantize_strength)

  mean = mean.to(device)
  std = std.to(device)
  target_delta = sample_x * std + mean
  recon_delta = recon * std + mean
  target_xy = xy_to_positions(target_delta, decode_absolute_positions).cpu()
  recon_xy = xy_to_positions(recon_delta, decode_absolute_positions).cpu()
  target_yaw = deltas_to_yaw(target_delta).cpu()
  recon_yaw = deltas_to_yaw(recon_delta).cpu()

  num_cols = min(5, len(sample_indices))
  num_rows = math.ceil(len(sample_indices) / num_cols)
  fig, axes = plt.subplots(
      num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), squeeze=False
  )
  flat_axes = axes.ravel()
  for axis, name, target, pred, target_heading, pred_heading, tokens in zip(
      flat_axes, sample_names, target_xy, recon_xy, target_yaw, recon_yaw, token_indices.cpu()
  ):
    axis.plot(target[:, 0], target[:, 1], label="target", linewidth=2)
    axis.plot(pred[:, 0], pred[:, 1], label="recon", linewidth=2, linestyle="--")
    axis.scatter(target[0, 0], target[0, 1], s=20, marker="o", color="black")
    if target_heading.numel() > 0:
      yaw_mae = (target_heading - pred_heading).abs().mean().item()
      axis.set_title(f"{name}: {len(torch.unique(tokens))} codes, yaw_mae={yaw_mae:.2f}")
    else:
      axis.set_title(f"{name}: {len(torch.unique(tokens))} codes")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
  for axis in flat_axes[len(sample_indices):]:
    axis.axis("off")
  flat_axes[0].legend(loc="best")
  fig.suptitle(
      f"Waymo trajectory VQ-VAE reconstruction, epoch {epoch}, "
      f"quantize_strength={quantize_strength:.2f}"
  )
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
    val_key = f"val_{key}"
    if val_key in metrics[0]:
      axis.plot(epochs, [row[val_key] for row in metrics], linewidth=1.8, linestyle="--")
      axis.legend(["train", "val"], loc="best")
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
    decode_absolute_positions: bool,
) -> torch.Tensor:
  mean = mean.to(target.device)
  std = std.to(target.device)
  recon_xy = xy_to_positions(recon * std + mean, decode_absolute_positions)
  target_xy = xy_to_positions(target * std + mean, decode_absolute_positions)
  return F.smooth_l1_loss(recon_xy, target_xy)


def evaluate_reconstruction(
    model: nn.Module,
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    decode_absolute_positions: bool,
) -> dict[str, float]:
  """Computes full-dataset reconstruction metrics without gradient updates."""
  loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
  total_recon = 0.0
  total_pos = 0.0
  total_mse = 0.0
  total_examples = 0

  model.eval()
  with torch.no_grad():
    for (batch_x,) in loader:
      batch_x = batch_x.to(device)
      recon, _, _ = model(batch_x)
      batch_size_actual = batch_x.shape[0]
      total_recon += float(F.smooth_l1_loss(recon, batch_x).detach()) * batch_size_actual
      total_pos += (
          float(
              absolute_position_loss(
                  recon, batch_x, mean, std, decode_absolute_positions
              ).detach()
          )
          * batch_size_actual
      )
      total_mse += float(F.mse_loss(recon, batch_x).detach()) * batch_size_actual
      total_examples += batch_size_actual

  return {
      "recon": total_recon / total_examples,
      "pos_loss": total_pos / total_examples,
      "mse": total_mse / total_examples,
  }


def max_abs_position_error_by_second(
    model: nn.Module,
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    decode_absolute_positions: bool,
    steps_per_second: int = WAYMO_STEPS_PER_SECOND,
) -> dict[int, float]:
  """Returns worst absolute XY reconstruction error at each whole second."""
  num_steps = x.shape[1]
  max_seconds = math.ceil(num_steps / steps_per_second)
  second_to_index = {
      second: min(second * steps_per_second - 1, num_steps - 1)
      for second in range(1, max_seconds + 1)
  }
  max_errors = {second: 0.0 for second in second_to_index}
  mean = mean.to(device)
  std = std.to(device)
  loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)

  model.eval()
  with torch.no_grad():
    for (batch_x,) in loader:
      batch_x = batch_x.to(device)
      recon, _, _ = model(batch_x)
      recon_xy = xy_to_positions(recon * std + mean, decode_absolute_positions)
      target_xy = xy_to_positions(batch_x * std + mean, decode_absolute_positions)
      abs_error = (recon_xy - target_xy).abs()
      for second, index in second_to_index.items():
        max_errors[second] = max(
            max_errors[second], float(abs_error[:, index].max().detach().cpu())
        )
  return max_errors


def codebook_perplexity(indices: torch.Tensor, num_codes: int) -> float:
  counts = torch.bincount(indices.reshape(-1), minlength=num_codes).float()
  probs = counts / counts.sum().clamp_min(1.0)
  entropy = -(probs * (probs + 1e-10).log()).sum()
  return float(torch.exp(entropy))


def quantize_strength_for_epoch(args: argparse.Namespace, epoch: int) -> float:
  if args.quantizer != "fsq":
    return 1.0
  if epoch <= args.quantize_warmup_epochs:
    return 0.0
  if args.quantize_anneal_epochs <= 0:
    return 1.0
  progress = (epoch - args.quantize_warmup_epochs) / args.quantize_anneal_epochs
  return min(1.0, max(0.0, progress))


def tensor_stats(name: str, value: torch.Tensor) -> str:
  value = value.detach().float().cpu()
  return (
      f"{name}_mean={value.mean().item():.4f} "
      f"{name}_std={value.std().item():.4f} "
      f"{name}_min={value.min().item():.4f} "
      f"{name}_max={value.max().item():.4f}"
  )


def fsq_latent_stats(
    model: nn.Module,
    sample_x: torch.Tensor,
    num_bins: int,
) -> str:
  if not isinstance(getattr(model, "quantizer", None), FiniteScalarQuantizer):
    return ""
  model.eval()
  with torch.no_grad():
    z = model.encode(sample_x)
    adapted, continuous, bounded = model.quantizer.project(z)
    indices = torch.round(bounded).long().cpu()
    input_scale = model.quantizer.log_input_scale.exp().detach().float().cpu()
  return (
      tensor_stats("z", z)
      + " "
      + tensor_stats("adapted", adapted)
      + " "
      + tensor_stats("tanh", continuous)
      + " "
      + f"input_scale_mean={input_scale.mean().item():.4f} "
      + f"input_scale_min={input_scale.min().item():.4f} "
      + f"input_scale_max={input_scale.max().item():.4f} "
      + f"sample_unique_bins={len(torch.unique(indices))} "
      + f"sample_perplexity={codebook_perplexity(indices, num_bins):.2f}"
  )


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
      include_yaw=args.include_yaw,
      include_all_states=args.include_all_states,
      decode_absolute_positions=args.decode_absolute_positions,
  )
  label_counts = {
      TRACK_TYPE_NAMES.get(int(label), str(int(label))): int((labels == label).sum())
      for label in torch.unique(labels)
  }
  print(f"Parsed stats: {stats}")
  print(f"Trajectory tensor: {tuple(x.shape)} label_counts={label_counts}")

  split_generator = torch.Generator().manual_seed(args.seed)
  permutation = torch.randperm(len(x), generator=split_generator)
  val_size = int(round(len(x) * args.val_fraction))
  if args.val_fraction > 0.0:
    val_size = max(1, min(len(x) - 1, val_size))
  train_indices = permutation[val_size:]
  val_indices = permutation[:val_size]
  train_x = x[train_indices]
  train_labels = labels[train_indices]
  val_x = x[val_indices] if val_size > 0 else x[:0]
  val_labels = labels[val_indices] if val_size > 0 else labels[:0]
  print(f"Split: train={len(train_x)} val={len(val_x)} val_fraction={args.val_fraction:.3f}")

  train_dataset = TensorDataset(train_x, train_labels)
  loader_generator = torch.Generator().manual_seed(args.seed)
  loader = DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=True,
      generator=loader_generator,
  )
  latent_stats_x = x[: min(args.latent_stats_samples, len(x))].to(device)
  model = TrajectoryVQVAE(
      architecture=args.architecture,
      num_steps=args.num_steps,
      input_dim=x.shape[-1],
      hidden_dim=args.hidden_dim,
      code_dim=args.code_dim,
      mlp_latent_tokens=args.mlp_latent_tokens,
      mlp_dropout=args.mlp_dropout,
      num_codes=args.num_codes,
      quantizer=args.quantizer,
      fsq_levels=args.fsq_levels,
      fsq_input_scale=args.fsq_input_scale,
      vq_loss_mode=args.vq_loss_mode,
  ).to(device)
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=args.lr, weight_decay=args.weight_decay
  )
  saved_plot_paths = []
  pdf_path = None
  if args.plot_every > 0:
    pdf_path = Path(args.plot_pdf) if args.plot_pdf else Path(args.plot_dir) / "reconstructions.pdf"
  metrics = []

  for epoch in range(1, args.epochs + 1):
    quantize_strength = quantize_strength_for_epoch(args, epoch)
    model.train()
    total_recon = 0.0
    total_pos = 0.0
    total_vq = 0.0
    batches = 0
    last_indices = None
    for batch_x, _ in loader:
      batch_x = batch_x.to(device)
      recon, vq_loss, indices = model(
          batch_x, quantize_strength=quantize_strength
      )
      recon_loss = F.smooth_l1_loss(recon, batch_x)
      pos_loss = absolute_position_loss(
          recon, batch_x, mean, std, args.decode_absolute_positions
      )
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
        "quantize_strength": quantize_strength,
    }
    if len(val_x) > 0:
      val_metrics = evaluate_reconstruction(
          model=model,
          x=val_x,
          mean=mean,
          std=std,
          device=device,
          batch_size=args.batch_size,
          decode_absolute_positions=args.decode_absolute_positions,
      )
      epoch_metrics.update(
          {
              "val_recon": val_metrics["recon"],
              "val_pos_loss": val_metrics["pos_loss"],
              "val_mse": val_metrics["mse"],
          }
      )
    metrics.append(epoch_metrics)

    if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
      val_log = ""
      if len(val_x) > 0:
        val_log = (
            f" val_recon={epoch_metrics['val_recon']:.5f} "
            f"val_pos_loss={epoch_metrics['val_pos_loss']:.5f} "
            f"val_mse={epoch_metrics['val_mse']:.5f}"
        )
      print(
          f"epoch={epoch:04d} "
          f"recon={epoch_metrics['recon']:.5f} "
          f"pos_loss={epoch_metrics['pos_loss']:.5f} "
          f"{val_log} "
          f"vq={epoch_metrics['vq']:.5f} "
          f"last_batch_perplexity={perplexity:.2f} "
          f"quantize_strength={quantize_strength:.2f}"
      )
    if (
        args.latent_stats_every > 0
        and (epoch == 1 or epoch % args.latent_stats_every == 0 or epoch == args.epochs)
    ):
      print(
          f"latent_stats epoch={epoch:04d} "
          + fsq_latent_stats(model, latent_stats_x, model.quantizer_bins)
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
              quantize_strength=quantize_strength,
              num_plot_samples=args.num_plot_samples,
              decode_absolute_positions=args.decode_absolute_positions,
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
  second_errors = max_abs_position_error_by_second(
      model=model,
      x=x,
      mean=mean,
      std=std,
      device=device,
      batch_size=args.batch_size,
      decode_absolute_positions=args.decode_absolute_positions,
  )
  formatted_second_errors = " ".join(
      f"{second}s={error:.6f}" for second, error in second_errors.items()
  )
  print(f"Final max abs XY position error by second: {formatted_second_errors}")
  final_train_metrics = evaluate_reconstruction(
      model=model,
      x=train_x,
      mean=mean,
      std=std,
      device=device,
      batch_size=args.batch_size,
      decode_absolute_positions=args.decode_absolute_positions,
  )
  print(
      "Final train metrics: "
      f"recon={final_train_metrics['recon']:.6f} "
      f"pos_loss={final_train_metrics['pos_loss']:.6f} "
      f"mse={final_train_metrics['mse']:.6f}"
  )
  if len(val_x) > 0:
    final_val_metrics = evaluate_reconstruction(
        model=model,
        x=val_x,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.batch_size,
        decode_absolute_positions=args.decode_absolute_positions,
    )
    val_second_errors = max_abs_position_error_by_second(
        model=model,
        x=val_x,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.batch_size,
        decode_absolute_positions=args.decode_absolute_positions,
    )
    formatted_val_second_errors = " ".join(
        f"{second}s={error:.6f}" for second, error in val_second_errors.items()
    )
    print(
        "Final val metrics: "
        f"recon={final_val_metrics['recon']:.6f} "
        f"pos_loss={final_val_metrics['pos_loss']:.6f} "
        f"mse={final_val_metrics['mse']:.6f}"
    )
    print(f"Final val max abs XY position error by second: {formatted_val_second_errors}")

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
  parser.add_argument(
      "--include-yaw",
      action="store_true",
      help="Append local yaw delta as a third trajectory channel.",
  )
  parser.add_argument(
      "--include-all-states",
      action="store_true",
      help=(
          "Append yaw delta, z delta, and local velocity state channels "
          "to the XY delta trajectory."
      ),
  )
  parser.add_argument(
      "--decode-absolute-positions",
      action="store_true",
      help="Train the first two channels as local absolute XY positions instead of XY deltas.",
  )
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
  parser.add_argument(
      "--architecture",
      choices=("conv", "mlp"),
      default="conv",
      help="Autoencoder architecture. Conv preserves the original temporal model.",
  )
  parser.add_argument("--hidden-dim", type=int, default=64)
  parser.add_argument("--code-dim", type=int, default=32)
  parser.add_argument(
      "--mlp-latent-tokens",
      type=int,
      default=1,
      help="Number of latent tokens used by the MLP architecture.",
  )
  parser.add_argument(
      "--mlp-dropout",
      type=float,
      default=0.0,
      help="Dropout probability applied after MLP hidden activations.",
  )
  parser.add_argument("--num-codes", type=int, default=64)
  parser.add_argument("--quantizer", choices=("vq", "fsq", "none"), default="fsq")
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
  parser.add_argument(
      "--fsq-input-scale",
      type=float,
      default=0.1,
      help="Initial learnable input scale applied before FSQ tanh quantization.",
  )
  parser.add_argument(
      "--quantize-warmup-epochs",
      type=int,
      default=0,
      help="For FSQ, train with continuous tanh latents for this many epochs.",
  )
  parser.add_argument(
      "--quantize-anneal-epochs",
      type=int,
      default=0,
      help="For FSQ, linearly blend continuous latents into quantized latents.",
  )
  parser.add_argument("--vq-loss-mode", choices=("single", "split"), default="single")
  parser.add_argument("--position-loss-weight", type=float, default=0.0)
  parser.add_argument(
      "--val-fraction",
      type=float,
      default=0.2,
      help="Fraction of extracted trajectories held out for validation metrics.",
  )
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--log-every", type=int, default=5)
  parser.add_argument(
      "--latent-stats-every",
      type=int,
      default=0,
      help="For FSQ, print encoder/pre-tanh/bin usage stats every N epochs.",
  )
  parser.add_argument(
      "--latent-stats-samples",
      type=int,
      default=64,
      help="Number of fixed samples to use for latent stats.",
  )
  parser.add_argument(
      "--plot-every",
      type=int,
      default=5,
      help="Save reconstruction plots every N epochs. Use 0 to disable.",
  )
  parser.add_argument(
      "--num-plot-samples",
      type=int,
      default=10,
      help="Number of representative trajectories to show in reconstruction plots.",
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
