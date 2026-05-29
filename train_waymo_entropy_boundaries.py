"""Learn entropy-style trajectory chunk boundaries from 10Hz Waymo states.

This is a Byte Latent Transformer-style boundary experiment for continuous
motion: train a causal next-trajectory density model, then split chunks where
the upcoming 1s trajectory has low probability under the model.

Example:
  env PYTHONUNBUFFERED=1 MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp \
    python train_waymo_entropy_boundaries.py \
      --tfrecord /Users/pengkai/Code/waymo-open-dataset/uncompressed_scenario_training_training.tfrecord-00000-of-01000 \
      --device mps \
      --max-trajectories 2048 \
      --epochs 50 \
      --plot-dir waymo_entropy_boundaries_10hz
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_waymo_mlp_fsq import (
    ensure_waymo_scenario_pb2,
    iter_tfrecord_records,
    rotate_to_local,
    wrap_angle,
)


TRACK_TYPE_NAMES = {
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}
WAYMO_HZ = 10


class NextTrajectoryWindowDataset(Dataset):
  def __init__(
      self,
      history: torch.Tensor,
      target: torch.Tensor,
      traj_index: torch.Tensor,
      step_index: torch.Tensor,
  ):
    self.history = history
    self.target = target
    self.traj_index = traj_index
    self.step_index = step_index

  def __len__(self) -> int:
    return self.target.shape[0]

  def __getitem__(self, index: int):
    return (
        self.history[index],
        self.target[index],
        self.traj_index[index],
        self.step_index[index],
    )


class AutoregressiveTrajectoryGaussianGRU(nn.Module):
  def __init__(
      self,
      state_dim: int,
      horizon_len: int,
      hidden_dim: int,
      num_layers: int,
      dropout: float,
  ):
    super().__init__()
    self.state_dim = state_dim
    self.horizon_len = horizon_len
    self.gru = nn.GRU(
        input_size=state_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    self.decoder = nn.GRUCell(input_size=state_dim, hidden_size=hidden_dim)
    self.decoder_norm = nn.LayerNorm(hidden_dim)
    self.head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 2 * state_dim),
    )

  def forward(
      self,
      history: torch.Tensor,
      target: torch.Tensor | None = None,
      teacher_force: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    _, hidden = self.gru(history)
    decoder_hidden = hidden[-1]
    prev_state = history[:, -1]
    means = []
    log_stds = []
    for step in range(self.horizon_len):
      decoder_hidden = self.decoder(prev_state, decoder_hidden)
      params = self.head(self.decoder_norm(decoder_hidden))
      mean, raw_log_std = params.chunk(2, dim=-1)
      means.append(mean)
      log_stds.append(raw_log_std.clamp(-5.0, 2.0))
      if teacher_force and target is not None:
        prev_state = target[:, step]
      else:
        prev_state = mean
    return torch.stack(means, dim=1), torch.stack(log_stds, dim=1)


class EndpointGaussianGRU(nn.Module):
  def __init__(
      self,
      state_dim: int,
      target_dim: int,
      hidden_dim: int,
      num_layers: int,
      dropout: float,
  ):
    super().__init__()
    self.gru = nn.GRU(
        input_size=state_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
    )
    self.head = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 2 * target_dim),
    )

  def forward(
      self,
      history: torch.Tensor,
      target: torch.Tensor | None = None,
      teacher_force: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del target, teacher_force
    _, hidden = self.gru(history)
    mean, raw_log_std = self.head(hidden[-1]).chunk(2, dim=-1)
    return mean.unsqueeze(1), raw_log_std.clamp(-5.0, 2.0).unsqueeze(1)


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
  var = torch.exp(2.0 * log_std)
  per_dim = 0.5 * ((target - mean).pow(2) / var + 2.0 * log_std + math.log(2.0 * math.pi))
  return per_dim.flatten(start_dim=1).sum(dim=1)


def gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
  per_dim = log_std + 0.5 * math.log(2.0 * math.pi * math.e)
  return per_dim.flatten(start_dim=1).sum(dim=1)


def extract_10hz_waymo_states(
    tfrecord_paths: list[Path],
    num_steps: int,
    max_trajectories: int | None,
    include_all_valid_tracks: bool,
    object_types: set[int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
  """Returns physical 10Hz local state trajectories.

  State channels are local absolute x/y, local yaw, and local velocity x/y.
  This deliberately uses consecutive Waymo states, not the 2Hz stride used by
  some compression baselines in this repo.
  """
  scenario_pb2 = ensure_waymo_scenario_pb2()
  trajectories = []
  labels = []
  stats = {"scenarios": 0, "candidate_tracks": 0, "valid_trajectories": 0}

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

        xy = torch.tensor([(state.center_x, state.center_y) for state in states], dtype=torch.float32)
        xy = rotate_to_local(xy - xy[:1], states[0].heading)
        headings = torch.tensor([state.heading for state in states], dtype=torch.float32)
        local_yaw = wrap_angle(headings - headings[:1]).unsqueeze(-1)
        velocity = torch.tensor(
            [(state.velocity_x, state.velocity_y) for state in states],
            dtype=torch.float32,
        )
        velocity = rotate_to_local(velocity, states[0].heading)
        trajectories.append(torch.cat([xy, local_yaw, velocity], dim=-1))
        labels.append(track.object_type)
        stats["valid_trajectories"] += 1

        if max_trajectories is not None and len(trajectories) >= max_trajectories:
          return stack_states(trajectories, labels, stats)

  return stack_states(trajectories, labels, stats)


def extract_10hz_sdc_states(
    tfrecord_paths: list[Path],
    num_steps: int,
    max_trajectories: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
  """Returns physical 10Hz local state trajectories for the SDC/ego track."""
  scenario_pb2 = ensure_waymo_scenario_pb2()
  trajectories = []
  labels = []
  stats = {"scenarios": 0, "candidate_tracks": 0, "valid_trajectories": 0}

  for tfrecord_path in tfrecord_paths:
    for raw_record in iter_tfrecord_records(tfrecord_path):
      scenario = scenario_pb2.Scenario()
      scenario.ParseFromString(raw_record)
      stats["scenarios"] += 1
      stats["candidate_tracks"] += 1

      track = scenario.tracks[scenario.sdc_track_index]
      start = scenario.current_time_index
      end = start + num_steps
      states = track.states[start:end]
      if len(states) != num_steps or not all(state.valid for state in states):
        continue

      xy = torch.tensor([(state.center_x, state.center_y) for state in states], dtype=torch.float32)
      xy = rotate_to_local(xy - xy[:1], states[0].heading)
      headings = torch.tensor([state.heading for state in states], dtype=torch.float32)
      local_yaw = wrap_angle(headings - headings[:1]).unsqueeze(-1)
      velocity = torch.tensor(
          [(state.velocity_x, state.velocity_y) for state in states],
          dtype=torch.float32,
      )
      velocity = rotate_to_local(velocity, states[0].heading)
      trajectories.append(torch.cat([xy, local_yaw, velocity], dim=-1))
      labels.append(track.object_type)
      stats["valid_trajectories"] += 1

      if max_trajectories is not None and len(trajectories) >= max_trajectories:
        return stack_states(trajectories, labels, stats)

  return stack_states(trajectories, labels, stats)


def stack_states(
    trajectories: list[torch.Tensor],
    labels: list[int],
    stats: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
  if not trajectories:
    raise ValueError("No valid 10Hz Waymo trajectories were extracted.")
  x = torch.stack(trajectories)
  y = torch.tensor(labels, dtype=torch.long)
  return x, y, stats


def rotate_xy_by_angle(xy: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  cos_h = torch.cos(angle)
  sin_h = torch.sin(angle)
  x = xy[..., 0] * cos_h - xy[..., 1] * sin_h
  y = xy[..., 0] * sin_h + xy[..., 1] * cos_h
  return torch.stack([x, y], dim=-1)


def transform_to_anchor_frame(segment: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  anchor_xy = anchor[:2]
  anchor_yaw = anchor[2]
  rel_xy = rotate_xy_by_angle(segment[:, :2] - anchor_xy, -anchor_yaw)
  rel_yaw = wrap_angle(segment[:, 2] - anchor_yaw).unsqueeze(-1)
  rel_velocity = rotate_xy_by_angle(segment[:, 3:5], -anchor_yaw)
  return torch.cat([rel_xy, rel_yaw, rel_velocity], dim=-1)


def make_relative_windows(
    x: torch.Tensor,
    history_len: int,
    horizon_len: int,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  history = []
  target = []
  traj_index = []
  step_index = []
  for traj_idx in range(x.shape[0]):
    for step in range(history_len, x.shape[1] - horizon_len + 1):
      anchor = x[traj_idx, step - 1]
      history_segment = x[traj_idx, step - history_len:step]
      target_segment = x[traj_idx, step:step + horizon_len]
      history.append(transform_to_anchor_frame(history_segment, anchor))
      target_frame = transform_to_anchor_frame(target_segment, anchor)
      if target_mode == "full-trajectory":
        target.append(target_frame)
      elif target_mode == "final-xy":
        target.append(target_frame[-1:, :2])
      elif target_mode == "final-velocity":
        target.append(target_frame[-1:, 3:5])
      else:
        raise ValueError(f"unknown target_mode: {target_mode}")
      traj_index.append(traj_idx)
      step_index.append(step)
  return (
      torch.stack(history),
      torch.stack(target),
      torch.tensor(traj_index, dtype=torch.long),
      torch.tensor(step_index, dtype=torch.long),
  )


def fit_normalizer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  flat = x.reshape(-1, x.shape[-1])
  mean = flat.mean(dim=0).view(1, 1, -1)
  std = flat.std(dim=0).clamp_min(1e-6).view(1, 1, -1)
  return mean, std


def normalize_windows(
    history: torch.Tensor,
    target: torch.Tensor,
    history_mean: torch.Tensor,
    history_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  return (history - history_mean) / history_std, (target - target_mean) / target_std


def split_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  generator = torch.Generator().manual_seed(seed)
  permutation = torch.randperm(len(x), generator=generator)
  val_count = int(round(len(x) * val_fraction))
  val_idx = permutation[:val_count]
  train_idx = permutation[val_count:]
  return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train_model(
    model: nn.Module,
    train_history: torch.Tensor,
    train_target: torch.Tensor,
    train_traj_index: torch.Tensor,
    train_step_index: torch.Tensor,
    val_history: torch.Tensor,
    val_target: torch.Tensor,
    val_traj_index: torch.Tensor,
    val_step_index: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
  train_ds = NextTrajectoryWindowDataset(
      train_history,
      train_target,
      train_traj_index,
      train_step_index,
  )
  val_ds = NextTrajectoryWindowDataset(
      val_history,
      val_target,
      val_traj_index,
      val_step_index,
  )
  generator = torch.Generator().manual_seed(args.seed)
  train_loader = DataLoader(
      train_ds,
      batch_size=args.batch_size,
      shuffle=True,
      generator=generator,
  )
  val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  print(
      f"windows: train={len(train_ds)} val={len(val_ds)} "
      f"history={args.history_len} steps ({args.history_len / WAYMO_HZ:.1f}s) "
      f"horizon={args.horizon_len} steps ({args.horizon_len / WAYMO_HZ:.1f}s)"
  )
  for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0
    for history, target, _, _ in train_loader:
      history = history.to(device)
      target = target.to(device)
      mean, log_std = model(history, target=target, teacher_force=True)
      loss = gaussian_nll(target, mean, log_std).mean()
      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
      train_loss += loss.item() * target.shape[0]

    if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
      model.eval()
      val_loss = 0.0
      val_entropy = 0.0
      with torch.no_grad():
        for history, target, _, _ in val_loader:
          history = history.to(device)
          target = target.to(device)
          mean, log_std = model(history, target=target, teacher_force=True)
          val_loss += gaussian_nll(target, mean, log_std).sum().item()
          val_entropy += gaussian_entropy(log_std).sum().item()
      print(
          f"epoch={epoch:04d} "
          f"train_nll={train_loss / len(train_ds):.4f} "
          f"val_nll={val_loss / len(val_ds):.4f} "
          f"val_entropy={val_entropy / len(val_ds):.4f}"
      )


def score_trajectories(
    model: nn.Module,
    history: torch.Tensor,
    target: torch.Tensor,
    traj_index: torch.Tensor,
    step_index: torch.Tensor,
    num_trajectories: int,
    num_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  dataset = NextTrajectoryWindowDataset(history, target, traj_index, step_index)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  nll = torch.full((num_trajectories, num_steps), float("nan"))
  entropy = torch.full((num_trajectories, num_steps), float("nan"))
  model.eval()
  with torch.no_grad():
    for history, target, traj_idx, step_idx in loader:
      history = history.to(device)
      target = target.to(device)
      mean, log_std = model(history, target=target, teacher_force=True)
      batch_nll = gaussian_nll(target, mean, log_std).cpu()
      batch_entropy = gaussian_entropy(log_std).cpu()
      nll[traj_idx, step_idx] = batch_nll
      entropy[traj_idx, step_idx] = batch_entropy
  return nll, entropy


def speed_norm(states_physical: torch.Tensor) -> torch.Tensor:
  """Returns local velocity norm at 10Hz."""
  return states_physical[:, :, 3:5].norm(dim=-1)


def acceleration_change_score(states_physical: torch.Tensor) -> torch.Tensor:
  """Returns a jerk-like action-change proxy from local velocity at 10Hz."""
  velocity = states_physical[:, :, 3:5]
  acceleration = torch.zeros_like(velocity)
  acceleration[:, 1:] = (velocity[:, 1:] - velocity[:, :-1]) * WAYMO_HZ
  jerk = torch.zeros(states_physical.shape[:2], dtype=torch.float32)
  jerk[:, 2:] = (acceleration[:, 2:] - acceleration[:, 1:-1]).norm(dim=-1) * WAYMO_HZ
  return jerk


def choose_boundaries(
    scores: torch.Tensor,
    threshold: float,
    min_gap: int,
    max_boundaries: int,
) -> list[int]:
  finite_scores = torch.nan_to_num(scores, nan=-float("inf"))
  candidates = []
  for step in range(finite_scores.numel()):
    value = float(finite_scores[step])
    if value < threshold:
      continue
    candidates.append((value, step))
  candidates.sort(reverse=True)
  selected = []
  for _, step in candidates:
    if all(abs(step - existing) >= min_gap for existing in selected):
      selected.append(step)
      if len(selected) >= max_boundaries:
        break
  return sorted(selected)


def summarize_boundaries(
    nll: torch.Tensor,
    entropy: torch.Tensor,
    jerk: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[list[dict[str, float | int | str]], torch.Tensor]:
  valid_scores = nll[torch.isfinite(nll)]
  threshold = float(torch.quantile(valid_scores, args.boundary_quantile))
  boundaries = torch.zeros_like(nll, dtype=torch.bool)
  rows = []
  print(f"boundary_threshold: quantile={args.boundary_quantile:.3f} horizon_nll={threshold:.4f}")
  for traj_idx in range(nll.shape[0]):
    split_steps = choose_boundaries(
        nll[traj_idx],
        threshold,
        args.min_boundary_gap,
        args.max_boundaries_per_traj,
    )
    for step in split_steps:
      boundaries[traj_idx, step] = True
      rows.append(
          {
              "traj_index": traj_idx,
              "step": step,
              "time_s": step / WAYMO_HZ,
              "horizon_nll": float(nll[traj_idx, step]),
              "entropy": float(entropy[traj_idx, step]),
              "jerk_proxy": float(jerk[traj_idx, step]),
              "object_type": int(labels[traj_idx]),
              "object_type_name": TRACK_TYPE_NAMES.get(int(labels[traj_idx]), "unknown"),
          }
      )
    if traj_idx < args.print_examples:
      jerk_top = torch.topk(jerk[traj_idx], k=min(len(split_steps) or 3, jerk.shape[1])).indices.tolist()
      print(
          f"traj={traj_idx} type={TRACK_TYPE_NAMES.get(int(labels[traj_idx]), 'unknown')} "
          f"splits={split_steps} split_times={[round(step / WAYMO_HZ, 2) for step in split_steps]} "
          f"top_jerk_steps={sorted(int(step) for step in jerk_top)}"
      )
  avg_chunks = 1.0 + float(boundaries.sum(dim=1).float().mean())
  print(
      f"boundary_summary: trajectories={nll.shape[0]} "
      f"avg_chunks={avg_chunks:.2f} total_splits={int(boundaries.sum())}"
  )
  for row in rows[: args.print_examples]:
    print(
        "split "
        f"traj={row['traj_index']} "
        f"step={row['step']} "
        f"time_s={float(row['time_s']):.2f} "
        f"horizon_nll={float(row['horizon_nll']):.4f} "
        f"entropy={float(row['entropy']):.4f} "
        f"jerk_proxy={float(row['jerk_proxy']):.4f}"
    )
  return rows, boundaries


def write_boundary_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    path.write_text("")
    return
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def plot_boundaries(
    states_physical: torch.Tensor,
    nll: torch.Tensor,
    entropy: torch.Tensor,
    speed: torch.Tensor,
    boundaries: torch.Tensor,
    output_path: Path,
    sample_indices: list[int],
    title: str,
    score_label: str,
) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  sample_count = len(sample_indices)
  if sample_count == 0:
    return
  fig, axes = plt.subplots(sample_count, 3, figsize=(15.0, 3.4 * sample_count), squeeze=False)
  steps = torch.arange(states_physical.shape[1])

  for row, sample_index in enumerate(sample_indices):
    xy = states_physical[sample_index, :, :2]
    split_steps = torch.nonzero(boundaries[sample_index]).flatten().tolist()

    path_axis = axes[row, 0]
    path_axis.plot(xy[:, 0], xy[:, 1], "o-", markersize=2.5, color="tab:blue")
    if split_steps:
      split_xy = xy[split_steps]
      path_axis.scatter(
          split_xy[:, 0],
          split_xy[:, 1],
          s=80,
          facecolors="none",
          edgecolors="tab:red",
          linewidths=1.8,
      )
    path_axis.set_title(f"traj={sample_index} path splits={split_steps}")
    path_axis.set_aspect("equal", adjustable="datalim")
    path_axis.grid(True, alpha=0.3)

    score_axis = axes[row, 1]
    score_axis.plot(steps, nll[sample_index], color="tab:red", label=score_label)
    score_axis.plot(
        steps,
        entropy[sample_index],
        color="tab:purple",
        alpha=0.8,
        label="predictive entropy",
    )
    for step in split_steps:
      score_axis.axvline(step, color="black", alpha=0.35, linewidth=1.0)
    score_axis.set_title("future surprise / entropy")
    score_axis.set_xlabel("10Hz step")
    score_axis.legend(loc="best", fontsize=8)
    score_axis.grid(True, alpha=0.3)

    action_axis = axes[row, 2]
    action_axis.plot(steps, speed[sample_index], color="black", linewidth=2.0, label="speed norm")
    for step in split_steps:
      action_axis.axvline(step, color="black", alpha=0.35, linewidth=1.0)
    action_axis.set_title("velocity norm")
    action_axis.set_xlabel("10Hz step")
    action_axis.set_ylabel("m/s")
    action_axis.grid(True, alpha=0.3)

  handles, labels = axes[0, 1].get_legend_handles_labels()
  handles2, labels2 = axes[0, 2].get_legend_handles_labels()
  fig.legend(handles + handles2, labels + labels2, loc="lower center", ncol=3)
  fig.suptitle(title, y=0.995)
  fig.tight_layout(rect=(0, 0.035, 1, 0.98))
  fig.savefig(output_path, dpi=170)
  plt.close(fig)


def top_chunk_indices(boundaries: torch.Tensor, num_samples: int) -> list[int]:
  counts = boundaries.sum(dim=1)
  order = torch.argsort(counts, descending=True)
  return order[: min(num_samples, len(order))].tolist()


def random_sample_indices(num_trajectories: int, num_samples: int, seed: int) -> list[int]:
  generator = torch.Generator().manual_seed(seed)
  order = torch.randperm(num_trajectories, generator=generator)
  return order[: min(num_samples, num_trajectories)].tolist()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train 10Hz next-trajectory surprise boundaries on Waymo.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-steps", type=int, default=50, help="Raw 10Hz states per trajectory.")
  parser.add_argument("--history-len", type=int, default=8, help="Causal context length in 10Hz steps.")
  parser.add_argument("--horizon-len", type=int, default=10, help="Predicted future length in 10Hz steps.")
  parser.add_argument(
      "--target-mode",
      choices=("full-trajectory", "final-xy", "final-velocity"),
      default="full-trajectory",
      help="Prediction target used for boundary NLL.",
  )
  parser.add_argument(
      "--track-source",
      choices=("tracks-to-predict", "sdc"),
      default="tracks-to-predict",
      help="Train on Waymo target tracks or the SDC/ego track.",
  )
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--hidden-dim", type=int, default=128)
  parser.add_argument("--num-layers", type=int, default=1)
  parser.add_argument("--dropout", type=float, default=0.0)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--weight-decay", type=float, default=1e-4)
  parser.add_argument("--grad-clip", type=float, default=1.0)
  parser.add_argument("--log-every", type=int, default=10)
  parser.add_argument("--boundary-quantile", type=float, default=0.90)
  parser.add_argument("--min-boundary-gap", type=int, default=3)
  parser.add_argument("--max-boundaries-per-traj", type=int, default=8)
  parser.add_argument("--print-examples", type=int, default=12)
  parser.add_argument("--num-plot-samples", type=int, default=8)
  parser.add_argument("--plot-dir", default="waymo_entropy_boundaries_10hz")
  parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.history_len + args.horizon_len > args.num_steps:
    raise ValueError("--history-len + --horizon-len must be <= --num-steps")

  torch.manual_seed(args.seed)
  device = torch.device(args.device)
  tfrecord_paths = [Path(path) for path in args.tfrecord]
  if args.track_source == "sdc":
    x, labels, stats = extract_10hz_sdc_states(
        tfrecord_paths=tfrecord_paths,
        num_steps=args.num_steps,
        max_trajectories=args.max_trajectories,
    )
  else:
    x, labels, stats = extract_10hz_waymo_states(
        tfrecord_paths=tfrecord_paths,
        num_steps=args.num_steps,
        max_trajectories=args.max_trajectories,
        include_all_valid_tracks=args.include_all_valid_tracks,
        object_types=set(args.object_type),
    )
  train_x, train_labels, val_x, val_labels = split_dataset(x, labels, args.val_fraction, args.seed)
  print(f"parsed_stats={stats}")
  print(
      f"split: train={len(train_x)} val={len(val_x)} "
      f"state_dim={x.shape[-1]} hz={WAYMO_HZ} "
      f"target_mode={args.target_mode} track_source={args.track_source} "
      f"object_types={sorted(set(args.object_type))}"
  )

  train_history, train_target, train_traj_index, train_step_index = make_relative_windows(
      train_x,
      args.history_len,
      args.horizon_len,
      args.target_mode,
  )
  val_history, val_target, val_traj_index, val_step_index = make_relative_windows(
      val_x,
      args.history_len,
      args.horizon_len,
      args.target_mode,
  )
  history_mean, history_std = fit_normalizer(train_history)
  target_mean, target_std = fit_normalizer(train_target)
  train_history, train_target = normalize_windows(
      train_history,
      train_target,
      history_mean,
      history_std,
      target_mean,
      target_std,
  )
  val_history, val_target = normalize_windows(
      val_history,
      val_target,
      history_mean,
      history_std,
      target_mean,
      target_std,
  )
  print(
      f"relative_history_mean={history_mean.flatten().tolist()} "
      f"relative_history_std={history_std.flatten().tolist()} "
      f"target_mean={target_mean.flatten().tolist()} "
      f"target_std={target_std.flatten().tolist()}"
  )

  if args.target_mode == "full-trajectory":
    model = AutoregressiveTrajectoryGaussianGRU(
        state_dim=x.shape[-1],
        horizon_len=args.horizon_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    score_label = "1s trajectory NLL"
  else:
    model = EndpointGaussianGRU(
        state_dim=x.shape[-1],
        target_dim=train_target.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    score_label = "1s endpoint XY NLL" if args.target_mode == "final-xy" else "1s endpoint velocity NLL"
  train_model(
      model,
      train_history,
      train_target,
      train_traj_index,
      train_step_index,
      val_history,
      val_target,
      val_traj_index,
      val_step_index,
      args,
      device,
  )

  val_nll, val_entropy = score_trajectories(
      model,
      val_history,
      val_target,
      val_traj_index,
      val_step_index,
      len(val_x),
      val_x.shape[1],
      args.batch_size,
      device,
  )
  val_physical = val_x
  jerk = acceleration_change_score(val_physical)
  speed = speed_norm(val_physical)
  rows, boundaries = summarize_boundaries(val_nll, val_entropy, jerk, val_labels, args)

  output_dir = Path(args.plot_dir)
  csv_path = output_dir / "boundary_splits.csv"
  top_chunks_plot_path = output_dir / "boundary_diagnostics_top_chunks.png"
  random_plot_path = output_dir / "boundary_diagnostics_random.png"
  write_boundary_csv(rows, csv_path)
  top_indices = top_chunk_indices(boundaries, args.num_plot_samples)
  random_indices = random_sample_indices(len(val_physical), args.num_plot_samples, args.seed)
  plot_boundaries(
      val_physical,
      val_nll,
      val_entropy,
      speed,
      boundaries,
      top_chunks_plot_path,
      top_indices,
      "Validation trajectories with the most learned chunks",
      score_label,
  )
  plot_boundaries(
      val_physical,
      val_nll,
      val_entropy,
      speed,
      boundaries,
      random_plot_path,
      random_indices,
      "Random validation trajectories",
      score_label,
  )
  print(f"wrote_boundary_csv={csv_path}")
  print(f"top_chunk_plot_indices={top_indices}")
  print(f"random_plot_indices={random_indices}")
  print(f"wrote_top_chunks_plot={top_chunks_plot_path}")
  print(f"wrote_random_plot={random_plot_path}")


if __name__ == "__main__":
  main()
