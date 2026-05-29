"""Plot random 10Hz Waymo trajectories with XY, yaw, and velocity diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train_waymo_entropy_boundaries import (
    TRACK_TYPE_NAMES,
    WAYMO_HZ,
    ensure_waymo_scenario_pb2,
    extract_10hz_waymo_states,
    iter_tfrecord_records,
    rotate_to_local,
    wrap_angle,
)


def extract_sdc_trajectories(
    tfrecord_paths: list[Path],
    num_steps: int,
    max_trajectories: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
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
        return torch.stack(trajectories), torch.tensor(labels, dtype=torch.long), stats

  if not trajectories:
    raise ValueError("No valid SDC trajectories were extracted.")
  return torch.stack(trajectories), torch.tensor(labels, dtype=torch.long), stats


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Plot random Waymo 10Hz trajectory snippets.")
  parser.add_argument("--tfrecord", action="append", required=True)
  parser.add_argument("--num-samples", type=int, default=10)
  parser.add_argument("--seconds", type=float, default=2.0)
  parser.add_argument("--max-trajectories", type=int, default=4096)
  parser.add_argument("--include-all-valid-tracks", action="store_true")
  parser.add_argument("--object-type", action="append", type=int, default=[1])
  parser.add_argument(
      "--track-source",
      choices=("tracks-to-predict", "sdc"),
      default="tracks-to-predict",
      help="Plot Waymo target tracks or the SDC/ego track.",
  )
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--output", default="waymo_random_trajectories_2s.png")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  num_steps = max(2, int(round(args.seconds * WAYMO_HZ)))
  tfrecord_paths = [Path(path) for path in args.tfrecord]
  if args.track_source == "sdc":
    x, labels, stats = extract_sdc_trajectories(
        tfrecord_paths=tfrecord_paths,
        num_steps=num_steps,
        max_trajectories=args.max_trajectories,
    )
  else:
    x, labels, stats = extract_10hz_waymo_states(
        tfrecord_paths=tfrecord_paths,
        num_steps=num_steps,
        max_trajectories=args.max_trajectories,
        include_all_valid_tracks=args.include_all_valid_tracks,
        object_types=set(args.object_type),
    )
  generator = torch.Generator().manual_seed(args.seed)
  sample_count = min(args.num_samples, len(x))
  sample_indices = torch.randperm(len(x), generator=generator)[:sample_count].tolist()
  time = torch.arange(num_steps, dtype=torch.float32) / WAYMO_HZ

  fig, axes = plt.subplots(sample_count, 3, figsize=(15.0, 2.6 * sample_count), squeeze=False)
  for row, index in enumerate(sample_indices):
    traj = x[index]
    xy = traj[:, :2]
    yaw = traj[:, 2]
    velocity = traj[:, 3:5]

    xy_axis = axes[row, 0]
    xy_axis.plot(xy[:, 0], xy[:, 1], "o-", markersize=2.5, color="tab:blue")
    xy_axis.scatter(xy[0, 0], xy[0, 1], s=45, color="tab:green", label="start")
    xy_axis.scatter(xy[-1, 0], xy[-1, 1], s=45, color="tab:red", label="end")
    object_name = TRACK_TYPE_NAMES.get(int(labels[index]), "unknown")
    xy_axis.set_title(f"{args.track_source} traj={index} {object_name} XY first {args.seconds:g}s")
    xy_axis.set_aspect("equal", adjustable="datalim")
    xy_axis.grid(True, alpha=0.3)

    yaw_axis = axes[row, 1]
    yaw_axis.plot(time, yaw, color="tab:purple")
    yaw_axis.set_title("yaw")
    yaw_axis.set_xlabel("time (s)")
    yaw_axis.set_ylabel("rad")
    yaw_axis.grid(True, alpha=0.3)

    speed = velocity.norm(dim=-1)
    speed_axis = axes[row, 2]
    speed_axis.plot(time, speed, color="black", linewidth=2.4, label="speed norm")
    speed_axis.set_title("velocity norm")
    speed_axis.set_xlabel("time (s)")
    speed_axis.set_ylabel("m/s")
    speed_axis.legend(loc="best", fontsize=8)
    speed_axis.grid(True, alpha=0.3)

  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=2)
  fig.suptitle(
      f"Random Waymo {args.track_source} trajectories: "
      f"{sample_count} samples, {num_steps} steps at {WAYMO_HZ}Hz\n"
      f"stats={stats}",
      y=0.995,
  )
  fig.tight_layout(rect=(0, 0.025, 1, 0.98))

  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output_path, dpi=170)
  plt.close(fig)
  print(f"sample_indices={sample_indices}")
  print(f"wrote_plot={output_path}")


if __name__ == "__main__":
  main()
