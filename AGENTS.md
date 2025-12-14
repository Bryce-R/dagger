# Repository Guidelines

## Project Structure & Module Organization
`toy_dataset.py` defines the synthetic control environment, dataset slicing helpers, and constants like `CONTEXT_LEN`; it now simulates a delayed actuator controlled by `CONTROL_DELAY` (default 2) so that actions affect the state multiple steps later. `model.py` holds the transformer policy components (`Block`, `TinyTransformerPolicy`). `train.py` wires the dataset, dataloaders, the CLI argument parser, and training loop; the CLI flags (`--num-traj`, `--epochs`, `--control-delay`, `--checkpoint-path`, etc.) enable faster smoke runs or checkpoint exports. `visualize_rollout.py` loads a saved checkpoint (required `--checkpoint-path`), computes expert vs. learned stats under the specified delay, and saves comparison plots. `requirements.txt` pins the Python dependencies.

## Build, Test, and Development Commands
Use Python 3.10+ with PyTorch + Matplotlib installed (`python -m venv .venv && source .venv/bin/activate`, then `pip install -r requirements.txt`). Key commands:
- `python train.py --num-traj 50 --epochs 2 --control-delay 2 --checkpoint-path checkpoints/smoke.pt`: generate a small dataset (respecting the delayed actuator), train quickly, and persist a portable checkpoint to `checkpoints/` (directory auto-created).
- `python train.py`: default, full run with dataset size from `toy_dataset.NUM_TRAJ`.
- `python visualize_rollout.py --checkpoint-path checkpoints/smoke.pt --horizon 100 --output plots/rollout.png`: load a saved model, compare trajectories over a chosen horizon, and save a two-panel state/action plot; pass `--control-delay` to override or rely on the value stored in the checkpoint.
- `python visualize_dataset.py --num-traj 100 --num-samples 6 --control-delay 2 --output plots/dataset.png`: regenerate the expert dataset, plot sampled context windows plus the action histogram, and write the figure to `plots/`.
Run commands from the repo root so relative imports such as `from toy_dataset import ...` resolve cleanly.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, snake_case filenames (as already used), and descriptive constant names in ALL_CAPS for hyperparameters. Keep modules importable without side effects beyond lightweight logging. Prefer small, pure helpers over long functions, and document tensor shapes inline (see `train.py` comments) when not obvious.

## Testing Guidelines
No formal test suite exists yet, but add `pytest`-style unit tests under `tests/` for any new dataset transforms or model components. Use deterministic seeds via `numpy.random.default_rng` and `torch.manual_seed` so regression tests remain stable. When touching training, validate with a short CLI run (e.g., `python train.py --num-traj 10 --epochs 1`) and attach the loss table plus checkpoint location in the PR description. Test names should describe intent (`test_dataset_shapes_match_context_len`).

## Commit & Pull Request Guidelines
Existing history uses short, imperative commits (e.g., `add initial model.py toy_dataset.py train.py visualize_rollout.py`). Follow that style, keep subject lines under ~60 chars, and put detailed rationale in the body if needed. Every PR should describe the motivation, key changes, and verification steps (commands + outputs). Link related issues, attach logs or plots from `visualize_rollout.py` when behavior changes, and request review before merging.
