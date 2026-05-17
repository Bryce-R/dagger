---
name: waymo-vqvae
description: Use for Waymo trajectory compression experiments in /Users/pengkai/Code/dagger, especially train_waymo_mlp_fsq.py MLP/FSQ/MPS runs, validation metrics, reconstruction plots, and experiment comparisons.
---

# Waymo MLP-FSQ Experiments

Work from `/Users/pengkai/Code/dagger`. Use `train_waymo_mlp_fsq.py` for new
Waymo MLP-FSQ experiments. The older `train_waymo_trajectory_vqvae.py` still
exists for historical VQ-VAE/conv comparisons, but should not be used for new
MLP-FSQ runs.

## Baseline Command Pattern

Use MPS outside the sandbox:

```bash
env PYTHONUNBUFFERED=1 MPLCONFIGDIR=/private/tmp/mplconfig XDG_CACHE_HOME=/private/tmp python train_waymo_mlp_fsq.py \
  --tfrecord /Users/pengkai/Code/waymo-open-dataset/uncompressed_scenario_training_training.tfrecord-00000-of-01000 \
  --hidden-dim 512 \
  --code-dim 64 \
  --mlp-latent-tokens 1 \
  --fsq-levels 256 \
  --fsq-input-scale 0.1 \
  --position-loss-weight 0 \
  --plot-every 1000 \
  --log-every 1000 \
  --latent-stats-every 1000 \
  --device mps \
  --epochs 12000 \
  --batch-size 512 \
  --include-all-states \
  --val-fraction 0.2
```

`--include-all-states` currently means XY delta + yaw delta + z delta + local velocity XY. It intentionally excludes length/width/height.

`train_waymo_mlp_fsq.py` intentionally has no `--architecture`, `--quantizer`,
`--num-codes`, or VQ loss flags.

## Current Findings

- The split is seeded random shuffle: validation is the first `val_fraction` slice of `torch.randperm(..., seed=args.seed)`.
- Validation metrics exposed heavy overfitting. Full-data metrics from older runs were optimistic.
- `--position-loss-weight 0` produced better same-data max-error tails than `10`, but validation still overfits.
- Dropout `0.2` and `0.05` hurt final validation versus no dropout. Dropout `0.05` had the best early val `pos_loss` around epoch 1000.
- More latent tokens helped same-data tails at 2 tokens, but 4 tokens regressed some horizons.
- Absolute-position decoding did not beat delta decoding.

## Metrics

- `recon`: Smooth L1 on normalized model outputs across all channels.
- `pos_loss`: Smooth L1 on physical local XY positions after integrating deltas unless `--decode-absolute-positions`.
- `mse`: MSE on normalized model outputs.
- `Final max abs XY position error by second`: worst physical XY coordinate error at 1s, 2s, etc.; use this for tail behavior.

## Plot Outputs

The script saves separate train and validation reconstruction plots:

- `reconstruction_train_epoch_XXXX.png`
- `reconstruction_val_epoch_XXXX.png`

Default `--num-plot-samples` is 5.

## Next Best Regularization Direction

Prefer best-validation checkpointing / early stopping over more dropout. The validation curve usually worsens after early epochs even as train metrics improve.
