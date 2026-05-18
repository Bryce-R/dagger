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
- To reduce the validation gap, use `--include-all-valid-tracks`, smaller capacity
  (`--hidden-dim 128 --code-dim 16 --fsq-levels 64`), light denoising
  (`--input-noise-std 0.02`), and the best validation checkpoint.
- Naive direct absolute-XY binning is a strong reconstruction baseline but uses
  far more bits than compact FSQ. Correct tokenized comparison is one timestep
  token with joint XY vocab: 50x50 XY bins means 50 tokens and vocab size 2500.
  Matched FSQ uses `--mlp-latent-tokens 50 --code-dim 2 --fsq-levels 50`,
  also 50 tokens with joint vocab 2500. On the 4,096 all-valid split, matched
  FSQ reached val `pos_loss=0.037426` vs 50x50 bins at `0.110475`, but bins
  had a much tighter max-error tail (`~1.47m` vs FSQ peaking above `14m`).
- Tail-focused losses improve sparse-case coverage. Matched FSQ plus
  `--tail-position-loss-weight 0.5 --tail-position-fraction 0.1`
  `--final-position-loss-weight 0.5` reached val `pos_loss=0.014194` and
  reduced 5s val max error to `5.52m`. A stronger `1.0/1.0` setting improved
  the 5s tail slightly to `5.26m` but worsened average val loss to `0.016001`.
- P99 validation absolute coordinate error is much better than max for tail-loss
  FSQ. Tail `0.5`/final `0.5` p99 by second is
  `0.536941 / 0.530983 / 0.685823 / 0.746115 / 0.828350`, better than 50x50
  bins p99 `1.397281 / 1.332420 / 1.363819 / 1.370701 / 1.408838`. Remaining
  FSQ issue is a few extreme max-error outliers.
- No-tail matched FSQ p99 is
  `0.735312 / 1.010828 / 1.295937 / 1.548877 / 1.620797`: better than bins at
  1s-3s, worse at 4s-5s. Tail loss is needed for long-horizon p99.
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
