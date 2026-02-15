# pi05 Results (Paper Notes)

This file condenses the pi05 training run results for direct use in the paper draft.

## Run Metadata

- Project: `dump-pocket-benchmark`
- Run name: `benchmark_pi05`
- Run path: `brik-meza-pucp/dump-pocket-benchmark/s1ilpudx`
- Author: `brik-meza`
- Start time: `2026-02-15 02:55:08` (local W&B display)
- Host GPU: `NVIDIA GeForce RTX 4090`
- Seed: `1000`
- Run state at capture: `Running` (intermediate snapshot)

## Dataset + Split

- Dataset repo: `Autobrik/SO-ARM100-dump-pocket-cleaning2`
- Split policy: fixed episode split (train/val)
- Train episodes: `129`
- Val episodes: `33`
- Total train steps planned: `29452` (4 epochs setup)

## pi05 Training Configuration (effective)

- Policy: `pi05`
- Pretrained base: `lerobot/pi05_base`
- Batch size: `8`
- Num workers: `4`
- Steps: `29452`
- Log frequency: `100`
- Checkpoint frequency: `7363`
- Offline val frequency: `3682`
- Best-checkpoint tracking: enabled (`val/loss`)
- Eval env: disabled (`eval_freq=-1`)
- Precision: `bfloat16`
- Gradient checkpointing: enabled
- Vision encoder frozen: enabled
- Train expert only: enabled

## Key Metrics Snapshot (from run dashboard)

This snapshot corresponds to the early stage of training (`train/steps=100`), so these values are not final.

- `train/steps`: `100`
- `train/samples`: `800`
- `train/epochs`: `0.0135817`
- `train/loss`: `0.1779469`
- `train/loss_per_dim_0`: `0.1417757`
- `train/loss_per_dim_1`: `0.1361435`
- `train/loss_per_dim_2`: `0.2049324`
- `train/loss_per_dim_3`: `0.0951984`
- `train/loss_per_dim_4`: `0.1076562`
- `train/loss_per_dim_5`: `0.3819750`
- `train/grad_norm`: `2.0257515`
- `train/lr`: `0.0000013111`
- `train/update_s`: `0.9522509`
- `train/dataloading_s`: `0.0172425`

## Produced Checkpoint Artifacts (observed)

- In progress. Checkpoints are expected at steps:
- `007363`
- `014726`
- `022089`
- `029452` (final step)

## Scientific Interpretation (pi05, interim)

1. Optimization starts stable:
- No exploding gradients are visible in the early snapshot (`grad_norm` low and stable).
- Step time (`update_s`) is consistent with a larger VLA policy and small batch size.

2. This snapshot is too early for model-selection conclusions:
- Only ~0.014 epochs have been processed.
- `val/loss` trend and `best_val_step` should be read after multiple validation cycles.

3. Expected behavior for this setup:
- Lower throughput than ACT due to heavier backbone and `batch_size=8`.
- Better semantic priors may appear later in training, not in first 100 steps.

## Paper-Ready pi05 Conclusion (current status)

The run is configured correctly and currently training as expected. No scientific ranking versus ACT/DP/SmolVLA should be made yet from this intermediate snapshot. The comparison checkpoint for robot evaluation must be selected from `checkpoints/best/pretrained_model` once training ends.

## To Finalize After Run Ends

1. Replace interim metrics with final run summary (including final `train/epochs` and final `train/steps`).
2. Record `val/best_val_loss` and `val/best_val_step` from the completed run.
3. List actual generated checkpoint artifacts from W&B/HF.
4. Record total wall-clock runtime and compute GPU-hours.
5. Use `checkpoints/best/pretrained_model` for real-robot evaluation (`N≈30` planned).
