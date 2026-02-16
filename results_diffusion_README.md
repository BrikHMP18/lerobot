# Diffusion Results (Paper Notes)

This file condenses the Diffusion Policy training run results for direct use in the paper draft.

## Run Metadata

- Project: `dump-pocket-benchmark`
- Run name: `benchmark_diffusion`
- Run path: `brik-meza-pucp/dump-pocket-benchmark/o25e73gn`
- Author: `brik-meza`
- Start time: `2026-02-15 15:41:22` (local W&B display)
- Runtime: `3h 17m 5s`
- Tracked hours: `2h 50m 36s`
- Host GPU: `NVIDIA GeForce RTX 4070`
- Seed: `1000`
- Run state: `Finished` (state recovered after terminal/W&B close issue)

## Dataset + Split

- Dataset repo: `Autobrik/SO-ARM100-dump-pocket-cleaning2`
- Split policy: fixed episode split (train/val)
- Train episodes: `129`
- Val episodes: `33`
- Total train steps planned: `7252` (4 epochs setup with batch size 32)

## Diffusion Training Configuration (effective)

- Policy: `diffusion`
- Batch size: `32`
- Num workers: `4`
- Steps: `7252`
- Log frequency: `100`
- Checkpoint frequency: `1813`
- Offline val frequency: `907`
- Best-checkpoint tracking: enabled (`val/loss`)
- Eval env: disabled (`eval_freq=-1`)
- Scheduler: `cosine`, warmup steps `500`

## Key Metrics Snapshot (final from run dashboard)

- `train/steps`: `7252`
- `train/samples`: `230400`
- `train/episodes`: `504.5855`
- `train/epochs`: `3.9115`
- `train/loss`: `0.0132536`
- `train/grad_norm`: `0.2765647`
- `train/lr`: `0.0000000602522`
- `train/update_s`: `0.1555586`
- `train/dataloading_s`: `0.7059742`

- `val/loss`: `0.0140658`
- `val/best_val_loss`: `0.0140658`
- `val/best_val_step`: `7252`

## Produced Checkpoint Artifacts

- Local checkpoints observed:
- `000907`
- `001813`
- `001814`
- `002721`
- `003626`
- `004535`
- `005439`
- `006349`
- `007252`
- `best -> 007252`
- `last -> 007252`

- W&B artifacts shown:
- `policy_diffusion-seed_1000-dataset_Autobrik_SO-ARM100-dump-pocket-cleaning2-000907:v1`

Note: W&B artifact list is incomplete because the terminal/W&B session ended uncleanly near run completion. Local training artifacts and `best_val.json` confirm the run fully completed.

## Scientific Interpretation (Diffusion)

1. Optimization is stable:
- Training loss decreases smoothly toward low values.
- Gradient norms remain controlled without instability spikes.

2. Validation trend indicates late improvement:
- `val/loss` descends overall and reaches best value at final step (`7252`).
- Mid-run bump is visible but followed by clear recovery and improved final validation.

3. Generalization behavior is strong in this run:
- Final train and val losses are close (`0.0133` vs `0.0141`), suggesting a small train-val gap.
- No signs of overfitting in the final stage.

## Paper-Ready Diffusion Conclusion

Diffusion Policy converged successfully in the 4-epoch setup and reached its best validation score at the final step (`7252`). The correct model candidate for robot benchmarking is `checkpoints/best/pretrained_model` (equivalent to step `007252`).

## To Finalize for Paper Table

1. Ensure HF upload includes `checkpoints/best`, `checkpoints/last`, and `best_val.json`.
2. Add GPU-hours and instance cost to compare compute with ACT/pi05/SmolVLA.
3. Keep model-selection rule consistent across methods: `best` by `val/loss`.
4. Use real-robot evaluation (`N≈30`) as the primary cross-model comparison.
