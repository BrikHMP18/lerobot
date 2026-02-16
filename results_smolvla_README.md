# SmolVLA Results (Paper Notes)

This file condenses the SmolVLA training run results for direct use in the paper draft.

## Run Metadata

- Project: `dump-pocket-benchmark`
- Run name: `benchmark_smolvla`
- Run path: `brik-meza-pucp/dump-pocket-benchmark/xz155lcl`
- Author: `brik-meza`
- Start time: `2026-02-15 12:31:15` (local W&B display)
- Runtime: `1h 17m 39s`
- Tracked hours: `1h 15m`
- Host GPU: `NVIDIA GeForce RTX 4090`
- Seed: `1000`
- Run state: `Finished`

## Dataset + Split

- Dataset repo: `Autobrik/SO-ARM100-dump-pocket-cleaning2`
- Split policy: fixed episode split (train/val)
- Train episodes: `129`
- Val episodes: `33`
- Total train steps planned: `14728` (4 epochs setup)

## SmolVLA Training Configuration (effective)

- Policy: `smolvla` (pretrained base: `lerobot/smolvla_base`)
- Batch size: `16`
- Num workers: `4`
- Steps: `14728`
- Log frequency: `100`
- Checkpoint frequency: `3682`
- Offline val frequency: `1841`
- Best-checkpoint tracking: enabled (`val/loss`)
- Eval env: disabled (`eval_freq=-1`)
- Camera compatibility settings:
- `--rename_map={"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}`
- `--policy.empty_cameras=1`

## Key Metrics Snapshot (final from run dashboard)

- `train/steps`: `14700`
- `train/samples`: `235200`
- `train/episodes`: `515.0977`
- `train/epochs`: `3.9930`
- `train/loss`: `0.0125621`
- `train/losses_after_forward`: `0.0125621`
- `train/losses_after_rm_padding`: `0.0125621`
- `train/grad_norm`: `0.2248338`
- `train/lr`: `0.0000025076`
- `train/update_s`: `0.2188487`
- `train/dataloading_s`: `0.0073851`

- `val/loss`: `0.0253878`
- `val/losses_after_forward`: `0.0253878`
- `val/losses_after_rm_padding`: `0.0253878`
- `val/best_val_loss`: `0.0253878`
- `val/best_val_step`: `14728`

## Produced Checkpoint Artifacts (observed)

- `...-001841:v0`
- `...-003682:v0`
- `...-005523:v0`
- `...-007364:v0`
- `...-009205:v0`
- `...-011046:v0`
- `...-012887:v0`
- `...-014728:v0`

## Scientific Interpretation (SmolVLA)

1. Optimization is stable and smooth:
- `train/loss` reaches a low value (`~1.26e-2`) with no instability signs.
- `train/grad_norm` remains low by the end of training.

2. Validation keeps improving up to final step:
- `val/loss` decreases consistently from early validation to final validation.
- Best checkpoint is at the final step (`14728`), indicating no late degradation.

3. Generalization gap is present but controlled:
- Final `val/loss` is about 2x `train/loss`, expected in this offline setup.
- No abrupt divergence between train and validation trends.

4. Camera compatibility workaround should be documented:
- SmolVLA base expects camera keys `camera1/2/3`.
- Dataset has two cameras (`top`, `wrist`), mapped to `camera1/2` plus one empty camera.

## Paper-Ready SmolVLA Conclusion

SmolVLA converged cleanly in 4-epoch training and achieved best offline validation at the final step (`14728`). For robot benchmarking, use `checkpoints/best/pretrained_model` from this run. Mention explicitly in Methods that SmolVLA was trained with camera-key remapping and one empty camera for input compatibility.

## To Finalize for Paper Table

1. Add direct comparison row versus ACT/pi05/DP in a unified table.
2. Add GPU-hours and instance cost for compute accounting.
3. Keep checkpoint selection rule consistent: always `best` by `val/loss`.
4. Report real-robot metrics (`N≈30`) as the primary cross-model benchmark.
