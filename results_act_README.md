# ACT Results (Paper Notes)

This file condenses the ACT training run results for direct use in the paper draft.

## Run Metadata

- Project: `dump-pocket-benchmark`
- Run name: `benchmark_act`
- Run path: `brik-meza-pucp/dump-pocket-benchmark/724dohyt`
- Author: `brik-meza`
- Start time: `2026-02-15 02:14:30` (local W&B display)
- Host GPU: `NVIDIA RTX PRO 4000 Blackwell`
- Seed: `1000`

## Dataset + Split

- Dataset repo: `Autobrik/SO-ARM100-dump-pocket-cleaning2`
- Split policy: fixed episode split (train/val)
- Train episodes: `129`
- Val episodes: `33`
- Total train steps planned: `14728` (4 epochs setup)

## ACT Training Configuration (effective)

- Policy: `act`
- Batch size: `16`
- Num workers: `4`
- Steps: `14728`
- Log frequency: `100`
- Checkpoint frequency: `3682`
- Offline val frequency: `1841`
- Best-checkpoint tracking: enabled (`val/loss`)
- Eval env: disabled (`eval_freq=-1`)

## Key Metrics Snapshot (from run dashboard)

- `train/steps`: `10900`
- `train/epochs`: `2.9608`
- `train/loss`: `0.1608446`
- `train/l1_loss`: `0.1407329`
- `train/kld_loss`: `0.0022179`
- `train/grad_norm`: `12.7969`
- `train/update_s`: `0.22764`
- `train/dataloading_s`: `0.00547`

- `val/loss`: `0.2745911`
- `val/l1_loss`: `0.2687818`
- `val/kld_loss`: `0.0005809`
- `val/best_val_loss`: `0.2745911`
- `val/best_val_step`: `9205`

## Produced Checkpoint Artifacts (observed)

- `...-001841:v0`
- `...-003682:v0`
- `...-005523:v0`
- `...-007364:v0`
- `...-009205:v0`
- `...-011046:v0`

## Scientific Interpretation (ACT)

1. Optimization appears stable:
- `train/loss` and `train/l1_loss` decrease smoothly.
- `grad_norm` decreases without instability spikes.

2. Validation behavior indicates convergence:
- `val/loss` drops strongly early (roughly from ~0.38 to ~0.28) and then plateaus.
- Best validation is reached at `step 9205`.

3. VAE/KL term behavior is expected:
- `train/kld_loss` decays toward very small values.
- `val/kld_loss` also remains low and stable.

4. Generalization gap is present but controlled:
- Train loss remains below validation loss (expected for this setup).
- No evidence of catastrophic overfitting in the observed window.

## Paper-Ready ACT Conclusion

ACT provides a stable and strong baseline on this dataset. The model reaches its best validation performance around step `9205`, after which additional optimization yields marginal gains. For model comparison and robot evaluation, the correct candidate is the `best` checkpoint selected by `val/loss` (not `last`).

## To Finalize After Run Ends

1. Confirm final `best_val.json` values in output directory.
2. Record total wall-clock runtime and compute GPU-hours.
3. Use `checkpoints/best/pretrained_model` for real-robot evaluation (`N≈30` planned).
