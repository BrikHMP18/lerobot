# Training Benchmark Guide (ACT, Diffusion, SmolVLA, pi05)

This guide runs a fair offline benchmark for your dataset with:

- Same train/val split for all models (80/20 by episode index).
- Same number of training steps (`FAIR_STEPS`) for all models.
- Offline validation during training (`val/loss`) every `VAL_FREQ` steps.
- Automatic best-checkpoint tracking (`checkpoints/best` + `best_val.json`).

## 1. Prerequisites

```bash
conda activate lerobot
wandb login
```

Install model-specific extras once:

```bash
pip install -e ".[smolvla,pi]"
```

## 2. Run Training

Defaults use your dataset:

- `Autobrik/SO-ARM100-dump-pocket-cleaning2`
- `TRAIN_RATIO=0.8`
- `SPLIT_SEED=2026`
- `FAIR_STEPS=10000`
- `VAL_FREQ=1000`
- `PUSH_TO_HUB=false`

Run each model:

```bash
./run_train_act.sh
./run_train_dp.sh
./run_train_smolvla.sh
./run_train_pi05.sh
```

### Common env vars

```bash
export DATASET_REPO_ID="Autobrik/SO-ARM100-dump-pocket-cleaning2"
export DATASET_ROOT=""                    # optional local path
export FAIR_STEPS=10000
export VAL_FREQ=1000
export WANDB_PROJECT="dump-pocket-benchmark"
export WANDB_ENABLE=true
export WANDB_ENTITY=""                    # optional
export POLICY_DEVICE="cuda"
```

## 3. Outputs

Each run writes to:

- `outputs/train/benchmark/<timestamp>_benchmark_<model>/`

Important files:

- `checkpoints/last/pretrained_model`
- `checkpoints/best/pretrained_model`
- `best_val.json` (contains best step and best val loss)

Use `checkpoints/best/pretrained_model` for final evaluation.

## 4. W&B Panels for Paper

Use one project (`WANDB_PROJECT`) and compare the 4 runs.

Recommended panels:

1. `train/loss`
2. `val/loss`
3. `train/lr`
4. `train/grad_norm`
5. `train/epochs`
6. `val/best_val_loss`
7. `val/best_val_step`

Model-specific useful panels:

1. ACT: `train/l1_loss`, `train/kld_loss`
2. pi05/SmolVLA: `train/losses_after_forward`, `train/losses_after_rm_padding`
3. pi05: `train/loss_per_dim_*`, `val/loss_per_dim_*`

For paper figures:

- Keep same smoothing across panels.
- Use step as x-axis.
- Use same y-axis scale when comparing models for the same metric.

You can generate a reusable panel template file with:

```bash
python generate_wandb_panel_template.py \
  --project "${WANDB_PROJECT}" \
  --entity "${WANDB_ENTITY}" \
  --output wandb_panel_template.json
```

## 5. Notes on Fairness

This setup enforces fairness by:

1. Same split.
2. Same number of steps.
3. Same validation cadence.
4. Same checkpoint-selection rule (best `val/loss`).

If you later want a compute-time fairness study, switch to fixed wall-clock budget.
