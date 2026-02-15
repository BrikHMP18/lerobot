# Training Benchmark Guide (ACT, Diffusion, SmolVLA, pi05)

This guide is the end-to-end flow to run training on Vast.ai using your fork/branch:

- Fork: `BrikHMP18/lerobot`
- Branch: `wmc-paper`
- 4 models: ACT, Diffusion Policy, SmolVLA, pi05
- 1 seed for now
- model selection by offline validation (`val/loss`)
- final comparison in real robot tests (you plan ~30 runs/model)

## 0. Important note about "4 models in parallel on 1 GPU"

With a single GPU instance, running the 4 models truly in parallel is not recommended (high OOM risk and unstable timing).

Recommended on `1x GPU`:

1. Run models sequentially on the same instance.
2. Keep same train/val split and same epoch budget for fairness.

If you want true parallelism, use `4 GPUs` (one model per GPU) or 4 separate instances.

## 1. Vast instance setup

Pick an instance with:

- CUDA-compatible GPU (your selected 96GB VRAM option is enough).
- Enough disk for checkpoints + dataset cache (>= 200GB recommended).
- Stable upload bandwidth for W&B/Hugging Face.

Then SSH into the instance.

## 2. Clone your repo and checkout branch

Use HTTPS:

```bash
git clone https://github.com/BrikHMP18/lerobot.git
cd lerobot
git fetch origin
git checkout wmc-paper
git pull origin wmc-paper
```

Or SSH (if keys are configured on that VM):

```bash
git clone git@github.com:BrikHMP18/lerobot.git
cd lerobot
git checkout wmc-paper
git pull origin wmc-paper
```

Optional: keep upstream configured.

```bash
git remote add upstream https://github.com/huggingface/lerobot.git || true
git remote -v
```

## 3. Create environment and install dependencies

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install --upgrade pip
pip install -e ".[smolvla,pi]"
```

Quick sanity check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Authenticate services

For dataset access/push:

```bash
huggingface-cli login
```

For training dashboards:

```bash
wandb login
```

## 5. Set benchmark env vars

Use one shared project for all 4 models:

```bash
export DATASET_REPO_ID="Autobrik/SO-ARM100-dump-pocket-cleaning2"
export DATASET_ROOT=""                  # optional local dataset path
export TRAIN_RATIO=0.8
export SPLIT_SEED=2026
export TARGET_EPOCHS=2
export OVERRIDE_STEPS=""                # optional fixed steps override
export VAL_FREQ=1000
export SAVE_FREQ=1000
export LOG_FREQ=100
export POLICY_DEVICE="cuda"
export WANDB_ENABLE=true
export WANDB_PROJECT="dump-pocket-benchmark"
export WANDB_ENTITY=""                  # optional
export PUSH_TO_HUB=false
export SEED=1000
```

## 6. Run the 4 trainings (1 GPU: sequential)

```bash
./run_train_act.sh
./run_train_dp.sh
./run_train_smolvla.sh
./run_train_pi05.sh
```

Each script:

1. Computes train/val episode split (same ratio + seed).
2. Computes steps from `TARGET_EPOCHS` (or uses `OVERRIDE_STEPS`).
3. Runs offline validation periodically.
4. Saves and tracks best checkpoint by minimum `val/loss`.

## 7. Outputs and best checkpoint

Each run is saved under:

- `outputs/train/benchmark/<timestamp>_benchmark_<model>/`

Key artifacts:

- `checkpoints/last/pretrained_model`
- `checkpoints/best/pretrained_model`
- `best_val.json`

`best_val.json` contains the selected best step and best `val/loss`.

## 8. W&B monitoring for paper figures

Recommended panels:

1. `train/loss`
2. `val/loss`
3. `train/lr`
4. `train/grad_norm`
5. `train/epochs`
6. `val/best_val_loss`
7. `val/best_val_step`

Useful model-specific panels:

1. ACT: `train/l1_loss`, `train/kld_loss`
2. SmolVLA/pi05: `train/losses_after_forward`, `train/losses_after_rm_padding`
3. pi05: `train/loss_per_dim_*`, `val/loss_per_dim_*`

Generate a reusable W&B panel template:

```bash
python generate_wandb_panel_template.py \
  --project "${WANDB_PROJECT}" \
  --entity "${WANDB_ENTITY}" \
  --output wandb_panel_template.json
```

Optional GPU logging for compute report:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv -l 30 | tee gpu_usage_log.csv
```

## 9. How to compare models correctly

Use `val/loss` only for model selection inside each architecture:

1. pick ACT best checkpoint by ACT `val/loss` minimum
2. pick DP best checkpoint by DP `val/loss` minimum
3. pick SmolVLA best checkpoint by SmolVLA `val/loss` minimum
4. pick pi05 best checkpoint by pi05 `val/loss` minimum

Do not claim cross-architecture superiority only from absolute `val/loss`.
Cross-model ranking should be based on your real-world evaluation metric (e.g., g/min removed, success rate), with your planned ~30 runs/model.

## 10. Fairness assumptions in this setup

Current setup fairness:

1. Same train/val split process (ratio + split seed).
2. Same nominal epoch budget (`TARGET_EPOCHS=2` by default).
3. Same validation cadence (`VAL_FREQ`).
4. Same checkpoint rule (min `val/loss` per model).
5. Same single-seed policy for now (`SEED=1000`).

When you are ready for stronger statistical claims, add multi-seed runs.
