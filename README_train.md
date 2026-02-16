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
pip install -e .
```

Model-specific installs (important):

1. ACT / Diffusion Policy instance:
```bash
pip install -e .
```
2. SmolVLA instance:
```bash
pip install -e ".[smolvla]"
```
3. pi05 instance:
```bash
pip install -e ".[pi]"
```

Do not install `.[smolvla,pi]` together in one environment because of a known `transformers` version conflict.

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
export TARGET_EPOCHS=4
export OVERRIDE_STEPS=""                # optional fixed steps override
export USE_REFERENCE_STEPS=false        # true => per-model reference steps
export VAL_FREQ=""                      # optional; empty => auto from VALS_PER_EPOCH
export SAVE_FREQ=""                     # optional; empty => auto from CHECKPOINTS_PER_EPOCH
export CHECKPOINTS_PER_EPOCH=1
export VALS_PER_EPOCH=2
export LOG_FREQ=100
export POLICY_DEVICE="cuda"
export WANDB_ENABLE=true
export WANDB_PROJECT="dump-pocket-benchmark"
export WANDB_ENTITY=""                  # optional
export PUSH_TO_HUB=false
export SEED=1000
```

Default batch sizes in scripts:

1. ACT: `BATCH_SIZE=16`
2. Diffusion: `BATCH_SIZE=32`
3. SmolVLA: `BATCH_SIZE=16`
4. pi05: `BATCH_SIZE=8`

## 6. Run the 4 trainings (1 GPU: sequential)

```bash
./run_train_act.sh
./run_train_dp.sh
./run_train_smolvla.sh
./run_train_pi05.sh
```

For 4 separate Vast instances, recommended mapping:

1. Instance A (base install): `./run_train_act.sh`
2. Instance B (base install): `./run_train_dp.sh`
3. Instance C (`.[smolvla]`): `./run_train_smolvla.sh`
4. Instance D (`.[pi]`): `./run_train_pi05.sh`

Each script:

1. Computes train/val episode split (same ratio + seed).
2. Computes steps from `TARGET_EPOCHS` (or uses `OVERRIDE_STEPS`).
3. Runs offline validation periodically.
4. Saves and tracks best checkpoint by minimum `val/loss`.

Step selection priority in scripts:

1. `OVERRIDE_STEPS` (if set)
2. `USE_REFERENCE_STEPS=true` (`REFERENCE_STEPS` per model)
3. computed from `TARGET_EPOCHS`

Frequency selection priority in scripts:

1. `SAVE_FREQ` / `VAL_FREQ` (if explicitly set)
2. auto from `CHECKPOINTS_PER_EPOCH` / `VALS_PER_EPOCH` using `steps_per_epoch`
3. fallback `1000` when `steps_per_epoch` is unknown (e.g. some custom step modes)

Default `REFERENCE_STEPS` in scripts:

1. ACT: `100000`
2. Diffusion: `200000`
3. SmolVLA: `20000`
4. pi05: `100000`

## 7. Outputs and best checkpoint

Each run is saved under:

- `outputs/train/benchmark/<timestamp>_benchmark_<model>/`

Key artifacts:

- `checkpoints/last/pretrained_model`
- `checkpoints/best/pretrained_model`
- `best_val.json`

`best_val.json` contains the selected best step and best `val/loss`.

Best checkpoint update cadence:

1. Offline validation runs every `VAL_FREQ` steps.
2. It also runs at the final step (`step == cfg.steps`), even if not divisible by `VAL_FREQ`.
3. If `val/loss` improves, `checkpoints/best` is updated.
4. Periodic checkpoints are still kept every `SAVE_FREQ` steps and at final step.

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
2. Same nominal epoch budget (`TARGET_EPOCHS=4` by default).
3. Same validation cadence (same `VALS_PER_EPOCH` or same explicit `VAL_FREQ`).
4. Same checkpoint rule (min `val/loss` per model).
5. Same single-seed policy for now (`SEED=1000`).

When you are ready for stronger statistical claims, add multi-seed runs.

## 11. Real-Robot Inference Rollouts (Policy-Only)

Use these scripts to run autonomous robot rollouts with recording (`lerobot-record --policy.path=...`) for each model:

```bash
./run_inference_act.sh
./run_inference_dp.sh
./run_inference_pi05.sh
./run_inference_smolvla.sh
```

Each script:

1. Resolves `POLICY_PATH` in this order:
- explicit `POLICY_PATH`
- `RUN_DIR/checkpoints/best/pretrained_model`
- latest `outputs/train/benchmark/*_benchmark_<model>/checkpoints/best/pretrained_model`
2. Validates `config.json` and `model.safetensors` exist in resolved `POLICY_PATH`.
3. Uses policy-only control (no teleop), dual cameras (`top`, `wrist`), and records rollouts.
4. Uses manual episode control by default.

Manual controls:

1. `Right Arrow`: save current episode and start next
2. `Left Arrow`: discard/re-record current episode
3. `Esc`: save current partial episode and exit session

Default environments:

1. `run_inference_act.sh` -> `CONDA_ENV=lerobot`
2. `run_inference_dp.sh` -> `CONDA_ENV=lerobot`
3. `run_inference_pi05.sh` -> `CONDA_ENV=lerobot-pi`
4. `run_inference_smolvla.sh` -> `CONDA_ENV=lerobot-smolvla`

Default eval dataset repos (separated by model):

1. `Autobrik/eval_SO-ARM100-dump-pocket-cleaning2-act`
2. `Autobrik/eval_SO-ARM100-dump-pocket-cleaning2-diffusion`
3. `Autobrik/eval_SO-ARM100-dump-pocket-cleaning2-pi05`
4. `Autobrik/eval_SO-ARM100-dump-pocket-cleaning2-smolvla`

Defaults chosen for field robustness:

1. `PUSH_TO_HUB=false` (local-first, upload later)
2. `MANUAL_EPISODE_CONTROL=true`
3. `DISPLAY_DATA=true`
4. `RESUME=auto`

`RESUME=auto` behavior:

1. If dataset already exists locally (`.../meta/info.json`), scripts set `--resume=true`.
2. Otherwise, scripts set `--resume=false`.

Useful overrides:

```bash
# Force a specific checkpoint
POLICY_PATH="outputs/train/benchmark/<run>/checkpoints/best/pretrained_model" ./run_inference_act.sh

# Force a specific run directory (POLICY_PATH auto-derived)
RUN_DIR="outputs/train/benchmark/<run>" ./run_inference_dp.sh

# Push rollout dataset to HF at end
PUSH_TO_HUB=true ./run_inference_pi05.sh
```
