#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot}"
if ! command -v lerobot-train >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

DATASET_REPO_ID="${DATASET_REPO_ID:-Autobrik/SO-ARM100-dump-pocket-cleaning2}"
DATASET_ROOT="${DATASET_ROOT:-}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
SPLIT_SEED="${SPLIT_SEED:-2026}"
export DATASET_REPO_ID DATASET_ROOT TRAIN_RATIO SPLIT_SEED

TARGET_EPOCHS="${TARGET_EPOCHS:-4}"
OVERRIDE_STEPS="${OVERRIDE_STEPS:-}"
USE_REFERENCE_STEPS="${USE_REFERENCE_STEPS:-false}"
REFERENCE_STEPS="${REFERENCE_STEPS:-100000}"
VAL_FREQ="${VAL_FREQ:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
CHECKPOINTS_PER_EPOCH="${CHECKPOINTS_PER_EPOCH:-1}"
VALS_PER_EPOCH="${VALS_PER_EPOCH:-2}"
LOG_FREQ="${LOG_FREQ:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

if (( CHECKPOINTS_PER_EPOCH < 1 )); then
  echo "CHECKPOINTS_PER_EPOCH must be >= 1" >&2
  exit 1
fi
if (( VALS_PER_EPOCH < 1 )); then
  echo "VALS_PER_EPOCH must be >= 1" >&2
  exit 1
fi

PI05_BASE="${PI05_BASE:-lerobot/pi05_base}"
PI05_DTYPE="${PI05_DTYPE:-bfloat16}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-dump-pocket-benchmark}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_NAME="${JOB_NAME:-benchmark_pi05}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/benchmark/${RUN_TS}_${JOB_NAME}}"

TOTAL_EPISODES="$(
  python - <<'PY'
import os
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

repo_id = os.environ["DATASET_REPO_ID"]
root = os.environ.get("DATASET_ROOT") or None
meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
print(meta.info["total_episodes"])
PY
)"
export TOTAL_EPISODES

if (( TOTAL_EPISODES < 2 )); then
  echo "Dataset needs at least 2 episodes for train/val split. Found: ${TOTAL_EPISODES}" >&2
  exit 1
fi

readarray -t SPLIT_EPISODES < <(
  python - <<'PY'
import os
import random

total = int(os.environ["TOTAL_EPISODES"])
train_ratio = float(os.environ["TRAIN_RATIO"])
seed = int(os.environ["SPLIT_SEED"])

episodes = list(range(total))
random.Random(seed).shuffle(episodes)
train_count = int(total * train_ratio)
train_count = max(1, min(total - 1, train_count))

train_eps = sorted(episodes[:train_count])
val_eps = sorted(episodes[train_count:])

def as_cli_list(values):
    return "[" + ",".join(str(v) for v in values) + "]"

print(as_cli_list(train_eps))
print(as_cli_list(val_eps))
PY
)

TRAIN_EPISODES="${SPLIT_EPISODES[0]}"
VAL_EPISODES="${SPLIT_EPISODES[1]}"
export TRAIN_EPISODES

DROP_LAST_FRAMES=0
export BATCH_SIZE TARGET_EPOCHS DROP_LAST_FRAMES
if [[ -n "${OVERRIDE_STEPS}" ]]; then
  TRAIN_STEPS="${OVERRIDE_STEPS}"
  STEP_MODE="override"
  STEP_REFERENCE="n/a"
  STEPS_PER_EPOCH="custom"
  TRAIN_SAMPLES="custom"
elif [[ "${USE_REFERENCE_STEPS}" == "true" ]]; then
  TRAIN_STEPS="${REFERENCE_STEPS}"
  STEP_MODE="reference_steps"
  STEP_REFERENCE="${REFERENCE_STEPS}"
  STEPS_PER_EPOCH="custom"
  TRAIN_SAMPLES="custom"
else
  readarray -t STEP_INFO < <(
    python - <<'PY'
import ast
import math
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = os.environ["DATASET_REPO_ID"]
root = os.environ.get("DATASET_ROOT") or None
episodes = ast.literal_eval(os.environ["TRAIN_EPISODES"])
batch_size = int(os.environ["BATCH_SIZE"])
target_epochs = float(os.environ["TARGET_EPOCHS"])
drop_last_frames = int(os.environ["DROP_LAST_FRAMES"])

ds = LeRobotDataset(repo_id, root=root, episodes=episodes)
train_samples = max(1, ds.num_frames - drop_last_frames * len(episodes))
steps_per_epoch = math.ceil(train_samples / batch_size)
train_steps = max(1, math.ceil(steps_per_epoch * target_epochs))

print(train_steps)
print(steps_per_epoch)
print(train_samples)
PY
  )
  TRAIN_STEPS="${STEP_INFO[0]}"
  STEPS_PER_EPOCH="${STEP_INFO[1]}"
  TRAIN_SAMPLES="${STEP_INFO[2]}"
  STEP_MODE="target_epochs"
  STEP_REFERENCE="n/a"
fi

if [[ -z "${SAVE_FREQ}" ]]; then
  if [[ "${STEPS_PER_EPOCH}" =~ ^[0-9]+$ ]]; then
    SAVE_FREQ=$(( (STEPS_PER_EPOCH + CHECKPOINTS_PER_EPOCH - 1) / CHECKPOINTS_PER_EPOCH ))
  else
    SAVE_FREQ=1000
  fi
fi

if [[ -z "${VAL_FREQ}" ]]; then
  if [[ "${STEPS_PER_EPOCH}" =~ ^[0-9]+$ ]]; then
    VAL_FREQ=$(( (STEPS_PER_EPOCH + VALS_PER_EPOCH - 1) / VALS_PER_EPOCH ))
  else
    VAL_FREQ=1000
  fi
fi

if (( SAVE_FREQ < 1 )); then
  SAVE_FREQ=1
fi
if (( VAL_FREQ < 1 )); then
  VAL_FREQ=1
fi

EST_SAVED_CHECKPOINTS=$(( (TRAIN_STEPS + SAVE_FREQ - 1) / SAVE_FREQ ))
EST_VAL_RUNS=$(( (TRAIN_STEPS - 1) / VAL_FREQ + 1 ))

echo "Training pi05 with ${DATASET_REPO_ID}"
echo "Train episodes: ${TRAIN_EPISODES}"
echo "Val episodes:   ${VAL_EPISODES}"
echo "Train samples:  ${TRAIN_SAMPLES}"
echo "Steps/epoch:    ${STEPS_PER_EPOCH}"
echo "Train steps:    ${TRAIN_STEPS}"
echo "Step mode:      ${STEP_MODE}"
echo "Reference step: ${STEP_REFERENCE}"
echo "Save freq:      ${SAVE_FREQ} (est. checkpoints=${EST_SAVED_CHECKPOINTS})"
echo "Val freq:       ${VAL_FREQ} (est. val runs=${EST_VAL_RUNS})"
echo "Output dir:     ${OUTPUT_DIR}"

DATASET_ROOT_ARGS=()
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ROOT_ARGS+=(--dataset.root="${DATASET_ROOT}")
fi

WANDB_ARGS=(
  --wandb.enable="${WANDB_ENABLE}"
  --wandb.project="${WANDB_PROJECT}"
)
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ARGS+=(--wandb.entity="${WANDB_ENTITY}")
fi

lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  "${DATASET_ROOT_ARGS[@]}" \
  --dataset.episodes="${TRAIN_EPISODES}" \
  --policy.type=pi05 \
  --policy.pretrained_path="${PI05_BASE}" \
  --policy.device="${POLICY_DEVICE}" \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.dtype="${PI05_DTYPE}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --seed="${SEED}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${TRAIN_STEPS}" \
  --eval_freq=-1 \
  --log_freq="${LOG_FREQ}" \
  --save_freq="${SAVE_FREQ}" \
  --offline_val.enable=true \
  --offline_val.episodes="${VAL_EPISODES}" \
  --offline_val.freq="${VAL_FREQ}" \
  --offline_val.track_best_checkpoint=true \
  "${WANDB_ARGS[@]}"
