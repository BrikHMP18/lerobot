#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot-smolvla}"
if ! command -v lerobot-record >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

MODEL_KEY="${MODEL_KEY:-smolvla}"
TOP_CAM="${TOP_CAM:-/dev/video2}"
WRIST_CAM="${WRIST_CAM:-/dev/video4}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so100_follower_main}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

DATASET_REPO_ID="${DATASET_REPO_ID:-Autobrik/SO-ARM100-dump-pocket-cleaning2-eval-smolvla}"
DATASET_ROOT="${DATASET_ROOT:-}"
SINGLE_TASK="${SINGLE_TASK:-use the scoop-like end effector to gather sand from the dump pocket and deposit it into the primary crusher at the center}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
PRIVATE="${PRIVATE:-false}"
MANUAL_EPISODE_CONTROL="${MANUAL_EPISODE_CONTROL:-true}"
NUM_EPISODES="${NUM_EPISODES:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-15}"
RESUME="${RESUME:-auto}"

RUN_DIR="${RUN_DIR:-}"
POLICY_PATH="${POLICY_PATH:-}"

if [[ -z "${POLICY_PATH}" ]]; then
  if [[ -n "${RUN_DIR}" ]]; then
    POLICY_PATH="${RUN_DIR}/checkpoints/best/pretrained_model"
  else
    LATEST_RUN="$(ls -dt outputs/train/benchmark/*_benchmark_${MODEL_KEY} 2>/dev/null | head -n1 || true)"
    if [[ -z "${LATEST_RUN}" ]]; then
      echo "Could not auto-detect run dir for model '${MODEL_KEY}'." >&2
      echo "Set RUN_DIR or POLICY_PATH explicitly." >&2
      exit 1
    fi
    RUN_DIR="${LATEST_RUN}"
    POLICY_PATH="${RUN_DIR}/checkpoints/best/pretrained_model"
  fi
fi

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "POLICY_PATH does not exist or is not a directory: ${POLICY_PATH}" >&2
  echo "Expected a local checkpoint directory like .../checkpoints/best/pretrained_model" >&2
  exit 1
fi

if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Missing ${POLICY_PATH}/config.json" >&2
  echo "Set POLICY_PATH to a valid pretrained_model directory." >&2
  exit 1
fi

if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
  echo "Missing ${POLICY_PATH}/model.safetensors" >&2
  echo "Set POLICY_PATH to a valid pretrained_model directory." >&2
  exit 1
fi

if [[ "${RESUME}" == "auto" ]]; then
  DATA_ROOT_DEFAULT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
  DATASET_BASE="${DATASET_ROOT:-${DATA_ROOT_DEFAULT}}"
  DATASET_PATH="${DATASET_BASE}/${DATASET_REPO_ID}"
  if [[ -f "${DATASET_PATH}/meta/info.json" ]]; then
    RESUME_RESOLVED="true"
  else
    RESUME_RESOLVED="false"
  fi
elif [[ "${RESUME}" == "true" || "${RESUME}" == "false" ]]; then
  RESUME_RESOLVED="${RESUME}"
else
  echo "RESUME must be one of: auto, true, false. Got: ${RESUME}" >&2
  exit 1
fi

echo "Model key:      ${MODEL_KEY}"
echo "Run dir:        ${RUN_DIR:-N/A (POLICY_PATH provided explicitly)}"
echo "Policy path:    ${POLICY_PATH}"
echo "Dataset repo:   ${DATASET_REPO_ID}"
echo "Dataset root:   ${DATASET_ROOT:-<default HF_LEROBOT_HOME>}"
echo "Resume mode:    ${RESUME_RESOLVED} (from RESUME=${RESUME})"
echo "Manual control: ${MANUAL_EPISODE_CONTROL}"
echo "Controls: Right Arrow=save+next | Left Arrow=rerecord | Esc=save+exit"

DATASET_ROOT_ARGS=()
if [[ -n "${DATASET_ROOT}" ]]; then
  DATASET_ROOT_ARGS+=(--dataset.root="${DATASET_ROOT}")
fi

lerobot-record \
  --robot.type=so100_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --policy.path="${POLICY_PATH}" \
  --display_data="${DISPLAY_DATA}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  "${DATASET_ROOT_ARGS[@]}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --dataset.private="${PRIVATE}" \
  --dataset.manual_episode_control="${MANUAL_EPISODE_CONTROL}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --resume="${RESUME_RESOLVED}"
