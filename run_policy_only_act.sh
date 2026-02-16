#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot}"
if ! command -v lerobot-record >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

TOP_CAM="${TOP_CAM:-/dev/video2}"
WRIST_CAM="${WRIST_CAM:-/dev/video0}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-so100_follower_main}"

HF_USER="${HF_USER:-Autobrik}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/eval_act_dump-pocket-cleaning2}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-60}"
SINGLE_TASK="${SINGLE_TASK:-use the scoop-like end effector to gather sand from the dump pocket and deposit it into the primary crusher at the center}"

DISPLAY_DATA="${DISPLAY_DATA:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
PRIVATE="${PRIVATE:-false}"
RESUME="${RESUME:-false}"
MANUAL_EPISODE_CONTROL="${MANUAL_EPISODE_CONTROL:-false}"

POLICY_PATH="${POLICY_PATH:-/home/leonardo/NONHUMAN/lerobot/outputs/hf_checkpoints/Autobrik__SO-ARM100-dump-pocket-cleaning2-act}"

resolve_policy_path() {
  local input_path="$1"
  local p
  p="$(realpath -m "${input_path}")"
  if [[ -f "${p}/config.json" && -f "${p}/model.safetensors" ]]; then
    echo "${p}"
    return 0
  fi
  if [[ -f "${p}/checkpoints/best/pretrained_model/config.json" && -f "${p}/checkpoints/best/pretrained_model/model.safetensors" ]]; then
    echo "${p}/checkpoints/best/pretrained_model"
    return 0
  fi
  if [[ -f "${p}/checkpoints/last/pretrained_model/config.json" && -f "${p}/checkpoints/last/pretrained_model/model.safetensors" ]]; then
    echo "${p}/checkpoints/last/pretrained_model"
    return 0
  fi
  return 1
}

if ! RESOLVED_POLICY_PATH="$(resolve_policy_path "${POLICY_PATH}")"; then
  echo "Could not resolve valid ACT pretrained_model from POLICY_PATH=${POLICY_PATH}" >&2
  exit 1
fi

echo "Dataset repo:   ${DATASET_REPO_ID}"
echo "Policy path:    ${RESOLVED_POLICY_PATH}"
echo "Robot port:     ${ROBOT_PORT}"
echo "Top camera:     ${TOP_CAM}"
echo "Wrist camera:   ${WRIST_CAM}"
echo "Push to hub:    ${PUSH_TO_HUB}"

lerobot-record \
  --robot.type=so100_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --display_data="${DISPLAY_DATA}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --dataset.private="${PRIVATE}" \
  --dataset.manual_episode_control="${MANUAL_EPISODE_CONTROL}" \
  --policy.path="${RESOLVED_POLICY_PATH}" \
  --resume="${RESUME}"
