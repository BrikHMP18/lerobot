#!/usr/bin/env bash
set -euo pipefail

TOP_CAM="${TOP_CAM:-/dev/video2}"
WRIST_CAM="${WRIST_CAM:-/dev/video4}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
TELEOP_PORT="${TELEOP_PORT:-/dev/ttyACM1}"

DATASET_REPO_ID="${DATASET_REPO_ID:-Autobrik/SO-ARM100-dump-pocket-cleaning2}"
NUM_EPISODES="${NUM_EPISODES:-300}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-15}"
SINGLE_TASK="${SINGLE_TASK:-use the scoop-like end effector to gather sand from the dump pocket and deposit it into the primary crusher at the center}"

DISPLAY_DATA="${DISPLAY_DATA:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
RESUME="${RESUME:-true}"
MANUAL_EPISODE_CONTROL="${MANUAL_EPISODE_CONTROL:-true}"

# In manual mode: right arrow saves current episode and starts next,
# left arrow discards current episode, ESC saves current episode and exits.

lerobot-record \
  --robot.type=so100_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id=so100_follower_main \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --teleop.type=so100_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id=so100_leader_main \
  --display_data="${DISPLAY_DATA}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${SINGLE_TASK}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --dataset.manual_episode_control="${MANUAL_EPISODE_CONTROL}" \
  --resume="${RESUME}"
