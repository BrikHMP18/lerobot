#!/usr/bin/env bash
set -euo pipefail

TOP_CAM="${TOP_CAM:-/dev/video2}"
WRIST_CAM="${WRIST_CAM:-/dev/video0}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
TELEOP_PORT="${TELEOP_PORT:-/dev/ttyACM1}"

ROBOT_ID="${ROBOT_ID:-so100_follower_main}"
TELEOP_ID="${TELEOP_ID:-so100_leader_main}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
FPS="${FPS:-60}"

lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --teleop.type=so100_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --display_data="${DISPLAY_DATA}" \
  --fps="${FPS}"
