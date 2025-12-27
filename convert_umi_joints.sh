#!/bin/bash
# Convert UMI .zarr dataset to LeRobot format with JOINT ANGLES
# Usage: ROBOT_TYPE=piper ./convert_umi_joints.sh

# Configuration
ZARR_PATH="${ZARR_PATH:-$HOME/NONHUMAN/universal_manipulation_interface/example_demo_session/dataset.zarr}"
REPO_ID="${REPO_ID:-NONHUMAN-RESEARCH/pick_the_cup_demo_dataset_joints}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/NONHUMAN/umi_lerobot_datasets/pick_the_cup_demo_dataset_joints}"
FPS="${FPS:-30}"
ROBOT_TYPE="${ROBOT_TYPE:-piper}"
TASK_DESC="${TASK_DESC:-pick up the cup and put it in the plate}"

python examples/port_datasets/port_umi_zarr_joints.py \
  --zarr-path "$ZARR_PATH" \
  --repo-id "$REPO_ID" \
  --output-dir "$OUTPUT_DIR" \
  --fps "$FPS" \
  --robot-type "$ROBOT_TYPE" \
  --task-description "$TASK_DESC" \
  --push-to-hub
