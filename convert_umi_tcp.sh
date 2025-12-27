#!/bin/bash
# Convert UMI .zarr dataset to LeRobot format
# Usage: FPS=30 ./convert_umi.sh

# Configuration
ZARR_PATH="${ZARR_PATH:-$HOME/NONHUMAN/universal_manipulation_interface/example_demo_session/dataset.zarr}"
REPO_ID="${REPO_ID:-NONHUMAN-RESEARCH/pick_the_cup_demo_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/NONHUMAN/umi_lerobot_datasets/pick_the_cup_demo_dataset}"
FPS="${FPS:-30}"
TASK_DESC="${TASK_DESC:-pick up the cup and put it in the plate}"

python examples/port_datasets/port_umi_zarr_tcp.py \
  --zarr-path "$ZARR_PATH" \
  --repo-id "$REPO_ID" \
  --output-dir "$OUTPUT_DIR" \
  --fps "$FPS" \
  --task-description "$TASK_DESC" \
  --push-to-hub