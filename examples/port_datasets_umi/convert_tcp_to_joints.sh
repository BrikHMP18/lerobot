#!/bin/bash
# Convert TCP pose dataset to joint angles using Piper IK
# Usage: ./convert_tcp_to_joints.sh

set -e

INPUT_DATASET="${INPUT_DATASET:-$HOME/NONHUMAN/lerobot/examples/port_datasets_umi/pick_the_cup_demo_dataset}"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-NONHUMAN-RESEARCH/pick_the_cup_demo_dataset_joints}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/NONHUMAN/umi_lerobot_datasets/pick_the_cup_demo_dataset_joints}"
DH_IS_OFFSET="${DH_IS_OFFSET:-1}"  # 0=old firmware, 1=new (>=S-V1.6-3)

python examples/port_datasets_umi/convert_tcp_to_joints.py \
  --input-dataset "$INPUT_DATASET" \
  --output-repo-id "$OUTPUT_REPO_ID" \
  --output-dir "$OUTPUT_DIR" \
  --dh-is-offset "$DH_IS_OFFSET" \
  --push-to-hub
