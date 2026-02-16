#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-lerobot}"
if ! command -v huggingface-cli >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -dt outputs/train/benchmark/*_benchmark_diffusion 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "Could not find RUN_DIR. Set RUN_DIR explicitly, e.g.:" >&2
  echo "  RUN_DIR=outputs/train/benchmark/20260215_203552_benchmark_diffusion ./run_upload_checkpoints_dp.sh" >&2
  exit 1
fi

CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/checkpoints}"
if [[ ! -d "${CKPT_DIR}/best" ]]; then
  echo "Missing checkpoint directory: ${CKPT_DIR}/best" >&2
  exit 1
fi
if [[ ! -d "${CKPT_DIR}/last" ]]; then
  echo "Missing checkpoint directory: ${CKPT_DIR}/last" >&2
  exit 1
fi

MODEL_REPO="${MODEL_REPO:-Autobrik/SO-ARM100-dump-pocket-cleaning2-diffusion}"
DATASET_REPO_ID="${DATASET_REPO_ID:-Autobrik/SO-ARM100-dump-pocket-cleaning2}"
SEED="${SEED:-1000}"
POLICY_TYPE="${POLICY_TYPE:-diffusion}"
MODEL_CARD_PATH="${MODEL_CARD_PATH:-/tmp/README_DIFFUSION_MODEL.md}"

BEST_VAL_PATH="${BEST_VAL_PATH:-${RUN_DIR}/best_val.json}"
if [[ ! -f "${BEST_VAL_PATH}" ]]; then
  echo "Warning: best_val.json not found at ${BEST_VAL_PATH}. README will use N/A values." >&2
fi
export RUN_DIR BEST_VAL_PATH MODEL_CARD_PATH DATASET_REPO_ID POLICY_TYPE SEED

python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
best_val_path = Path(os.environ["BEST_VAL_PATH"])
out_path = Path(os.environ["MODEL_CARD_PATH"])
dataset_repo_id = os.environ["DATASET_REPO_ID"]
policy_type = os.environ["POLICY_TYPE"]
seed = os.environ["SEED"]

best_step = "N/A"
best_loss = "N/A"
if best_val_path.exists():
    try:
        payload = json.loads(best_val_path.read_text())
        best_step = payload.get("best_step", payload.get("step", payload.get("best_val_step", "N/A")))
        best_loss = payload.get("best_val_loss", payload.get("loss", payload.get("best_val_loss", "N/A")))
    except Exception:
        pass

readme = f"""---
library_name: lerobot
tags:
- lerobot
- robotics
- imitation-learning
- diffusion
---

# Diffusion Policy finetuned on SO-ARM100 dump-pocket-cleaning

This is a **Diffusion Policy** finetuned with LeRobot.

## Training summary
- policy: `{policy_type}`
- dataset: `{dataset_repo_id}`
- run_dir: `{run_dir}`
- seed: `{seed}`

## Best offline validation
- best_val_step: `{best_step}`
- best_val_loss: `{best_loss}`

## Included artifacts
- `checkpoints/best` (recommended for eval/deploy)
- `checkpoints/last` (resume/debug)
- `best_val.json`
"""
out_path.write_text(readme)
print(f"Wrote model card: {out_path}")
PY

echo "Creating model repo (if needed): ${MODEL_REPO}"
huggingface-cli repo create "${MODEL_REPO}" --type model -y || true

echo "Uploading model card"
huggingface-cli upload "${MODEL_REPO}" "${MODEL_CARD_PATH}" README.md --repo-type model

echo "Uploading checkpoints/best"
huggingface-cli upload "${MODEL_REPO}" "${CKPT_DIR}/best" checkpoints/best --repo-type model

echo "Uploading checkpoints/last"
huggingface-cli upload "${MODEL_REPO}" "${CKPT_DIR}/last" checkpoints/last --repo-type model

if [[ -f "${BEST_VAL_PATH}" ]]; then
  echo "Uploading best_val.json"
  huggingface-cli upload "${MODEL_REPO}" "${BEST_VAL_PATH}" best_val.json --repo-type model
fi

echo "Done: https://huggingface.co/${MODEL_REPO}"
