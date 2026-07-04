#!/usr/bin/env bash
set -e

MODEL_ID=${1:-1}
DATASET_DIR=${2:-../../data}
SAVE_DIR=${3:-results/baselines/${MODEL_ID}}
MODE=${4:-full}
PATCH_SIZE=${5:-256}

python infer_baseline.py \
  --model_id "${MODEL_ID}" \
  --mode "${MODE}" \
  --dataset_dir "${DATASET_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --patch_size "${PATCH_SIZE}"
