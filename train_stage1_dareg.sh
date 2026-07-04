#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src/dareg"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" accelerate launch train_stage1.py \
  --dataset_dir ../../data/ \
  --sub_set ../../data/training_crop256_stride128 \
  --logdir ../../checkpoints/dareg_stage1
