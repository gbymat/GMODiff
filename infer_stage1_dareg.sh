#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src/dareg"

python inference_stage1.py \
  --dataset_dir ../../data/ \
  --pretrained_model ../../checkpoints/dareg_stage1/best_checkpoint.pth \
  --save_dir ../../results/dareg_stage1
