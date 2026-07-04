#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src/gmodiff"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main_test_gmodiff.py \
  --pretrained_model stabilityai/stable-diffusion-2-1-base \
  --datasets ../../configs/gmodiff.json \
  --output_dir ../../results/gmodiff_test \
  --gmodiff_path ../../model_zoo/gmodiff_81000.pkl \
  --dareg_path ../../model_zoo/dareg_mask.pth
