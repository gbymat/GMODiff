#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src/dareg"

python inference_stage2.py \
  --input_dir ../../data \
  --checkpoint ../../checkpoints/dareg_stage2/best_checkpoint.pth \
  --output_dir ../../results/dareg_stage2
