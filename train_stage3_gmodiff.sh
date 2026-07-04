#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src/gmodiff"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}" accelerate launch --main_process_port 18890 main_train_gmodiff.py \
  --pretrained_model stabilityai/stable-diffusion-2-1-base \
  --datasets ../../configs/gmodiff.json \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_train_steps 300000 \
  --checkpointing_steps 5000 \
  --lora_rank 16 \
  --dareg_path ../../model_zoo/dareg_mask.pth \
  --output_dir ../../results/train_gmodiff
