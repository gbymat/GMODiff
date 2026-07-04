#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src/gmodiff"

python evaluate_metrics.py \
  --pred_dir ../../results/gmodiff_test \
  --gt_dir ../../data/Test \
  --output_csv ../../results/gmodiff_metrics.csv \
  --summary_json ../../results/gmodiff_metrics_summary.json
