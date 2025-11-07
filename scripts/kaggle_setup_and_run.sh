#!/usr/bin/env bash
set -euo pipefail

echo "[Kaggle] Installing dependencies..."
pip install -q -r requirements.txt || true

echo "[Kaggle] Running LeGrad demo..."
python scripts/kaggle_run.py \
  --image_url "http://images.cocodataset.org/val2017/000000039769.jpg" \
  --prompt "cat" \
  --output_dir "outputs"

echo "[Kaggle] Done. Check the outputs/ directory for saved heatmaps."


