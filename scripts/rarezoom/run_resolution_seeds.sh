#!/usr/bin/env bash
# Train global ResNet50 classifiers with batch 32: 224 for seeds 42-44, 512 for seeds 42 and 44.
# 512 seed 42 reuses checkpoints/rarezoom/v3/global512, trained with the same settings.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=checkpoints/rarezoom/v4
mkdir -p "$OUT/logs"

if [ ! -f "$OUT/s512_seed42/best_model.pth" ]; then
  mkdir -p "$OUT/s512_seed42"
  cp checkpoints/rarezoom/v3/global512/{best_model.pth,last_model.pth,history.jsonl} "$OUT/s512_seed42/"
fi

train() {
  local size=$1 seed=$2 workers=$3
  local dir="$OUT/s${size}_seed${seed}"
  [ -f "$dir/best_model.pth" ] && grep -q best_macro_auprc "$OUT/logs/s${size}_seed${seed}.log" 2>/dev/null && return
  rm -rf "$dir"
  python scripts/rarezoom/train_classifier.py --image-size "$size" --batch-size 32 --patience 3 \
    --seed "$seed" --workers "$workers" --save-dir "$dir" > "$OUT/logs/s${size}_seed${seed}.log" 2>&1
}

# Data loading of 1024 PNGs is the bottleneck, so runs share the GPU.
for seed in 42 43 44; do train 224 "$seed" 12 & done
train 512 44 24 &
wait

python scripts/rarezoom/resolution_study.py --seeds 42 44
python scripts/rarezoom/cascade.py --seeds 42 44
