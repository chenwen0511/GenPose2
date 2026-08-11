#!/bin/bash
# 使用 datasets/train_set_1st_0810 从零训练 EnergyNet
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-$ROOT/datasets/train_set_1st_0810/SOPE}"
LOG_DIR="${LOG_DIR:-EnergyNet_tray_0810}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_EPOCHS="${N_EPOCHS:-50}"
NUM_WORKER="${NUM_WORKER:-8}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f configs/obj_meta.json ]]; then
  echo "[error] 缺少 configs/obj_meta.json"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export OPENCV_IO_ENABLE_OPENEXR=1

echo "[train] EnergyNet  data=$DATA_PATH  log=$LOG_DIR"
python runners/trainer.py \
  --data_path "$DATA_PATH" \
  --log_dir "$LOG_DIR" \
  --agent_type energy \
  --sampler_mode ode \
  --sampling_steps 500 \
  --eval_freq 1 \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$N_EPOCHS" \
  --percentage_data_for_train 1.0 \
  --percentage_data_for_test 1.0 \
  --percentage_data_for_val 1.0 \
  --seed 0 \
  --is_train \
  --dino pointwise \
  --num_workers "$NUM_WORKER"
