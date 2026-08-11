#!/bin/bash
# 使用 datasets/train_set_1st_0810 训练 ScaleNet（需已训好的 ScoreNet）
# 料盘尺寸固定时通常可跳过本步，推理写死 size_3d 即可。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-$ROOT/datasets/train_set_1st_0810/SOPE}"
LOG_DIR="${LOG_DIR:-ScaleNet_tray_0810}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_EPOCHS="${N_EPOCHS:-4}"
NUM_WORKER="${NUM_WORKER:-8}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
# 默认取 ScoreNet_tray_0810 下最新 ckpt_epoch*.pth；可用 SCORE_CKPT 覆盖
SCORE_DIR="${SCORE_DIR:-$ROOT/results/ckpts/ScoreNet_tray_0810}"
if [[ -z "${SCORE_CKPT:-}" ]]; then
  SCORE_CKPT="$(ls -1t "$SCORE_DIR"/ckpt_epoch*.pth 2>/dev/null | head -1 || true)"
fi

if [[ ! -f configs/obj_meta.json ]]; then
  echo "[error] 缺少 configs/obj_meta.json"
  exit 1
fi
if [[ -z "${SCORE_CKPT}" || ! -f "$SCORE_CKPT" ]]; then
  echo "[error] 找不到 ScoreNet 权重: ${SCORE_CKPT:-<empty>}"
  echo "       先跑 scripts/train_score_tray_0810.sh，或设置 SCORE_CKPT=/path/to/ckpt_epochN.pth"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export OPENCV_IO_ENABLE_OPENEXR=1

echo "[train] ScaleNet  data=$DATA_PATH  score=$SCORE_CKPT"
python runners/trainer.py \
  --data_path "$DATA_PATH" \
  --log_dir "$LOG_DIR" \
  --agent_type scale \
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
  --num_workers "$NUM_WORKER" \
  --pretrained_score_model_path "$SCORE_CKPT"
