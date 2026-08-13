#!/bin/bash
# 对 datasets/train_set_1st_0810-eval 跑 tray_0810 推理 + 可视化落盘
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs .pids
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG:-$ROOT/logs/infer_tray0810_eval_${TS}.log}"
PID_FILE="$ROOT/.pids/infer_tray0810_eval.pid"
FOREGROUND="${FOREGROUND:-0}"

DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/train_set_1st_0810-eval}"
SCORE_CKPT="${SCORE_CKPT:-$ROOT/results/ckpts/ScoreNet_tray_0810/ckpt_epoch50.pth}"
ENERGY_CKPT="${ENERGY_CKPT:-$ROOT/results/ckpts/EnergyNet_tray_0810/ckpt_epoch50.pth}"
SCALE_CKPT="${SCALE_CKPT:-$ROOT/results/ckpts/ScaleNet_tray_0810/ckpt_epoch4.pth}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1

cmd=(
  python datasets/infer_tray0810_on_sope.py
  --data_root "$DATA_ROOT"
  --score_ckpt "$SCORE_CKPT"
  --energy_ckpt "$ENERGY_CKPT"
  --scale_ckpt "$SCALE_CKPT"
)

echo "[infer] ROOT=$ROOT"
echo "[infer] data=$DATA_ROOT"
echo "[infer] log=$RUN_LOG"
if [[ ! -f datasets/infer_tray0810_on_sope.py ]]; then
  echo "[error] 找不到 datasets/infer_tray0810_on_sope.py（ROOT 是否正确？）"
  exit 1
fi
if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee -a "$RUN_LOG"
else
  nohup "${cmd[@]}" >>"$RUN_LOG" 2>&1 &
  echo $! >"$PID_FILE"
  echo "[infer] pid=$(cat "$PID_FILE")  tail -f $RUN_LOG"
fi
