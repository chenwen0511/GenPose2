#!/bin/bash
# 使用 datasets/train_set_blender_v2 训练 ScaleNet（需已训好的 ScoreNet）
# 料盘尺寸固定时通常可跳过本步，推理写死 size_3d 即可。
# 默认 nohup 后台运行，终端日志写入 logs/；设 FOREGROUND=1 可前台跑。
#
# 用法:
#   bash scripts/train/train_scale_tray_0810.sh
#   tail -f logs/train_scale_tray_0810_*.log
#   FOREGROUND=1 bash scripts/train/train_scale_tray_0810.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-$ROOT/datasets/train_set_blender_v2/SOPE}"
LOG_DIR="${LOG_DIR:-ScaleNet_tray_0810}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_EPOCHS="${N_EPOCHS:-4}"
NUM_WORKER="${NUM_WORKER:-8}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
FOREGROUND="${FOREGROUND:-0}"
# 默认取 ScoreNet_tray_0810 下最新 ckpt_epoch*.pth；可用 SCORE_CKPT 覆盖
SCORE_DIR="${SCORE_DIR:-$ROOT/results/ckpts/ScoreNet_tray_0810}"
if [[ -z "${SCORE_CKPT:-}" ]]; then
  SCORE_CKPT="$(ls -1t "$SCORE_DIR"/ckpt_epoch*.pth 2>/dev/null | head -1 || true)"
fi

if [[ ! -f configs/obj_meta.json ]]; then
  echo "[error] 缺少 configs/obj_meta.json"
  exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
  echo "[error] 数据路径不存在: $DATA_PATH"
  exit 1
fi
if [[ -z "${SCORE_CKPT}" || ! -f "$SCORE_CKPT" ]]; then
  echo "[error] 找不到 ScoreNet 权重: ${SCORE_CKPT:-<empty>}"
  echo "       先跑 scripts/train_score_tray_0810.sh，或设置 SCORE_CKPT=/path/to/ckpt_epochN.pth"
  exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/.pids"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG:-$ROOT/logs/train_scale_tray_0810_${TS}.log}"
PID_FILE="$ROOT/.pids/train_scale_tray_0810.pid"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[error] 已有训练在跑: pid=$old_pid（见 $PID_FILE）"
    echo "        结束后再启: kill $old_pid"
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export OPENCV_IO_ENABLE_OPENEXR=1

cmd=(
  python runners/trainer.py
  --data_path "$DATA_PATH"
  --log_dir "$LOG_DIR"
  --agent_type scale
  --sampler_mode ode
  --sampling_steps 500
  --eval_freq 1
  --batch_size "$BATCH_SIZE"
  --n_epochs "$N_EPOCHS"
  --percentage_data_for_train 1.0
  --percentage_data_for_test 1.0
  --percentage_data_for_val 1.0
  --seed 0
  --is_train
  --dino pointwise
  --num_workers "$NUM_WORKER"
  --pretrained_score_model_path "$SCORE_CKPT"
)

echo "[train] ScaleNet  data=$DATA_PATH  ckpt_log=$LOG_DIR  score=$SCORE_CKPT"
echo "[train] run_log=$RUN_LOG"

if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee -a "$RUN_LOG"
else
  nohup "${cmd[@]}" >>"$RUN_LOG" 2>&1 &
  echo $! >"$PID_FILE"
  echo "[train] 已后台启动 pid=$(cat "$PID_FILE")"
  echo "[train] 查看日志: tail -f $RUN_LOG"
  echo "[train] 结束训练: kill \$(cat $PID_FILE)"
fi
