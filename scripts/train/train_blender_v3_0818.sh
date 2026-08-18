#!/usr/bin/env bash
# blender_v3 微调：0815-stephen 权重 → 0818-stephen
# 顺序：ScoreNet → EnergyNet → ScaleNet（Scale 用新 Score 提特征）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-$ROOT/datasets/train_set_blender_v3/SOPE}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-$ROOT/results/ckpts/0815-stephen}"
OUT_ROOT="${OUT_ROOT:-0818-stephen}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_EPOCHS="${N_EPOCHS:-50}"
SCALE_EPOCHS="${SCALE_EPOCHS:-4}"
NUM_WORKER="${NUM_WORKER:-8}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
FOREGROUND="${FOREGROUND:-0}"

SCORE_INIT="${SCORE_INIT:-$PRETRAIN_ROOT/ScoreNet_tray_0810/ckpt_epoch50.pth}"
ENERGY_INIT="${ENERGY_INIT:-$PRETRAIN_ROOT/EnergyNet_tray_0810/ckpt_epoch50.pth}"
SCALE_INIT="${SCALE_INIT:-$PRETRAIN_ROOT/ScaleNet_tray_0810/ckpt_epoch4.pth}"

mkdir -p "$ROOT/logs" "$ROOT/.pids" "$ROOT/results/ckpts/$OUT_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_LOG:-$ROOT/logs/train_blender_v3_0818_${TS}.log}"
PID_FILE="$ROOT/.pids/train_blender_v3_0818.pid"

if [[ ! -f configs/obj_meta.json ]]; then
  echo "[error] 缺少 configs/obj_meta.json"
  exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
  echo "[error] 数据路径不存在: $DATA_PATH"
  exit 1
fi
for f in "$SCORE_INIT" "$ENERGY_INIT" "$SCALE_INIT"; do
  if [[ ! -f "$f" ]]; then
    echo "[error] 缺少预训练权重: $f"
    exit 1
  fi
done

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[error] 已有训练在跑: pid=$old_pid"
    exit 1
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate genpose2

common=(
  python runners/trainer.py
  --data_path "$DATA_PATH"
  --sampler_mode ode
  --sampling_steps 500
  --eval_freq 1
  --batch_size "$BATCH_SIZE"
  --percentage_data_for_train 1.0
  --percentage_data_for_test 1.0
  --percentage_data_for_val 1.0
  --seed 0
  --is_train
  --use_pretrain
  --dino pointwise
  --num_workers "$NUM_WORKER"
)

run_all() {
  echo "[1/3] ScoreNet  init=$SCORE_INIT  out=results/ckpts/$OUT_ROOT/ScoreNet"
  "${common[@]}" \
    --log_dir "$OUT_ROOT/ScoreNet" \
    --agent_type score \
    --n_epochs "$N_EPOCHS" \
    --pretrained_score_model_path "$SCORE_INIT"

  SCORE_NEW="$(ls -1t "$ROOT/results/ckpts/$OUT_ROOT/ScoreNet"/ckpt_epoch*.pth | head -1)"
  echo "[2/3] EnergyNet init=$ENERGY_INIT  out=results/ckpts/$OUT_ROOT/EnergyNet"
  "${common[@]}" \
    --log_dir "$OUT_ROOT/EnergyNet" \
    --agent_type energy \
    --n_epochs "$N_EPOCHS" \
    --pretrained_energy_model_path "$ENERGY_INIT"

  echo "[3/3] ScaleNet  init=$SCALE_INIT  score=$SCORE_NEW  out=results/ckpts/$OUT_ROOT/ScaleNet"
  "${common[@]}" \
    --log_dir "$OUT_ROOT/ScaleNet" \
    --agent_type scale \
    --n_epochs "$SCALE_EPOCHS" \
    --pretrained_score_model_path "$SCORE_NEW" \
    --pretrained_scale_model_path "$SCALE_INIT"

  echo "[done] ckpts → $ROOT/results/ckpts/$OUT_ROOT"
}

echo "[train] data=$DATA_PATH"
echo "[train] pretrain=$PRETRAIN_ROOT"
echo "[train] out=results/ckpts/$OUT_ROOT"
echo "[train] log=$RUN_LOG"

if [[ "$FOREGROUND" == "1" ]]; then
  run_all 2>&1 | tee -a "$RUN_LOG"
else
  (
    run_all
  ) >>"$RUN_LOG" 2>&1 &
  echo $! >"$PID_FILE"
  echo "[train] 已后台启动 pid=$(cat "$PID_FILE")"
  echo "[train] 查看日志: tail -f $RUN_LOG"
  echo "[train] 结束训练: kill \$(cat $PID_FILE)"
fi
