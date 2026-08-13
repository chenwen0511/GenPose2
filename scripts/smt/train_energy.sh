#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/quinn/smt
REPO="$ROOT/GenPose2"
DATA_PATH="${DATA_PATH:-$ROOT/datasets/genpose2_smt_train_v1}"
SCORE_CKPT="${SCORE_CKPT:-}"
LOG_DIR="${LOG_DIR:-EnergyNet_smt_v1}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${N_EPOCHS:-20}"
WORKERS="${NUM_WORKERS:-8}"

test -d "$REPO"
test -d "$DATA_PATH"
test -f "$REPO/configs/obj_meta.json"
if [ -z "$SCORE_CKPT" ] || [ ! -s "$SCORE_CKPT" ]; then
  echo "SCORE_CKPT must point to a non-empty final ScoreNet checkpoint" >&2
  exit 2
fi

FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -d ' ')"
if [ "${ALLOW_BUSY_GPU:-0}" != "1" ] && [ "$FREE_MIB" -lt 14336 ]; then
  echo "GPU $GPU has only ${FREE_MIB} MiB free; require at least 14336 MiB" >&2
  exit 3
fi

PY=/home/ubuntu/miniconda3/envs/genpose2/bin/python
test -x "$PY"
cd "$REPO"
mkdir -p "$ROOT/logs" "$ROOT/results/ckpts" "$ROOT/results/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export OPENCV_IO_ENABLE_OPENEXR=1

"$PY" "$ROOT/scripts/preflight_dataset.py" "$DATA_PATH" \
  > "$ROOT/validation/preflight_energy_$(date +%Y%m%d_%H%M%S).json"

exec "$PY" runners/trainer.py \
  --data_path "$DATA_PATH" \
  --log_dir "$LOG_DIR" \
  --agent_type energy \
  --pretrained_score_model_path "$SCORE_CKPT" \
  --sampler_mode ode \
  --sampling_steps 500 \
  --eval_freq 1 \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$EPOCHS" \
  --percentage_data_for_train 1.0 \
  --percentage_data_for_test 1.0 \
  --percentage_data_for_val 1.0 \
  --seed 0 \
  --is_train \
  --load_per_object \
  --pose_mode rot_matrix \
  --regression_head Rx_Ry_and_T \
  --pts_encoder pointnet2 \
  --dino pointwise \
  --num_workers "$WORKERS" \
  --lr 1e-4 \
  --repeat_num 12
