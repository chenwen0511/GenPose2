#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/quinn/smt
REPO="$ROOT/GenPose2"
DATA_PATH="${DATA_PATH:-$ROOT/datasets/genpose2_smt_train_v2_20260813/SOPE}"
MODEL_PATH="${SMT_EVAL_MODEL_PATH:-$REPO/assets/tray_z_normal_v2.obj}"
LOG_DIR="${LOG_DIR:-ScoreNet_smt_v2_symmetry}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${N_EPOCHS:-40}"
WORKERS="${NUM_WORKERS:-8}"

test -d "$REPO"
test -d "$DATA_PATH"
test -f "$REPO/configs/obj_meta.json"
test -f "$MODEL_PATH"

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

"$PY" "$REPO/scripts/smt/validate_smt_dataset_v2.py" "$(dirname "$DATA_PATH")" \
  > "$ROOT/validation/preflight_score_$(date +%Y%m%d_%H%M%S).json"

exec "$PY" runners/trainer.py \
  --data_path "$DATA_PATH" \
  --log_dir "$LOG_DIR" \
  --agent_type score \
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
  --symmetry_augment \
  --smt_eval_model_path "$MODEL_PATH" \
  --smt_object_diameter 0.1778 \
  --pose_mode rot_matrix \
  --regression_head Rx_Ry_and_T \
  --pts_encoder pointnet2 \
  --dino pointwise \
  --num_workers "$WORKERS" \
  --lr 1e-4 \
  --repeat_num 12
