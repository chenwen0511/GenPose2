#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate genpose2
export OPENCV_IO_ENABLE_OPENEXR=1 PYTHONUNBUFFERED=1
mkdir -p logs .pids
# stop old
if [[ -f .pids/ui_eval_7862.pid ]]; then
  kill "$(cat .pids/ui_eval_7862.pid)" 2>/dev/null || true
fi
pkill -f 'train_set_1st_0810-eval --host 0.0.0.0 --port 7862' 2>/dev/null || true
sleep 1
: > logs/ui_eval_7862.log
nohup python datasets/ui_display/app.py \
  --root "$ROOT/datasets/train_set_1st_0810-eval" \
  --host 0.0.0.0 --port 7862 \
  >> logs/ui_eval_7862.log 2>&1 &
echo $! > .pids/ui_eval_7862.pid
sleep 3
echo "pid=$(cat .pids/ui_eval_7862.pid)"
ss -ltn | grep 7862 || true
echo "open: http://$(hostname -I | awk '{print $1}'):7862/"
echo "log:  tail -f $ROOT/logs/ui_eval_7862.log"
