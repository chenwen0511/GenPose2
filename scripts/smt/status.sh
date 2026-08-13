#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/quinn/smt
STAGE="${1:-score}"
SESSION="genpose2_smt_${STAGE}_v1"
PID_FILE="$ROOT/logs/${SESSION}.pid"

echo '=== tmux ==='
if command -v tmux >/dev/null 2>&1; then
  tmux has-session -t "$SESSION" 2>&1 || true
  tmux capture-pane -t "$SESSION" -p 2>/dev/null | tail -30 || true
else
  echo 'tmux not installed'
fi
echo '=== process ==='
if [ -s "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE")"
  ps -o pid,user,lstart,etime,args -p "$pid" || true
else
  echo "no pid file: $PID_FILE"
fi
echo '=== GPU ==='
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
echo '=== latest log ==='
latest="$(find "$ROOT/logs" -maxdepth 1 -type f -name "${SESSION}_*.log" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- || true)"
if [ -n "$latest" ]; then
  echo "$latest"
  tail -40 "$latest"
else
  echo 'no log found'
fi
echo '=== checkpoints ==='
find "$ROOT/GenPose2/results/ckpts" -maxdepth 2 -type f -name '*.pth' -printf '%p %s bytes %TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sort | tail -30
