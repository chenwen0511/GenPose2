#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/quinn/smt
STAGE="${1:-}"
if ! command -v tmux >/dev/null 2>&1; then
  echo 'tmux is required for persistent training but is not installed' >&2
  exit 6
fi
case "$STAGE" in
  score|energy) ;;
  *) echo "usage: $0 score|energy" >&2; exit 2 ;;
esac

SESSION="genpose2_smt_${STAGE}_v1"
PID_FILE="$ROOT/logs/${SESSION}.pid"
RUN_LOG="$ROOT/logs/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
TRAIN_SCRIPT="$ROOT/scripts/train_${STAGE}.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 3
fi
if [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "training process already running: pid=$(cat "$PID_FILE")" >&2
  exit 4
fi

mkdir -p "$ROOT/logs"
tmux new-session -d -s "$SESSION"
launch="cd $ROOT && nohup bash $TRAIN_SCRIPT > $RUN_LOG 2>&1 & echo \$! > $PID_FILE"
tmux send-keys -t "$SESSION" "$launch" C-m

echo "session=$SESSION"
echo "pid_file=$PID_FILE"
echo "log=$RUN_LOG"
echo "status: bash $ROOT/scripts/status.sh $STAGE"
