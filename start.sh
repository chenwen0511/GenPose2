#!/usr/bin/env bash
# GenPose2 Gradio UI：start / stop / restart / status
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/miniconda3/envs/genpose2/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18090}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/.pids"
NAME="ui"
PID_FILE="${PID_DIR}/${NAME}.pid"
LOG_FILE="${LOG_DIR}/${NAME}.log"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

usage() {
  cat <<EOF
用法: bash start.sh {start|stop|restart|status}

环境变量（可选）:
  PYTHON   默认 ${PYTHON}
  HOST     默认 ${HOST}
  PORT     默认 ${PORT}
  CUDA_VISIBLE_DEVICES  默认 ${CUDA_VISIBLE_DEVICES}

UI: http://<host>:${PORT}/
配置: config/conf.json
依赖: pip install -r requirements.txt（需在 genpose2 环境）
EOF
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

port_in_use() {
  ss -tln 2>/dev/null | grep -q ":${1} "
}

wait_for_port() {
  local port="$1"
  local timeout="$2"
  local elapsed=0
  echo -n "[wait] UI ready on port ${port}"
  while (( elapsed < timeout )); do
    if port_in_use "${port}"; then
      echo " ok (${elapsed}s)"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    echo -n "."
  done
  echo " timeout"
  echo "[warn] 未在 ${timeout}s 内监听 ${port}，请查看日志: ${LOG_FILE}"
  return 1
}

cmd_start() {
  if is_running; then
    echo "[skip] UI already running (pid $(cat "${PID_FILE}"))"
    echo "       url: http://${HOST}:${PORT}/"
    return 0
  fi

  if [[ ! -x "${PYTHON}" ]]; then
    echo "Python not found: ${PYTHON}"
    exit 1
  fi

  if port_in_use "${PORT}"; then
    local pids
    pids=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
    echo "[error] port ${PORT} already in use (pid(s): ${pids:-unknown})"
    echo "        bash start.sh stop  或  kill ${pids}"
    exit 1
  fi

  echo "== GenPose2 Gradio UI =="
  echo "  python: ${PYTHON}"
  echo "  url:    http://${HOST}:${PORT}/"
  echo "  sam3:   见 config/conf.json"
  echo "请确认 SAM3 HTTP 服务已启动（默认 18003）。"

  cd "${ROOT_DIR}"
  : >"${LOG_FILE}"
  nohup "${PYTHON}" run_ui.py --host "${HOST}" --port "${PORT}" >>"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "[start] pid=$(cat "${PID_FILE}") log=${LOG_FILE}"

  wait_for_port "${PORT}" "${STARTUP_TIMEOUT}" || true
  echo "[ok] UI: http://${HOST}:${PORT}/"
}

cmd_stop() {
  if ! is_running; then
    if port_in_use "${PORT}"; then
      local pids
      pids=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
      if [[ -n "${pids}" ]]; then
        echo "[stop] port ${PORT} residual pid(s): ${pids}"
        kill ${pids} 2>/dev/null || true
        sleep 1
        kill -9 ${pids} 2>/dev/null || true
      fi
    else
      echo "[skip] UI not running"
    fi
    rm -f "${PID_FILE}"
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  echo "[stop] UI (pid ${pid})"
  kill "${pid}" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] force kill ${pid}"
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "[ok] stopped"
}

cmd_status() {
  if is_running; then
    echo "[status] running pid=$(cat "${PID_FILE}") url=http://${HOST}:${PORT}/"
  elif port_in_use "${PORT}"; then
    echo "[status] port ${PORT} in use, but pid file missing/stale"
  else
    echo "[status] stopped"
  fi
}

cmd_restart() {
  cmd_stop
  cmd_start
}

main() {
  local action="${1:-}"
  case "${action}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_restart ;;
    status) cmd_status ;;
    -h|--help|help|"") usage; [[ -n "${action}" ]] || exit 1 ;;
    *) echo "未知命令: ${action}"; usage; exit 1 ;;
  esac
}

main "$@"
