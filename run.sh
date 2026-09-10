#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/data/medasr.pids"

load_env() {
  cd "$ROOT"
  if [ -f .env ]; then
    set -a
    source .env
    set +a
  fi
}

stop_service() {
  if [ ! -f "$PID_FILE" ]; then
    echo "MedASR is not running"
    return 0
  fi

  local pid
  while read -r pid; do
    case "$pid" in (*[!0-9]*|'') continue ;; esac
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done < "$PID_FILE"

  local deadline=$((SECONDS + 20))
  while [ "$SECONDS" -lt "$deadline" ]; do
    local alive=0
    while read -r pid; do
      case "$pid" in (*[!0-9]*|'') continue ;; esac
      kill -0 "$pid" 2>/dev/null && alive=1
    done < "$PID_FILE"
    [ "$alive" -eq 0 ] && break
    sleep 1
  done

  while read -r pid; do
    case "$pid" in (*[!0-9]*|'') continue ;; esac
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done < "$PID_FILE"

  : > "$PID_FILE"
  echo "MedASR stopped"
}

status_service() {
  load_env
  local python_bin="${MEDASR_PYTHON:-python3}"
  local port
  port="$("$python_bin" -c 'from asrsvc import config; print(config.MEDASR_PORT)')"
  if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${port}/health"; then
    echo
  else
    echo "MedASR is not running or not ready" >&2
    return 1
  fi
}

stop_on_signal() {
  trap - EXIT INT TERM HUP
  stop_service
  exit 130
}

start_service() {
  load_env
  local python_bin="${MEDASR_PYTHON:-python3}"
  local stagger="${MEDASR_STAGGER:-2}"
  local timeout="${MEDASR_STARTUP_TIMEOUT:-900}"
  mkdir -p logs data/spool

  if [ -s "$PID_FILE" ]; then
    while read -r pid; do
      case "$pid" in (*[!0-9]*|'') continue ;; esac
      if kill -0 "$pid" 2>/dev/null; then
        echo "MedASR is already running (pid=$pid)" >&2
        return 1
      fi
    done < "$PID_FILE"
  fi

  if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
    echo "Warning: FFmpeg is unavailable; some offline audio formats may not be supported." >&2
  fi
  command -v setsid >/dev/null || { echo "setsid is required" >&2; return 1; }

  "$python_bin" -m asrsvc.check_config

  local plan
  plan="$("$python_bin" -c 'from asrsvc import config; [print(c, p) for c, p in config.MEDASR_INSTANCE_PLAN]')"
  if [ -z "$plan" ]; then
    echo "No GPU was detected. Set MEDASR_CARDS or expose NVIDIA devices." >&2
    return 1
  fi

  export MEDASR_CARDS="$(awk '{printf "%s%s", (NR > 1 ? "," : ""), $1}' <<< "$plan")"
  : > "$PID_FILE"
  trap stop_service EXIT
  trap stop_on_signal INT TERM HUP

  local card port
  local worker_total=0
  while read -r card port; do
    [ -z "$port" ] && continue
    worker_total=$((worker_total + 1))
    echo "Starting worker on GPU ${card}, port ${port} (log: logs/worker_${port}.log)"
    setsid env CUDA_VISIBLE_DEVICES="$card" MEDASR_WORKER_PORT="$port" \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      "$python_bin" -m asrsvc.worker.qwen_app > "logs/worker_${port}.log" 2>&1 &
    echo "$!" >> "$PID_FILE"
    sleep "$stagger"
  done <<< "$plan"

  local deadline=$((SECONDS + timeout))
  local ready=0
  local total=0
  echo "Waiting for ${worker_total} worker(s) to become ready..."
  while [ "$SECONDS" -lt "$deadline" ]; do
    ready=0
    total=0
    while read -r _ port; do
      [ -z "$port" ] && continue
      total=$((total + 1))
      curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:${port}/health" 2>/dev/null \
        | grep -q '"ready": *true' && ready=$((ready + 1)) || true
    done <<< "$plan"
    printf '\rWorkers ready: %s/%s' "$ready" "$total"
    [ "$ready" -eq "$total" ] && break
    sleep 5
  done
  echo

  if [ "$ready" -ne "$total" ]; then
    echo "Workers did not become ready before timeout; see logs/." >&2
    return 1
  fi

  setsid "$python_bin" -m asrsvc.gateway.app > logs/gateway.log 2>&1 &
  local gateway_pid=$!
  echo "$gateway_pid" >> "$PID_FILE"

  local gateway_port="${MEDASR_PORT:-18080}"
  local gateway_deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$gateway_deadline" ]; do
    if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:${gateway_port}/health" >/dev/null 2>&1; then
      trap - EXIT INT TERM HUP
      echo "MedASR started on port ${gateway_port} (log: logs/gateway.log)"
      return 0
    fi
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
      echo "Gateway failed to start; see logs/gateway.log" >&2
      tail -20 logs/gateway.log >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Gateway did not become ready before timeout; see logs/gateway.log" >&2
  return 1
}

usage() {
  echo "Usage: bash run.sh --start|--stop|--status|--restart"
}

case "${1:-}" in
  --start) start_service ;;
  --stop) stop_service ;;
  --status) status_service ;;
  --restart) stop_service; start_service ;;
  --help|-h) usage ;;
  *) usage; exit 2 ;;
esac
