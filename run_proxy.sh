#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="${PROXY_CONFIG_FILE:-$SCRIPT_DIR/proxy_config.json}"
PID_FILE="${VLLM_PROXY_PID_FILE:-$SCRIPT_DIR/vllm_proxy.pid}"
LOG_DIR="${VLLM_PROXY_LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="$LOG_DIR/proxy_$(date '+%Y-%m-%d_%H-%M-%S').log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    echo "Proxy is already running with PID $PID"
    exit 1
  fi
  echo "Removing stale proxy PID file: $PID_FILE"
  rm -f "$PID_FILE"
fi

nohup uv run python -m vllm_proxy.app \
  --config "$CONFIG_FILE" \
  >>"$LOG_FILE" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"

# Authentication and configuration validation happen inside the Python app.
# Give immediate startup failures a chance to reach the log before reporting success.
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Proxy failed to start. Recent log output:" >&2
  tail -n 50 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE"
  exit 1
fi

echo "Started smart proxy."
echo "PID: $PID"
echo "Config: $CONFIG_FILE"
echo "Log: $LOG_FILE"
