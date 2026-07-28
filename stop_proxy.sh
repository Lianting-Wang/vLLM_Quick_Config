#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${VLLM_PROXY_PID_FILE:-$SCRIPT_DIR/vllm_proxy.pid}"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No proxy PID file found."
  exit 0
fi

PID="$(cat "$PID_FILE" 2>/dev/null || true)"
if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
  echo "Invalid proxy PID file: $PID_FILE"
  rm -f "$PID_FILE"
  exit 0
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Proxy PID $PID is not running. Removing stale PID file."
  rm -f "$PID_FILE"
  exit 0
fi

echo "Stopping proxy PID $PID..."
kill "$PID"

for _ in {1..50}; do
  if ! kill -0 "$PID" 2>/dev/null; then
    break
  fi
  sleep 0.1
done

if kill -0 "$PID" 2>/dev/null; then
  echo "Proxy did not stop gracefully; force killing PID $PID."
  kill -9 "$PID"
fi

rm -f "$PID_FILE"
echo "Proxy stopped."
