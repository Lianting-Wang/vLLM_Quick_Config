#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PIDFILE="vllm_proxy.pid"
LOG="proxy_$(date +%F_%H-%M-%S).log"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Proxy already running with PID $(cat "$PIDFILE")"
  exit 1
fi
rm -f "$PIDFILE"
nohup uv run python -m vllm_proxy.app --config proxy_config.json >> "$LOG" 2>&1 &
echo "$!" > "$PIDFILE"
echo "Started smart proxy."
echo "PID: $(cat "$PIDFILE")"
echo "Log: $LOG"
