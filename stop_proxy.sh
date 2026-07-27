#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PIDFILE="vllm_proxy.pid"
if [ ! -f "$PIDFILE" ]; then echo "No proxy PID file found."; exit 0; fi
PID="$(cat "$PIDFILE")"
if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping proxy PID $PID"
  kill "$PID"
  for _ in $(seq 1 50); do kill -0 "$PID" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$PID" 2>/dev/null; then echo "Force killing proxy PID $PID"; kill -9 "$PID"; fi
fi
rm -f "$PIDFILE"
