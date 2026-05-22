#!/bin/bash

PIDFILE=vllm_server.pid

if [ ! -f "$PIDFILE" ]; then
  echo "No PID file found."
  exit 0
fi

PID=$(cat "$PIDFILE")

if kill -0 "$PID" 2>/dev/null; then
  echo "Stopping vLLM PID $PID"
  kill "$PID"
  sleep 5

  if kill -0 "$PID" 2>/dev/null; then
    echo "Process still running. Force killing PID $PID"
    kill -9 "$PID"
  fi
else
  echo "PID $PID is not running."
fi

rm -f "$PIDFILE"