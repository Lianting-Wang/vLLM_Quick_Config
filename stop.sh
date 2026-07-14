#!/bin/bash
set -euo pipefail

usage() {
  cat <<EOF
Usage:
  ./stop.sh          Stop the legacy single vLLM instance.
  ./stop.sh PROFILE  Stop the vLLM instance for PROFILE.
  ./stop.sh --all    Stop all vLLM instances started by run.sh.
EOF
}

profile_file_key() {
  printf '%s\n' "$1" | tr -c '[:alnum:]._' '_'
}

stop_pidfile() {
  local pidfile="$1"
  local pid

  if [ ! -f "$pidfile" ]; then
    echo "No PID file found: $pidfile"
    return 0
  fi

  pid="$(< "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping vLLM PID $pid"
    kill "$pid"
    sleep 5

    if kill -0 "$pid" 2>/dev/null; then
      echo "Process still running. Force killing PID $pid"
      kill -9 "$pid"
    fi
  else
    echo "PID $pid is not running."
  fi

  rm -f "$pidfile"
}

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  "")
    stop_pidfile "vllm_server.pid"
    ;;
  --all)
    if [ "$#" -ne 1 ]; then
      usage
      exit 1
    fi

    shopt -s nullglob
    PIDFILES=(vllm_server.pid vllm_server.*.pid)
    if [ "${#PIDFILES[@]}" -eq 0 ]; then
      echo "No profile-specific PID files found."
      exit 0
    fi

    for pidfile in "${PIDFILES[@]}"; do
      stop_pidfile "$pidfile"
    done
    ;;
  *)
    if [ "$#" -ne 1 ]; then
      usage
      exit 1
    fi

    stop_pidfile "vllm_server.$(profile_file_key "$1").pid"
    ;;
esac
