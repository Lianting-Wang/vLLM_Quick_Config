#!/bin/bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-models.conf}"

usage() {
  cat <<EOF
Usage:
  ./stop.sh            Select a running model interactively.
  ./stop.sh PROFILE    Stop the vLLM instance for PROFILE.
  ./stop.sh --all      Stop all running vLLM instances.
  ./stop.sh --list     List running vLLM instances.

Environment:
  CONFIG_FILE          Model config file path. Default: models.conf
EOF
}

profile_file_key() {
  printf '%s\n' "$1" | tr -c '[:alnum:]._' '_'
}

list_profiles() {
  [ -f "$CONFIG_FILE" ] || return 0
  awk '
    {
      line = $0
      sub(/\r$/, "", line)
      if (line ~ /^[[:space:]]*\[[^]]+\][[:space:]]*$/) {
        section_name = line
        sub(/^[[:space:]]*\[/, "", section_name)
        sub(/\][[:space:]]*$/, "", section_name)
        if (section_name != "defaults") print section_name
      }
    }
  ' "$CONFIG_FILE"
}

pid_is_running() {
  local pidfile="$1"
  [ -s "$pidfile" ] || return 1
  local pid
  pid="$(< "$pidfile")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

cleanup_stale_pidfile() {
  local pidfile="$1"
  if [ -e "$pidfile" ] && ! pid_is_running "$pidfile"; then
    echo "Removing stale PID file: $pidfile" >&2
    rm -f "$pidfile"
  fi
}

stop_pidfile() {
  local pidfile="$1"
  local label="${2:-$pidfile}"
  local pid

  if [ ! -f "$pidfile" ]; then
    echo "$label is not running (PID file not found)."
    return 0
  fi

  pid="$(< "$pidfile")"
  if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid PID file for $label: $pidfile"
    rm -f "$pidfile"
    return 0
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping $label (PID $pid)..."
    kill "$pid"

    for _ in {1..10}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done

    if kill -0 "$pid" 2>/dev/null; then
      echo "$label is still running. Force killing PID $pid."
      kill -9 "$pid"
    fi
    echo "Stopped $label."
  else
    echo "$label PID $pid is not running."
  fi

  rm -f "$pidfile"
}

running_profiles=()
running_pidfiles=()
running_pids=()

collect_running() {
  running_profiles=()
  running_pidfiles=()
  running_pids=()

  local profile key pidfile pid
  while IFS= read -r profile; do
    [ -n "$profile" ] || continue
    key="$(profile_file_key "$profile")"
    pidfile="vllm_server.${key}.pid"
    cleanup_stale_pidfile "$pidfile"
    if pid_is_running "$pidfile"; then
      pid="$(< "$pidfile")"
      running_profiles+=("$profile")
      running_pidfiles+=("$pidfile")
      running_pids+=("$pid")
    fi
  done < <(list_profiles)

  cleanup_stale_pidfile "vllm_server.pid"
  if pid_is_running "vllm_server.pid"; then
    running_profiles+=("legacy")
    running_pidfiles+=("vllm_server.pid")
    running_pids+=("$(< vllm_server.pid)")
  fi

  shopt -s nullglob
  local known candidate basename
  for candidate in vllm_server.*.pid; do
    known=false
    for pidfile in "${running_pidfiles[@]:-}"; do
      if [ "$candidate" = "$pidfile" ]; then
        known=true
        break
      fi
    done
    $known && continue
    cleanup_stale_pidfile "$candidate"
    if pid_is_running "$candidate"; then
      basename="${candidate#vllm_server.}"
      basename="${basename%.pid}"
      running_profiles+=("$basename")
      running_pidfiles+=("$candidate")
      running_pids+=("$(< "$candidate")")
    fi
  done
  shopt -u nullglob
}

print_running() {
  collect_running
  if [ "${#running_profiles[@]}" -eq 0 ]; then
    echo "No running vLLM instances found."
    return 1
  fi
  echo "Running vLLM instances:"
  local i
  for ((i=0; i<${#running_profiles[@]}; i++)); do
    printf '  %d) %s (PID %s)\n' "$((i+1))" "${running_profiles[$i]}" "${running_pids[$i]}"
  done
}

stop_all() {
  collect_running
  if [ "${#running_profiles[@]}" -eq 0 ]; then
    echo "No running vLLM instances found."
    return 0
  fi
  local i
  for ((i=0; i<${#running_profiles[@]}; i++)); do
    stop_pidfile "${running_pidfiles[$i]}" "${running_profiles[$i]}"
  done
}

select_and_stop() {
  collect_running
  if [ "${#running_profiles[@]}" -eq 0 ]; then
    echo "No running vLLM instances found."
    return 0
  fi
  if [ ! -t 0 ]; then
    echo "No interactive terminal. Specify PROFILE or use --all." >&2
    usage >&2
    exit 1
  fi

  echo "Running vLLM instances:"
  local i all_index choice selected
  for ((i=0; i<${#running_profiles[@]}; i++)); do
    printf '  %d) %s (PID %s)\n' "$((i+1))" "${running_profiles[$i]}" "${running_pids[$i]}"
  done
  all_index=$((${#running_profiles[@]} + 1))
  printf '  %d) All running models\n' "$all_index"
  printf 'Choose a model to stop: '
  read -r choice

  if [ "$choice" = "all" ] || [ "$choice" = "ALL" ] || [ "$choice" = "$all_index" ]; then
    stop_all
    return 0
  fi

  if [[ "$choice" =~ ^[0-9]+$ ]]; then
    selected=$((10#$choice - 1))
    if [ "$selected" -lt 0 ] || [ "$selected" -ge "${#running_profiles[@]}" ]; then
      echo "Invalid selection: $choice" >&2
      exit 1
    fi
    stop_pidfile "${running_pidfiles[$selected]}" "${running_profiles[$selected]}"
    return 0
  fi

  for ((i=0; i<${#running_profiles[@]}; i++)); do
    if [ "$choice" = "${running_profiles[$i]}" ]; then
      stop_pidfile "${running_pidfiles[$i]}" "${running_profiles[$i]}"
      return 0
    fi
  done
  echo "Unknown running profile: $choice" >&2
  exit 1
}

case "${1:-}" in
  --help|-h)
    usage
    ;;
  --list|-l)
    print_running || true
    ;;
  --all)
    [ "$#" -eq 1 ] || { usage; exit 1; }
    stop_all
    ;;
  "")
    select_and_stop
    ;;
  *)
    [ "$#" -eq 1 ] || { usage; exit 1; }
    if [ "$1" = "legacy" ]; then
      stop_pidfile "vllm_server.pid" "legacy"
    else
      stop_pidfile "vllm_server.$(profile_file_key "$1").pid" "$1"
    fi
    ;;
esac
