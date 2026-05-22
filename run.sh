#!/bin/bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-models.conf}"
LOG=output_$(date +%F_%H-%M-%S).log
PIDFILE=vllm_server.pid

usage() {
  cat <<EOF
Usage:
  ./run.sh            Select a model profile interactively, or use default_profile when non-interactive.
  ./run.sh PROFILE    Start vLLM with the named model profile.
  ./run.sh --list     List available model profiles.

Environment:
  CONFIG_FILE         Model config file path. Default: models.conf
EOF
}

require_config_file() {
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Model config file not found: $CONFIG_FILE"
    exit 1
  fi
}

config_value() {
  local section="$1"
  local key="$2"

  awk -v want_section="$section" -v want_key="$key" '
    function trim(value) {
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      return value
    }

    {
      line = $0
      sub(/\r$/, "", line)

      if (line ~ /^[[:space:]]*$/) {
        next
      }

      if (line ~ /^[[:space:]]*[#;]/) {
        next
      }

      if (line ~ /^[[:space:]]*\[[^]]+\][[:space:]]*$/) {
        section_name = line
        sub(/^[[:space:]]*\[/, "", section_name)
        sub(/\][[:space:]]*$/, "", section_name)
        in_section = (section_name == want_section)
        next
      }

      if (in_section) {
        equals_at = index(line, "=")
        if (equals_at == 0) {
          next
        }

        key = trim(substr(line, 1, equals_at - 1))
        value = trim(substr(line, equals_at + 1))

        if (key == want_key) {
          print value
          exit
        }
      }
    }
  ' "$CONFIG_FILE"
}

list_profiles() {
  awk '
    {
      line = $0
      sub(/\r$/, "", line)

      if (line ~ /^[[:space:]]*\[[^]]+\][[:space:]]*$/) {
        section_name = line
        sub(/^[[:space:]]*\[/, "", section_name)
        sub(/\][[:space:]]*$/, "", section_name)

        if (section_name != "defaults") {
          print section_name
        }
      }
    }
  ' "$CONFIG_FILE"
}

profile_exists() {
  local profile="$1"

  while IFS= read -r existing_profile; do
    if [ "$existing_profile" = "$profile" ]; then
      return 0
    fi
  done <<EOF
$(list_profiles)
EOF

  return 1
}

print_profiles() {
  echo "Available model profiles:"
  while IFS= read -r profile; do
    [ -n "$profile" ] && echo "  $profile"
  done <<EOF
$(list_profiles)
EOF
}

resolve_value() {
  local key="$1"
  local profile_value
  local default_value

  profile_value="$(config_value "$PROFILE" "$key")"
  if [ -n "$profile_value" ]; then
    printf '%s\n' "$profile_value"
    return 0
  fi

  default_value="$(config_value defaults "$key")"
  printf '%s\n' "$default_value"
}

is_true() {
  case "${1:-}" in
    true|TRUE|yes|YES|1|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

select_profile() {
  local default_profile="$1"
  local profiles=()
  local profile
  local index=1
  local choice
  local selected_index

  while IFS= read -r profile; do
    [ -n "$profile" ] && profiles+=("$profile")
  done <<EOF
$(list_profiles)
EOF

  if [ "${#profiles[@]}" -eq 0 ]; then
    echo "No model profiles found in $CONFIG_FILE" >&2
    exit 1
  fi

  echo "Available model profiles:" >&2
  for profile in "${profiles[@]}"; do
    if [ "$profile" = "$default_profile" ]; then
      echo "  $index) $profile (default)" >&2
    else
      echo "  $index) $profile" >&2
    fi
    index=$((index + 1))
  done

  printf "Choose a profile" >&2
  if [ -n "$default_profile" ]; then
    printf " [%s]" "$default_profile" >&2
  fi
  printf ": " >&2

  if ! read -r choice; then
    echo "No profile selected." >&2
    exit 1
  fi

  if [ -z "$choice" ]; then
    if [ -n "$default_profile" ]; then
      printf '%s\n' "$default_profile"
      return 0
    fi

    echo "No profile selected." >&2
    exit 1
  fi

  case "$choice" in
    *[!0-9]*)
      printf '%s\n' "$choice"
      ;;
    *)
      selected_index=$((10#$choice - 1))
      if [ "$selected_index" -lt 0 ] || [ "$selected_index" -ge "${#profiles[@]}" ]; then
        echo "Invalid profile selection: $choice" >&2
        exit 1
      fi
      printf '%s\n' "${profiles[$selected_index]}"
      ;;
  esac
}

require_config_file

case "${1:-}" in
  --help|-h)
    usage
    exit 0
    ;;
  --list|-l)
    print_profiles
    exit 0
    ;;
  "")
    DEFAULT_PROFILE="$(config_value defaults default_profile)"
    if [ -t 0 ]; then
      PROFILE="$(select_profile "$DEFAULT_PROFILE")"
    else
      PROFILE="$DEFAULT_PROFILE"
    fi
    ;;
  *)
    if [ "$#" -gt 1 ]; then
      usage
      exit 1
    fi
    PROFILE="$1"
    ;;
esac

if [ -z "$PROFILE" ]; then
  echo "No profile specified and defaults.default_profile is empty."
  exit 1
fi

if ! profile_exists "$PROFILE"; then
  echo "Unknown model profile: $PROFILE"
  print_profiles
  exit 1
fi

MODEL="$(resolve_value model)"
if [ -z "$MODEL" ]; then
  echo "No model specified for profile: $PROFILE"
  exit 1
fi

if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
  echo "vLLM already running with PID $(cat $PIDFILE)"
  exit 1
fi

CUDA_DEVICES="$(resolve_value cuda_visible_devices)"
PORT="$(resolve_value port)"
MAX_NUM_SEQS="$(resolve_value max_num_seqs)"
MAX_MODEL_LEN="$(resolve_value max_model_len)"
GPU_MEMORY_UTILIZATION="$(resolve_value gpu_memory_utilization)"
ENABLE_PREFIX_CACHING="$(resolve_value enable_prefix_caching)"
ENABLE_AUTO_TOOL_CHOICE="$(resolve_value enable_auto_tool_choice)"
SERVED_MODEL_NAME="$(resolve_value served_model_name)"
REASONING_PARSER="$(resolve_value reasoning_parser)"
TOOL_CALL_PARSER="$(resolve_value tool_call_parser)"
SPECULATIVE_CONFIG="$(resolve_value speculative_config)"
CUDAGRAPH_MODE="$(resolve_value cudagraph_mode)"

CMD=(uv run vllm serve "$MODEL")

[ -n "$SPECULATIVE_CONFIG" ] && CMD+=(--speculative-config "$SPECULATIVE_CONFIG")
[ -n "$MAX_NUM_SEQS" ] && CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
[ -n "$MAX_MODEL_LEN" ] && CMD+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "$PORT" ] && CMD+=(--port "$PORT")
is_true "$ENABLE_PREFIX_CACHING" && CMD+=(--enable-prefix-caching)
[ -n "$GPU_MEMORY_UTILIZATION" ] && CMD+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
[ -n "$REASONING_PARSER" ] && CMD+=(--reasoning-parser "$REASONING_PARSER")
is_true "$ENABLE_AUTO_TOOL_CHOICE" && CMD+=(--enable-auto-tool-choice)
[ -n "$SERVED_MODEL_NAME" ] && CMD+=(--served-model-name "$SERVED_MODEL_NAME")
[ -n "$TOOL_CALL_PARSER" ] && CMD+=(--tool-call-parser "$TOOL_CALL_PARSER")
[ -n "$CUDAGRAPH_MODE" ] && CMD+=("-cc.cudagraph_mode=$CUDAGRAPH_MODE")

echo "Starting vLLM."
echo "Profile: $PROFILE"
echo "Model: $MODEL"
[ -n "$SERVED_MODEL_NAME" ] && echo "Served model name: $SERVED_MODEL_NAME"
[ -n "$CUDA_DEVICES" ] && echo "CUDA_VISIBLE_DEVICES: $CUDA_DEVICES"
[ -n "$PORT" ] && echo "Port: $PORT"
echo "Log: $LOG"

if [ -n "$CUDA_DEVICES" ]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" nohup "${CMD[@]}" >> "$LOG" 2>&1 &
else
  nohup "${CMD[@]}" >> "$LOG" 2>&1 &
fi

echo "$!" > "$PIDFILE"

echo "Started vLLM."
echo "PID: $(cat $PIDFILE)"
