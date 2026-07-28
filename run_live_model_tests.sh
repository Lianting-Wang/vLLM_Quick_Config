#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env ]]; then
  echo "Missing $SCRIPT_DIR/.env" >&2
  exit 1
fi

if [[ ! -f proxy_config.json ]]; then
  echo "Missing $SCRIPT_DIR/proxy_config.json" >&2
  exit 1
fi

cat <<'EOF'
WARNING: this is a live integration test.
It will send real inference requests, sleep selected vLLM instances,
wake them again, and temporarily lower idle timeouts before restoring them.
Do not run it while other users or applications are actively using the models.
EOF

if [[ "${VLLM_LIVE_TEST_CONFIRM:-}" != "YES" ]]; then
  if [[ ! -t 0 ]]; then
    echo "Non-interactive execution requires VLLM_LIVE_TEST_CONFIRM=YES" >&2
    exit 2
  fi
  read -r -p "Type LIVE to continue: " confirmation
  if [[ "$confirmation" != "LIVE" ]]; then
    echo "Cancelled."
    exit 2
  fi
fi

exec uv run python tools/live_model_sleep_wake_test.py --confirm-live "$@"
