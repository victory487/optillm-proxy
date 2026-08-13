#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-local"

resolve_python() {
  if [[ -n "${BON_PROXY_PYTHON:-}" && -x "${BON_PROXY_PYTHON}" ]]; then
    echo "${BON_PROXY_PYTHON}"
    return 0
  fi
  # Prefer the repo-root uv venv when present (same machine layout as llm_team).
  if [[ -x "/kwkj-k8s/llm_team/.venv/bin/python3.12" ]]; then
    echo "/kwkj-k8s/llm_team/.venv/bin/python3.12"
    return 0
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    command -v python3.12
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  echo "Error: need python3.12 (or set BON_PROXY_PYTHON)" >&2
  return 1
}

# Recreate when missing or when the interpreter symlink is broken.
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
  rm -rf "$VENV"
  "$(resolve_python)" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! "$VENV/bin/python" -c 'import bon_proxy' >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip install -e '.[dev]'
fi

CONFIG="${1:-$ROOT/config.sglang.yaml}"
exec "$VENV/bin/bon-proxy" --config "$CONFIG"
