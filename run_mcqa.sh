#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to run this script." >&2
  exit 1
fi

echo "Running mcqa_run.py in the torch-metal environment from $ROOT_DIR"
conda run -n torch-metal python mcqa_run.py "$@"
