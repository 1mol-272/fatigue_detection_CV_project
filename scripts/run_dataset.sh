#!/usr/bin/env bash
set -euo pipefail

# This script makes "src/" importable for scripts/run_all.py.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src"

python "${REPO_ROOT}/scripts/run_all.py" "$@"
