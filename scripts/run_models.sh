#!/usr/bin/env bash
set -euo pipefail

# Model runner wrapper (keeps .sh, but real logic is in run_models.py)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src"

python "${REPO_ROOT}/scripts/run_models.py" "$@"
