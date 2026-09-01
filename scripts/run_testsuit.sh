#!/usr/bin/env bash
# Runs the full automated test suite (unittest discovery) from the repo root.
#
# Usage:
#   scripts/run_testsuit.sh
#
# On Windows, the README's `py -m unittest -v` from the repo root is
# equivalent; this script is the POSIX/CI counterpart of that.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"

echo "Running full test suite: ${PYTHON} -m unittest discover -v"
exec "${PYTHON}" -m unittest discover -v
