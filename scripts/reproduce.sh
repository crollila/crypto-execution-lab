#!/usr/bin/env bash
# One-command reproduction: install, test, regenerate every result + figure offline.
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -e ".[dev]"
pytest -q
python -m execlab reproduce "$@"
