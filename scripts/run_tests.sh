#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${CAPSULE_TEST_PYTHON:-$PROJECT_DIR/.venv/bin/python}

if [[ ! -x $PYTHON_BIN ]]; then
  echo "Test interpreter not found: $PYTHON_BIN" >&2
  echo "Create .venv from requirements.txt or set CAPSULE_TEST_PYTHON." >&2
  exit 1
fi

# Some CUDA PyTorch wheels keep cuDSS beside the wheel without an ELF rpath to
# that directory. Add only the matching missing library directory, avoiding a
# broad mix of CUDA-version directories from the environment.
CUDSS_PATH=$(find "$PROJECT_DIR/.venv" -type f -name 'libcudss.so.0' -print -quit 2>/dev/null || true)
if [[ -n $CUDSS_PATH ]]; then
  CUDSS_LIB_DIR=$(dirname "$CUDSS_PATH")
  export LD_LIBRARY_PATH="$CUDSS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/capsule-matplotlib}
export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-/tmp/capsule-ultralytics}

cd "$PROJECT_DIR"
"$PYTHON_BIN" scripts/verify_dependency_versions.py requirements.shared.txt
exec "$PYTHON_BIN" -m unittest discover -s tests -v
