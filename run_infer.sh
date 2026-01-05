#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${SM_CODE_DIR:-/opt/ml/processing/input/code}"
MODEL_DIR="${SM_MODEL_DIR:-/opt/ml/processing/input/model}"

sm_pip_bootstrap() {
  python -m pip install --upgrade pip >/dev/null 2>&1 || true
}

sm_install_requirements() {
  # Default: do NOT install extra deps (the SageMaker image already includes torch/pandas/scipy).
  # Opt-in by setting INSTALL_REQUIREMENTS=1.
  if [[ "${INSTALL_REQUIREMENTS:-0}" == "1" ]]; then
    python -m pip install --no-cache-dir -r "${CODE_DIR}/requirements.txt"
  fi
}

mkdir -p "${MODEL_DIR}"

sm_pip_bootstrap
sm_install_requirements

exec python "${CODE_DIR}/infer.py" "$@"
