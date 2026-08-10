#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"
GCPO_TOPK="${GCPO_TOPK:-16}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
GCPO_BASIS_PATH="${GCPO_BASIS_PATH:-${PROJECT_ROOT}/artifacts/gcpo_basis/${MODEL_TAG}/k${GCPO_TOPK}/basis.pt}"

mkdir -p "$(dirname "${GCPO_BASIS_PATH}")"

python "${PROJECT_ROOT}/scripts/prepare_gcpo_basis.py" \
  --model-path "${MODEL_PATH}" \
  --output "${GCPO_BASIS_PATH}" \
  --topk "${GCPO_TOPK}" \
  --target-modules "${TARGET_MODULES}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

echo "GCPO basis written to ${GCPO_BASIS_PATH}"
