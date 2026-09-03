#!/usr/bin/env bash
set -euo pipefail

# Controlled x/v diagnostic. Pair manifests must already exist.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/launch/diffsbdd_global_e2po_good_bad_alidiff_t500.yaml"
PAIR_DIR="${PROJECT_ROOT}/analysis/diffsbdd_global_e2po_good_bad_alidiff_260825/pairs"
PAIR_COUNT="${PAIR_COUNT:-4096}"
EPOCHS="${EPOCHS:-1}"

if [[ ! -f "${PAIR_DIR}/train_pairs.csv" || ! -f "${PAIR_DIR}/validation_pairs.csv" ]]; then
  echo "Missing good/bad pair manifests under ${PAIR_DIR}" >&2
  exit 2
fi

for variant in x_only v_only native_joint; do
  output_dir="${PROJECT_ROOT}/analysis/diffsbdd_global_e2po_good_bad_xv_smoke_260825/${variant}"
  conda run -n dif python "${PROJECT_ROOT}/ddpo/main_diffsbdd_global_dpo.py" \
    --config "${CONFIG}" \
    --variant "${variant}" \
    --pair-count "${PAIR_COUNT}" \
    --allow-small-pairs \
    --epochs "${EPOCHS}" \
    --disable-generation \
    --output-dir "${output_dir}"
done
