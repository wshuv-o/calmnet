#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# Wait for the base-size MoBI ablation to finish, then repeat at `small`.
# Size is chosen from FIT-SET SIZE, which is known before any test data is
# touched -- 1757 fit windows against 619k parameters is 352 params/sample,
# roughly 3x more data-starved than cohort A, and this project has measured
# repeatedly that models collapse in that regime. Selecting on fit-set size is
# an a priori rule; selecting on test accuracy would be leakage.
while ! grep -q "DN_MOBI_S0_DONE" logs/driftnight.log 2>/dev/null; do sleep 60; done
echo "=== MoBI at DriftNet-small (62 params/fit-window) ==="
CX_MODEL=driftnet CX_COHORT=mobi CX_SIZE=small CX_SEEDS=0 CX_EPOCHS=30 \
  CX_OUT=driftnet_mobi_small.json python -u src/exp_calmnetx.py
echo MOBI_SMALL_DONE
