#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export CX_MODEL=atcplus CX_EPOCHS=30 CX_SEEDS=0 CX_ARMS=atc+ours
echo "=== cohort A :: stock ATCNet + ctx + gate vs stock ATCNet 0.9129 ==="
CX_OUT=atcours_ds.json python -u src/exp_calmnetx.py
echo ATCOURS_DS_DONE
