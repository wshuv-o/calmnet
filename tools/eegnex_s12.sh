#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# EEGNeX seeds 1 and 2, pipeline C, cohort A, on the 5080.
# Writes its OWN file so bd_pipeline_c.json (owned by the other machine)
# is never touched; merge downstream.
export CX_MODEL="bd:EEGNeX" CX_FULL=1 CX_ICA=0 CX_SEEDS="1,2"
export CX_OUT=bd_eegnex_s12.json
echo "=== EEGNeX seeds 1,2 :: pipeline C, cohort A ==="; date
python -u src/exp_calmnetx.py
echo EEGNEX_S12_DONE; date
