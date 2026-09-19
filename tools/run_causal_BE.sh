#!/usr/bin/env bash
# Look-ahead control, cohorts B and E (the 5080 holds their data). Same as
# tools/run_causal.sh: only the stem+branch arm, DN_TAN_CAUSAL=1, every other
# setting as in the reported runs, so each file pairs against the existing stem
# control in b_tangent_3seed.json and e_decoded_3seed.json.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1 DN_TAN_CAUSAL=1
export CX_FULL=1 CX_ICA=0

export CX_COHORT=decoded CX_OUT=e_tangent_causal_3seed.json
python -u src/exp_calmnetx.py
echo "=== COHORT E CAUSAL DONE ==="; date

export CX_COHORT=mobi CX_OUT=b_tangent_causal_3seed.json
python -u src/exp_calmnetx.py
echo "CAUSAL_BE_DONE"; date
