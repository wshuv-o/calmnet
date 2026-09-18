#!/usr/bin/env bash
# Tangent-space branch, cohort A, seeds 0-2. Each arm is its control plus one
# module: dn_tan against dn_noctx (0.8682), dn_stem_tan against dn_stem.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=ds007788
export CX_ARMS=dn_stem_tan,dn_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 CX_FULL=1 CX_ICA=0
export CX_OUT=a_tangent_3seed.json
python -u src/exp_calmnetx.py
echo "TANGENT_DONE"; date
