#!/usr/bin/env bash
# Cohort B at the DERIVED rate. m* = 32/(2*214) = 0.0748, fixed by
# paper/PREREGISTRATION_derived_rate.md before this ran. Control dn_stem 0.7436.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=mobi
export CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.0748 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_FULL=1 CX_ICA=0
export CX_OUT=b_tangent_derived_3seed.json
python -u src/exp_calmnetx.py
echo "B_DERIVED_DONE"; date
