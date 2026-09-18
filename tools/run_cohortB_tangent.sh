#!/usr/bin/env bash
# CONFIRMATION RUN, cohort B. Shrinkage frozen at 0.1, the untuned default,
# chosen before the sweep and unchanged by it. Nothing here is tuned.
#
# Two module ablations, each a control and that control plus the branch, in one
# file so every comparison is within-file and matched:
#   dn_stem  vs dn_stem_tan     (mirrors cohort A)
#   dn_gate  vs dn_gate_tan     (dn_gate 0.7289 is this cohort's reference)
# Alignment is off in all four, which is what the rate condition requires here.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=mobi
export CX_ARMS=dn_stem,dn_stem_tan,dn_gate,dn_gate_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_SHRINK=0.1 CX_FULL=1 CX_ICA=0
export CX_OUT=b_tangent_3seed.json
python -u src/exp_calmnetx.py
echo "COHORT_B_TANGENT_DONE"; date
