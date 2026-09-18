#!/usr/bin/env bash
# Does the reference point explain the cohort B collapse?
#
# dn_stem_tan fell from 0.744 to 0.568 on cohort B with a running reference.
# If the cause is the base point tracking the streaming class, then removing
# its ability to track should recover it, and nothing else about the branch
# changes. Cohort B is where the failure is and it is the fast cohort, so the
# test runs where it is cheapest to be wrong.
cd "$(dirname "$0")/.."
while pgrep -f "CX_OUT=b_tangent_3seed" >/dev/null 2>&1 || \
      grep -q "COHORT_B_TANGENT_DONE" logs/cohortB_tangent.log 2>/dev/null && false; do sleep 20; done
until grep -q "COHORT_B_TANGENT_DONE" logs/cohortB_tangent.log 2>/dev/null; do sleep 20; done

export CX_MODEL=driftnet CX_COHORT=mobi CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_SHRINK=0.1 CX_FULL=1 CX_ICA=0
export CX_ARMS=dn_stem_tan

for mode in frozen none; do
  export DN_TAN_REF=$mode
  export CX_OUT=b_tangent_ref_${mode}.json
  echo "=== cohort B, reference mode: $mode ==="
  python -u src/exp_calmnetx.py
done
echo "REF_MODES_DONE"; date
