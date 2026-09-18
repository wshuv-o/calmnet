#!/usr/bin/env bash
# Cohort A with the reference removed. The +0.0452 on this cohort was measured
# with a running reference, which cohort B then showed to be the component that
# fails, so that number is not a result for this variant and A has to be rerun
# under it. Control dn_stem is already at 3 seeds in a_ablation_s12.json.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=ds007788
export CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_SHRINK=0.1 DN_TAN_REF=none
export CX_FULL=1 CX_ICA=0
export CX_OUT=a_tangent_ref_none.json
python -u src/exp_calmnetx.py
echo "A_REFNONE_DONE"; date
