#!/usr/bin/env bash
# 1. dn_stem seed 0, so the control has the same three seeds as the branch and
#    the comparison stops being unmatched. No tuning, just rigour.
# 2. Shrinkage sweep on dn_stem_tan. sub-05 loses 0.113 with the branch and is
#    the same participant alignment hurt, so its covariance is the suspect.
#    Shrinkage was fixed at 0.1 without ever being varied.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=ds007788 DN_MOMENTUM=0.2 CX_FULL=1 CX_ICA=0

export CX_ARMS=dn_stem CX_SEEDS=0 CX_OUT=a_ablation_s12.json
python -u src/exp_calmnetx.py
echo "=== dn_stem seed 0 done, control now matched ==="

for sh in 0.3 0.5; do
  export DN_TAN_SHRINK=$sh
  export CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
  export CX_OUT=a_tangent_shrink${sh}.json
  echo "=== shrink $sh ==="
  python -u src/exp_calmnetx.py
done
echo "TANGENT2_DONE"; date
