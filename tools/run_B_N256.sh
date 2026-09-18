#!/usr/bin/env bash
# Cohort B at N=256. The other half of tab:batchsweep; the 2060 has cohort A at
# N=8. Cohort B's blocks are 214 windows, so at the default N=32 an update sits
# inside a block and the branch collapsed (-0.175). At N=256 an update spans a
# whole block, and the boundary predicts the collapse should disappear.
# Momentum is held at 0.2 so the memory in UPDATES is unchanged and only the
# windows-per-update moves.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=mobi
export CX_ARMS=dn_stem,dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_BATCH=256 CX_FULL=1 CX_ICA=0
export CX_OUT=b_batch256_3seed.json
python -u src/exp_calmnetx.py
echo "B_N256_DONE"; date
