#!/usr/bin/env bash
# The rate condition applied to the tangent branch, on cohort B.
# tau_blk there is 309 windows. At m=0.2 the memory is 32/0.2 = 160 windows,
# shorter than a class block, so the running reference converges on the
# streaming class and the branch collapsed (-0.175). At m=0.01 the memory is
# 3200 windows, an order of magnitude longer than a block, so the rule predicts
# the tracking stops. Control dn_stem carries no running statistic at all, so
# momentum does not touch it and the existing 0.7436 stands as the control.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=mobi
export CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.01 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_FULL=1 CX_ICA=0
export CX_OUT=b_tangent_m001_3seed.json
python -u src/exp_calmnetx.py
echo "B_SLOW_DONE"; date
