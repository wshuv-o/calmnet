#!/usr/bin/env bash
cd "$(dirname "$0")/.."
set -u
# Locate the floor of the rate condition -- the one claim that survives.
#
# Cohort B has tau_blk = 309 windows. The existing sweep measures either side
# of it (mu = 160, 640, 3200) but never AT it. Momentum 0.10 gives mu = 320,
# which sits on the threshold; 0.02 gives mu = 1600, filling the gap between
# 640 and 3200. Three seeds each, so the step can be tested rather than eyeballed.
export CX_MODEL=driftnet CX_COHORT=mobi CX_EPOCHS=30 CX_SIZE=base
export CX_ARMS=dn_noctx,dn_noalign
for M in 0.10 0.02; do
  echo "=== cohort B :: momentum $M (memory $(python -c "print(int(32/$M))") windows) ==="
  date
  CX_SEEDS=0,1,2 DN_MOMENTUM=$M CX_OUT=b_floor_m${M}.json python -u src/exp_calmnetx.py
done
echo FLOOR_LOCATE_DONE
date
