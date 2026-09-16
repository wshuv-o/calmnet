#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# The cohort A comparison against published decoders rests on one seed, and the
# baselines' own per-seed spread reaches 0.041. Measure the best arm at seed 1
# so the comparison is at least two-on-two rather than one-on-two.
export CX_MODEL=driftnet CX_EPOCHS=30 CX_SIZE=base CX_SEEDS=1
export CX_ARMS=dn_noctx,dn_full
echo "=== ds007788 :: seed 1, fixed estimator, m=0.20 ==="
DN_MOMENTUM=0.2 CX_OUT=driftseed1_ds.json python -u src/exp_calmnetx.py
echo SEED1_DONE
