#!/usr/bin/env bash
# E1: causal HMM forward filter over the decoder output, cohort A, seeds 0-2.
# Both transformer-free arms, so the filter's effect is measured on the two
# configurations the paper actually reports.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=ds007788
export CX_ARMS=dn_noctx,dn_stem CX_SEEDS=0,1,2
export CX_SMOOTH=1 CX_FULL=1 CX_ICA=0
export CX_OUT=a_smooth_3seed.json
python -u src/exp_calmnetx.py
echo "E1_DONE"; date
