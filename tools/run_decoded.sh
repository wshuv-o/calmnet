#!/usr/bin/env bash
# Cohort E: DECODED exoskeleton. tau_blk=13 < N=32, so the SAFE regime, and the
# registered prediction is that the branch works at m=0.2. Nothing tuned.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=decoded
export CX_ARMS=dn_stem,dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_FULL=1 CX_ICA=0
export CX_OUT=e_decoded_3seed.json
python -u src/exp_calmnetx.py
echo "DECODED_DONE"; date
