#!/usr/bin/env bash
# Fourth cohort: BNCI2014-001. Control and branch in one file, 3 seeds.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=bnci
export CX_ARMS=dn_stem,dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_FULL=1 CX_ICA=0
export CX_OUT=d_bnci_3seed.json
python -u src/exp_calmnetx.py
echo "BNCI_DONE"; date
