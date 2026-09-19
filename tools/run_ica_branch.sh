#!/usr/bin/env bash
# Artefact control for the REPORTED model. The existing control was run on a
# configuration that is no longer in the paper, so there is currently no
# evidence that the branch's gain survives ICA removal of muscle and ocular
# components. Same arm, same cohort, same seeds, ICA on.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=ds007788
export CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_FULL=1 CX_ICA=1
export CX_OUT=a_tangent_ica_3seed.json
python -u src/exp_calmnetx.py
echo "ICA_BRANCH_DONE"; date
