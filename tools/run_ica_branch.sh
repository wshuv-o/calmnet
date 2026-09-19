#!/usr/bin/env bash
# Artefact control for the REPORTED model. The control in the manuscript was run
# on dn_noctx, a configuration that is no longer in the paper, so there is
# currently no evidence that the branch's accuracy survives ICA removal of
# muscle and ocular components.
#
# CX_FULL=0 restricts both arms to the training task, as the original control
# did, so the two differ only in the cleaning step. It also matches the cache
# key of the already-cleaned epochs (..._training_..._ica.npz); CX_FULL=1 asks
# for a subset those caches do not cover, which forces ICLabel to re-run and
# fail on this montage.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_COHORT=ds007788
export CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_FULL=0

export CX_ICA=0 CX_OUT=a_tan_icactl_orig.json
python -u src/exp_calmnetx.py
echo "=== UNCLEANED ARM DONE ==="

export CX_ICA=1 CX_OUT=a_tan_icactl_ica.json
python -u src/exp_calmnetx.py
echo "ICA_BRANCH_DONE"; date
