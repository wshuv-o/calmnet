#!/usr/bin/env bash
cd "$(dirname "$0")/.."
python -m pip install --quiet mne_icalabel onnxruntime python-picard 2>&1 | tail -2
python -c "import mne_icalabel, picard; print('deps ok')" || exit 1
export CX_MODEL=driftnet CX_COHORT=ds007788
export CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1
export CX_FULL=1 CX_ICA=1
export CX_OUT=a_tangent_ica_3seed.json
python -u src/exp_calmnetx.py
echo "ICA_BRANCH_DONE"; date
