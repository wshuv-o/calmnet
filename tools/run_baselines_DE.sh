#!/usr/bin/env bash
# The eight published decoders on the two cohorts added last, three seeds each.
# Same wrapper, splits, seeds and metric code as cohorts A, B and C, so the
# comparison is within-harness. Fills the 16 red cells of tab:newbase.
cd "$(dirname "$0")/.."
M="bd:EEGNeX+ShallowFBCSPNet+EEGNet+EEGConformer+TSception+FBLightConvNet+Deep4Net+EEGTCNet"
export CX_MODEL="$M" CX_SEEDS=0,1,2 CX_FULL=1 CX_ICA=0

export CX_COHORT=bnci   CX_OUT=bd_cohort_d.json
echo "=== cohort D (BCI IV-2a) ==="
python -u src/exp_calmnetx.py

export CX_COHORT=decoded CX_OUT=bd_cohort_e.json
echo "=== cohort E (DECODED exoskeleton) ==="
python -u src/exp_calmnetx.py

echo "BASELINES_DE_DONE"; date
