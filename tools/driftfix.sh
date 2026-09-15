#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_EPOCHS=30 CX_SIZE=base CX_SEEDS=0
# The fix targets the diagnosed cause: the running covariance estimate was
# power-weighted, so on MoBI (87.7% walk, walk windows carrying 1.52x the power)
# it sat at Riemannian distance 0.306 from the walk covariance -- it WAS the
# majority class. Per-window normalisation moves it to 3.899 / 3.194, between
# the classes. No labels involved.
#
# Cohort B runs first: it is where the layer failed, so it is where the repair
# has to show. Cohort A second, as a regression check that the fix does not
# break what already worked there.
echo "=== FIXED estimator :: Cohort B (MoBI), where alignment failed ==="
CX_COHORT=mobi CX_OUT=driftfix_mobi.json python -u src/exp_calmnetx.py
echo FIX_MOBI_DONE
echo "=== FIXED estimator :: Cohort A (regression check) ==="
CX_OUT=driftfix_ds.json python -u src/exp_calmnetx.py
echo FIX_DS_DONE
