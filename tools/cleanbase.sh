#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# The three-point momentum sweep mixed estimator versions: its m=0.20 column came
# from runs finished before the per-window covariance fix landed (05:16), while
# m=0.05 and m=0.01 came after. For dn_noalign and dn_stem that cannot matter --
# alignment is off, so the estimator is never called, and dn_stem reproduces
# 0.737 exactly at every setting. For dn_noctx and dn_nogate it is a real
# confound worth up to 0.017 on cohort A.
#
# Re-measure those two arms at m=0.20 with the current (fixed) estimator so every
# cell of the table comes from one code version.
export CX_MODEL=driftnet CX_EPOCHS=30 CX_SIZE=base CX_SEEDS=0 CX_COHORT=mobi
export CX_ARMS=dn_noctx,dn_nogate
echo "=== MoBI :: m=0.20, FIXED estimator (cleans the sweep baseline) ==="
DN_MOMENTUM=0.2 CX_OUT=driftfix_mobi_m020.json python -u src/exp_calmnetx.py
echo CLEANBASE_DONE
