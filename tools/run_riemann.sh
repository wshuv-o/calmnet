#!/usr/bin/env bash
# Classical Riemannian baselines in the reporting harness, parallel over cells.
#
# The Karcher mean is the expensive step, about 48 s on cohort A's fitting
# split, and it is not safe to approximate: fitting it on a 1000-covariance
# subsample moved one participant's accuracy by 0.012 and a 500-subsample by
# +0.014 in the other direction, which is the size of the effects this paper
# interprets. So the mean is computed in full and the work is spread over
# workers instead, each pinned to 2 BLAS threads.
cd "$(dirname "$0")/.."
export CX_SEEDS=0,1,2 CX_WORKERS=8 CX_THREADS=2

export CX_COHORT=ds007788 CX_OUT=a_riemann.json
python -u src/exp_riemann_baseline.py
echo "=== COHORT A DONE ==="

export CX_COHORT=eegbci CX_OUT=c_riemann.json
python -u src/exp_riemann_baseline.py
echo "RIEMANN_DONE"; date
