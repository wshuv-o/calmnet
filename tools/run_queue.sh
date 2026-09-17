#!/usr/bin/env bash
cd "$(dirname "$0")/.."
echo "=== HEAD-TO-HEAD: our model vs published, ONE harness, full data ==="
for sd in 0 1; do
  SEED=$sd FULL_DATA=1 SCHEMES=global PUB_OUT=fullbench.json MODELS=CALMNet-bare \
    python -u src/exp_published.py 4.0
done
echo H2HDONE
echo "=== ATCNet+ module ablation, seed 0 ==="
CX_SEEDS=0 CX_EPOCHS=30 python -u src/exp_calmnetx.py
echo ATCPLUS_S0_DONE
echo "=== ATCNet rate mechanism control ==="
for sd in 0 1 2; do
  SEED=$sd FULL_DATA=1 SCHEMES=global PUB_OUT=atcrate.json \
    MODELS=ATC-default,ATC-rate,ATC-rate_nw3 python -u src/exp_published.py 4.0
done
echo RATEDONE
