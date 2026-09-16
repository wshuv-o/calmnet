#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# Wait for the in-flight ds007788 seed-0 ablation to finish.
while ! grep -q "ATCPLUS_S0_DONE" logs/atcplus.log 2>/dev/null; do sleep 60; done
echo "=== EXTERNAL VALIDATION: MoBI, seed 0 ==="
CX_COHORT=mobi CX_OUT=atcplus_mobi.json CX_SEEDS=0 CX_EPOCHS=30 python -u src/exp_calmnetx.py
echo MOBI_S0_DONE
echo "=== ds007788, seed 1 (replication) ==="
CX_SEEDS=1 CX_EPOCHS=30 python -u src/exp_calmnetx.py
echo DS_S1_DONE
echo "=== MoBI, seed 1 ==="
CX_COHORT=mobi CX_OUT=atcplus_mobi.json CX_SEEDS=1 CX_EPOCHS=30 python -u src/exp_calmnetx.py
echo MOBI_S1_DONE
echo "=== ds007788, seed 2 ==="
CX_SEEDS=2 CX_EPOCHS=30 python -u src/exp_calmnetx.py
echo ALLNIGHTDONE
