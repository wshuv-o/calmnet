#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_EPOCHS=30 CX_SIZE=base

echo "=== DriftNet ablation :: Cohort A (ds007788), seed 0 ==="
CX_SEEDS=0 CX_OUT=driftnet_ds.json python -u src/exp_calmnetx.py
echo DN_DS_S0_DONE

echo "=== DriftNet ablation :: Cohort B (MoBI) EXTERNAL VALIDATION, seed 0 ==="
CX_COHORT=mobi CX_SEEDS=0 CX_OUT=driftnet_mobi.json python -u src/exp_calmnetx.py
echo DN_MOBI_S0_DONE

echo "=== DriftNet :: Cohort A, seed 1 ==="
CX_SEEDS=1 CX_OUT=driftnet_ds.json python -u src/exp_calmnetx.py
echo DN_DS_S1_DONE

echo "=== DriftNet :: Cohort B, seed 1 ==="
CX_COHORT=mobi CX_SEEDS=1 CX_OUT=driftnet_mobi.json python -u src/exp_calmnetx.py
echo DN_MOBI_S1_DONE

echo "=== DriftNet :: Cohort A, seed 2 ==="
CX_SEEDS=2 CX_OUT=driftnet_ds.json python -u src/exp_calmnetx.py
echo DN_DS_S2_DONE

echo "=== DriftNet :: Cohort B, seed 2 ==="
CX_COHORT=mobi CX_SEEDS=2 CX_OUT=driftnet_mobi.json python -u src/exp_calmnetx.py
echo DRIFTNIGHT_DONE
