#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# Does slow adaptation, which recovered +0.097 on MoBI, also hold on ds007788?
#
# This is the test that decides whether momentum 0.01 is simply the correct
# default or a cohort-specific patch. ds007788's class blocks are short (median
# 18 windows), so its estimate was already class-mixed at momentum 0.2 and
# slowing further should be roughly neutral there. If alignment still helps on
# cohort A at 0.01 while costing less on cohort B, the layer has one setting that
# works on both, with the operating condition stated.
#
# Only dn_full and dn_noalign are needed: those two arms isolate the alignment
# contribution, which is the whole question.
while ! grep -q "MOMENTUM_TEST_DONE" logs/momentum.log 2>/dev/null; do sleep 60; done
export CX_MODEL=driftnet CX_EPOCHS=30 CX_SIZE=base CX_SEEDS=0
echo "=== ds007788 :: DN_MOMENTUM=0.01 (does slow adaptation hold on cohort A?) ==="
DN_MOMENTUM=0.01 CX_OUT=driftmom_ds_0.01.json python -u src/exp_calmnetx.py
echo SLOWMOM_DS_DONE
