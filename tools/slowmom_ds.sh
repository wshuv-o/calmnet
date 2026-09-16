#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# Does slow adaptation, which recovered +0.09 to +0.10 on every alignment-using
# arm of cohort B, also hold on cohort A?
#
# This decides whether momentum 0.01 is simply the correct default or a
# cohort-B-specific patch. Cohort A's class blocks are short (median 18 windows),
# far below even the fastest setting's ~160-window memory, so its estimate was
# already class-mixed at momentum 0.2 and slowing further is predicted to be
# roughly neutral. If instead cohort A degrades, there is no single setting for
# both cohorts and the design rule needs qualifying.
#
# Only dn_full and dn_noalign are run: those two isolate the alignment
# contribution, which is the entire question, and the other three arms are
# already measured at the default momentum.
export CX_MODEL=driftnet CX_EPOCHS=30 CX_SIZE=base CX_SEEDS=0
export CX_ARMS=dn_full,dn_noalign
echo "=== ds007788 :: DN_MOMENTUM=0.01 (does slow adaptation hold on cohort A?) ==="
DN_MOMENTUM=0.01 CX_OUT=driftmom_ds_0.01.json python -u src/exp_calmnetx.py
echo SLOWMOM_DS_DONE
