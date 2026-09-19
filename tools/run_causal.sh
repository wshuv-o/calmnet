#!/usr/bin/env bash
# Look-ahead control. The reported stem+branch runs update the running
# reference with a batch before mapping that batch, so a test window's
# reference includes up to 31 later windows. DN_TAN_CAUSAL=1 maps each batch at
# the reference from before it and updates afterwards. Everything else matches
# the reported runs (momentum 0.2, shrinkage 0.1, running reference, batch 32,
# seeds 0-2), so each file pairs against the existing stem control:
#   C  c_stem_tangent_3seed.json  dn_stem
#   D  d_bnci_3seed.json          dn_stem
#   A  a_ablation_s12.json        dn_stem   (seeds 0-2)
# Order C, D, A: shortest first, for an early read.
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_ARMS=dn_stem_tan CX_SEEDS=0,1,2
export DN_MOMENTUM=0.2 DN_TAN_REF=running DN_TAN_SHRINK=0.1 DN_TAN_CAUSAL=1
export CX_FULL=1 CX_ICA=0

export CX_COHORT=eegbci CX_OUT=c_tangent_causal_3seed.json
python -u src/exp_calmnetx.py
echo "=== COHORT C CAUSAL DONE ==="; date

export CX_COHORT=bnci CX_OUT=d_tangent_causal_3seed.json
python -u src/exp_calmnetx.py
echo "=== COHORT D CAUSAL DONE ==="; date

export CX_COHORT=ds007788 CX_OUT=a_tangent_causal_3seed.json
python -u src/exp_calmnetx.py
echo "CAUSAL_DONE"; date
