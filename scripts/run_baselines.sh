#!/usr/bin/env bash
# Eight published decoders in the same pipeline (Table 11) and the capacity
# control (Table 12: the stem widened to the reported model's size).
set -e
cd "$(dirname "$0")/.."
BD=$(printf 'bd:%s,' EEGNeX ShallowFBCSPNet EEGNet EEGConformer TSception FBLightConvNet Deep4Net EEGTCNet)
BD=${BD%,}
python train.py --config config/cohortA.yaml --arms $BD --out bd_pipeline_c.json
python train.py --config config/cohortB.yaml --arms $BD --out bd_cohort_b.json
python train.py --config config/cohortC.yaml --arms $BD --out bd_cohort_c.json
python train.py --config config/cohortD.yaml --arms $BD --out bd_cohort_d.json
python train.py --config config/cohortE.yaml --arms $BD --out bd_cohort_e.json
python train.py --config config/cohortA.yaml --arms stem,stem_tangent,stem_wide --out capacity_ds007788.json
