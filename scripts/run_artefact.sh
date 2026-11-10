#!/usr/bin/env bash
# Artefact control on cohort A (Table 3): ICLabel artefact removal, both arms
# fitted on the training task only so they differ only in the cleaning step.
set -e
cd "$(dirname "$0")/.."
python train.py --config config/cohortA.yaml --arms stem_tangent --no-extra-trials       --out a_tan_icactl_orig.json
python train.py --config config/cohortA.yaml --arms stem_tangent --no-extra-trials --ica --out a_tan_icactl_ica.json
