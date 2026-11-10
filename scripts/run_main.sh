#!/usr/bin/env bash
# Five-cohort comparison, stem against stem + branch (Table 5), 3 seeds.
set -e
cd "$(dirname "$0")/.."
python train.py --config config/cohortA.yaml --arms stem         --out a_ablation_s12.json
python train.py --config config/cohortA.yaml --arms stem_tangent --out a_tangent_3seed.json
python train.py --config config/cohortB.yaml --out b_tangent_3seed.json
python train.py --config config/cohortC.yaml --out c_stem_tangent_3seed.json
python train.py --config config/cohortD.yaml --out d_bnci_3seed.json
python train.py --config config/cohortE.yaml --out e_decoded_3seed.json
