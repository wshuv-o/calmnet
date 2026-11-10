#!/usr/bin/env bash
# Where the tangent reference comes from (Table 4): running, frozen, none.
set -e
cd "$(dirname "$0")/.."
python train.py --config config/cohortA.yaml --arms stem_tangent --ref-mode frozen --out a_tangent_frozen_3seed.json
python train.py --config config/cohortA.yaml --arms stem_tangent --ref-mode none   --out a_tangent_ref_none.json
python train.py --config config/cohortB.yaml --arms stem_tangent --ref-mode frozen --out b_tangent_ref_frozen.json
python train.py --config config/cohortB.yaml --arms stem_tangent --ref-mode none   --out b_tangent_ref_none.json
python train.py --config config/cohortC.yaml --ref-mode none --out c_tangent_noref_3seed.json
