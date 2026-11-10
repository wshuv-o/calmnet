#!/usr/bin/env bash
# Causal processing (Tables 6 and 9, Table S6, Figures 8-9).
#   causal update order: each batch mapped at the reference from before it
#   fully causal: causal update + forward-only filtering + hold resampling
set -e
cd "$(dirname "$0")/.."
for c in a b c d e; do
  cfg=config/cohort${c^^}.yaml
  python train.py --config $cfg --arms stem_tangent --causal-update --out ${c}_tangent_causal_3seed.json
  FC="--causal-filter --causal-update"
  python train.py --config $cfg $FC --out ${c}_fullcausal_3seed.json
  python train.py --config $cfg $FC --smooth --out ${c}_fullcausal_smooth.json
  python train.py --config $cfg $FC --arms stem_tangent --test-init fitmean    --out ${c}_fullcausal_init_fitmean.json
  python train.py --config $cfg $FC --arms stem_tangent --test-init firstbatch --out ${c}_fullcausal_init_firstbatch.json
  # per-window posteriors, for the AUC and the matched false-activation analysis
  python train.py --config $cfg $FC --dump --out ${c}_fullcausal_dump.json
done
