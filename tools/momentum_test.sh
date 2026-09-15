#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export CX_MODEL=driftnet CX_EPOCHS=30 CX_SIZE=base CX_SEEDS=0 CX_COHORT=mobi
# Decisive test of the block-length diagnosis.
#
# ds007788: 109 class blocks, median 18 windows -- SHORTER than the layer's
#           adaptation memory, so the running estimate stays class-mixed.
# MoBI:       7 class blocks, median 309, max 2211 windows -- far LONGER, so at
#           momentum 0.2 (memory ~5 batches, ~160 windows) the estimate becomes
#           whichever class is currently streaming, and the layer whitens the
#           class away.
#
# If that is the cause, slowing adaptation until its memory exceeds a class block
# should restore MoBI. momentum 0.01 gives memory ~100 batches (~3200 windows),
# longer than the longest block, which is effectively classical Euclidean
# Alignment estimated over a whole recording.
#
# Note momentum 0.2 was chosen because it MAXIMISED drift reduction (60% vs 25%).
# If slow momentum decodes better while removing less drift, that is direct
# evidence that the mechanism metric and the task metric point in opposite
# directions here.
for m in 0.01 0.05; do
  echo "=== MoBI :: DN_MOMENTUM=$m ==="
  DN_MOMENTUM=$m CX_OUT=driftmom_${m}.json python -u src/exp_calmnetx.py
done
echo MOMENTUM_TEST_DONE
