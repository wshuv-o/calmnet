#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# Run A: the backbone screening sweep on all seven participants instead of
# three, so the Screen column of tab:landscape becomes a full-cohort number
# and the "three participants cannot rank these architectures" caveat is
# retired. ATCNet is included this time (it was skipped in the original run),
# which also gives it a value in the same column as the other 19.
#
# Cost basis: the 3-participant sweep was 715 s for 18 models and trains one
# model per participant, so 7 participants is roughly 1670 s.
export SB_OUT=backbone_full.json
export SB_SUBJECTS=sub-01,sub-02,sub-03,sub-04,sub-05,sub-06,sub-07
export SB_SKIP=
echo "=== screening sweep, full cohort (7 participants) ==="
date
python -u src/select_backbone.py
echo SCREEN_FULL_DONE
date
