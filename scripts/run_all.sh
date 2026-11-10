#!/usr/bin/env bash
# Every experiment in the paper, then the tables and figures.
set -e
cd "$(dirname "$0")"
for s in run_main run_reference run_artefact run_memory_sweep run_causal run_baselines; do
  bash $s.sh
done
cd ..
python make_tables.py
python make_figures.py
