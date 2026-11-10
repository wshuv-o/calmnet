#!/usr/bin/env bash
# Cost benchmark, statistics behind the tables, and the earlier pipeline (Tables 3-4).
set -e
cd "$(dirname "$0")/.."
python -m analysis.cost
python -m analysis.stats --write
cd analysis/earlier_pipeline
python exp_zscore.py
COHORT=mobi ZS_OUT=zscore_mobi.json python exp_zscore.py
python paired_zscore.py
python exp_published.py
SCHEMES=global PUB_OUT=amp_ablation.json AMP=on      python exp_published.py
SCHEMES=global PUB_OUT=amp_ablation.json AMP=shuffle python exp_published.py
