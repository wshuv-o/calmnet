# Earlier training pipeline

Code behind Table 4 (where amplitude normalisation is applied, six published decoders on
cohort A) and the representation block of Table 3 (band power, tangent space and correlation
features, per-window z-scoring on or off, cohorts A and B). The paper labels these as an
earlier training pipeline; the files are kept as they were run, with only the data and
results paths changed. Data root: `$CALMNET_DATA` or `../../datasets`.

```bash
cd analysis/earlier_pipeline
python exp_zscore.py                                          # results/zscore.json
COHORT=mobi ZS_OUT=zscore_mobi.json python exp_zscore.py      # results/zscore_mobi.json
python paired_zscore.py                                       # paired tests for Table 3
python exp_published.py                                       # results/published.json
SCHEMES=global PUB_OUT=amp_ablation.json AMP=on      python exp_published.py
SCHEMES=global PUB_OUT=amp_ablation.json AMP=shuffle python exp_published.py
```
