"""Run our CALMNet across all subjects and merge into results/multi_subject.json."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, session_days
from experiments import run_longitudinal

RESULTS = Path(__file__).resolve().parent.parent / "results"
R = json.loads((RESULTS / "multi_subject.json").read_text())

for sub in sorted(R):
    r = R[sub]
    es = build_epochs(subject=sub)          # cached -> instant
    print(f"\n######## {sub}: CALMNet ########", flush=True)
    r["calmnet_full"] = run_longitudinal(
        es, "calmnet", r["train_sessions"], r["test_sessions"],
        epochs=100, days=session_days(sub), per_session_recal=True)
    (RESULTS / "multi_subject.json").write_text(json.dumps(R, indent=2))

# summary
def m(s, f):
    return float(np.mean([x[f] for x in s.values()]))

print("\n\n=========== CALMNet vs baselines (mean over test sessions) ===========")
print(f"{'subj':6}{'CALMNet':>8}{'Conf':>7}{'EEGNet':>8}{'CALM-AUROC':>11}{'Conf-AUROC':>11}{'CALM-ECEraw':>12}")
acc, ca, co = [], [], []
for sub, r in R.items():
    cm = m(r["calmnet_full"]["sessions"], "bal_acc")
    cf = m(r["conformer_full"]["sessions"], "bal_acc")
    en = m(r["eegnet_full"]["sessions"], "bal_acc")
    cau = m(r["calmnet_full"]["sessions"], "conf_auroc")
    fau = m(r["conformer_full"]["sessions"], "conf_auroc")
    ce = m(r["calmnet_full"]["sessions"], "ece_raw")
    acc.append(cm); ca.append(cau); co.append(ce)
    print(f"{sub:6}{cm:8.3f}{cf:7.3f}{en:8.3f}{cau:11.3f}{fau:11.3f}{ce:12.3f}")
print("-" * 63)
print(f"{'MEAN':6}{np.mean(acc):8.3f}{np.mean([m(R[s]['conformer_full']['sessions'],'bal_acc') for s in R]):7.3f}"
      f"{np.mean([m(R[s]['eegnet_full']['sessions'],'bal_acc') for s in R]):8.3f}"
      f"{np.mean(ca):11.3f}{np.mean([m(R[s]['conformer_full']['sessions'],'conf_auroc') for s in R]):11.3f}"
      f"{np.mean(co):12.3f}")
