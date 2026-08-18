"""Scale CALM-Net to all available subjects.

For each subject: build epochs (training task), pick early sessions for training
and the rest for longitudinal testing, then run EEGNet + EEG-Conformer (60 ch) and
the IMU-motion-only + EEGNet-motor-only confound controls. Aggregate to
results/multi_subject.json and print a cross-subject summary."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from dataio import build_epochs, list_sessions, session_days, DATA_ROOT
from experiments import run_longitudinal, imu_only_baseline, select_channels, MOTOR_CH

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_TRAIN = 3          # first N sessions used for training
EPOCHS = 80


def available_subjects():
    return sorted(p.name for p in DATA_ROOT.glob("sub-*") if p.is_dir())


def run_subject(subject):
    sess = list_sessions(subject)
    if len(sess) < N_TRAIN + 1:
        print(f"[{subject}] only {len(sess)} sessions, skipping"); return None
    es = build_epochs(subject=subject)
    days = session_days(subject)
    present = set(int(s) for s in np.unique(es.session))       # sessions that yielded epochs
    sess = [s for s in sess if s in present]
    if len(sess) < N_TRAIN + 1:
        print(f"[{subject}] only {len(sess)} usable sessions, skipping"); return None
    train = sess[:N_TRAIN]
    test = sess[N_TRAIN:]
    print(f"\n######## {subject}: {len(es)} epochs, train {train} test {test} ########", flush=True)

    out = {"subject": subject, "train_sessions": train, "test_sessions": test,
           "days": days, "n_epochs": len(es)}
    out["imu_only"] = imu_only_baseline(es, train, test)
    out["eegnet_full"] = run_longitudinal(es, "eegnet", train, test, epochs=EPOCHS,
                                          days=days, per_session_recal=True)
    out["conformer_full"] = run_longitudinal(es, "conformer", train, test, epochs=EPOCHS,
                                             days=days, per_session_recal=True)
    _, motor_idx = select_channels(es, MOTOR_CH)
    out["eegnet_motor"] = run_longitudinal(es, "eegnet", train, test, channels=motor_idx,
                                           epochs=EPOCHS, days=days,
                                           per_session_recal=False, tag="motor")
    return out


def _mean(sessions, field):
    return float(np.mean([s[field] for s in sessions.values()]))


if __name__ == "__main__":
    subjects = available_subjects()
    print("Subjects found:", subjects)
    results = {}
    for sub in subjects:
        r = run_subject(sub)
        if r:
            results[sub] = r
            (RESULTS / "multi_subject.json").write_text(json.dumps(results, indent=2))

    # ---- cross-subject summary ----
    print("\n\n================ CROSS-SUBJECT SUMMARY (mean over test sessions) ================")
    hdr = f"{'subj':6} {'EEGNet':>7} {'Conf':>7} {'Conf-AUROC':>11} {'IMU':>6} {'motor':>7} {'ConfECEraw':>11} {'ConfECEcal':>11}"
    print(hdr)
    agg = {k: [] for k in ["eegnet", "conf", "auroc", "imu", "motor", "ece_raw", "ece_cal"]}
    for sub, r in results.items():
        eeg = _mean(r["eegnet_full"]["sessions"], "bal_acc")
        conf = _mean(r["conformer_full"]["sessions"], "bal_acc")
        auroc = _mean(r["conformer_full"]["sessions"], "conf_auroc")
        imu = float(np.mean(list(r["imu_only"].values())))
        motor = _mean(r["eegnet_motor"]["sessions"], "bal_acc")
        er = _mean(r["conformer_full"]["sessions"], "ece_raw")
        ec = _mean(r["conformer_full"]["sessions"], "ece_cal")
        for k, v in zip(agg, [eeg, conf, auroc, imu, motor, er, ec]):
            agg[k].append(v)
        print(f"{sub:6} {eeg:7.3f} {conf:7.3f} {auroc:11.3f} {imu:6.3f} {motor:7.3f} {er:11.3f} {ec:11.3f}")
    print("-" * len(hdr))
    print(f"{'MEAN':6} {np.mean(agg['eegnet']):7.3f} {np.mean(agg['conf']):7.3f} "
          f"{np.mean(agg['auroc']):11.3f} {np.mean(agg['imu']):6.3f} {np.mean(agg['motor']):7.3f} "
          f"{np.mean(agg['ece_raw']):11.3f} {np.mean(agg['ece_cal']):11.3f}")
    (RESULTS / "multi_subject.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {RESULTS/'multi_subject.json'}")
