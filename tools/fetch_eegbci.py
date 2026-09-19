"""Fetch the cohort C runs from PhysioNet.

Cohort C lives on the other machine in this project's two-machine split, so the
Riemannian baseline could not be run here without the data. Only the six runs
the loader uses are fetched, at about 2.6 MB each, so roughly 310 MB for the
twenty participants rather than the whole database.

    python tools/fetch_eegbci.py
"""
import os
import warnings

warnings.filterwarnings("ignore")
import mne

mne.set_log_level("ERROR")
from mne.datasets import eegbci

RUNS = (3, 5, 7, 9, 11, 13)
total = 0
for s in range(1, 21):
    try:
        paths = eegbci.load_data(s, list(RUNS), path="data/eegbci",
                                 update_path=False, verbose=False)
        total += sum(os.path.getsize(p) for p in paths)
        print("  S%03d  %d runs" % (s, len(paths)), flush=True)
    except Exception as e:
        print("  S%03d FAILED %s: %s" % (s, type(e).__name__, str(e)[:80]),
              flush=True)
print("fetched %.0f MB" % (total / 1e6))
