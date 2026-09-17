"""Start tau_drift once the ICA precompute has finished and freed the CPUs.

tau_drift is CPU-only and loads one subject at a time, so it fits alongside the
GPU queue; it only has to avoid competing with the ICA workers for cores and
commit charge.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "ica_precompute.log"
end = time.time() + 90 * 60
while time.time() < end:
    try:
        if re.search(r"ICA PRECOMPUTE DONE", LOG.read_text(errors="replace")):
            break
    except FileNotFoundError:
        pass
    time.sleep(20)

env = dict(os.environ, TAU_COHORT="BOTH", PYTHONIOENCODING="utf-8")
with open(ROOT / "results" / "tau_drift_run.log", "w", encoding="utf-8") as out:
    out.write("started %s\n" % time.strftime("%H:%M:%S")); out.flush()
    rc = subprocess.call([sys.executable, "-u", str(ROOT / "src" / "exp_tau_drift.py")],
                         cwd=str(ROOT), env=env, stdout=out, stderr=subprocess.STDOUT)
    out.write("TAU DONE rc=%d %s\n" % (rc, time.strftime("%H:%M:%S")))
