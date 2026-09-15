"""Pre-build epoch caches at the window lengths the project never varied.

Every cache on disk is w2.0_st0.5. Window length is the one representation knob
that was flagged as untested in features.py and never run. Step is held at 0.5 s
for every length so the window STREAM rate is identical across conditions -- the
temporal-context experiment needs a constant sampling rate to compare dwell
priors fairly, and holding step fixed also keeps the epoch count comparable.
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
from dataio import build_epochs

WINS = [float(w) for w in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["1.0", "3.0", "4.0"])]
SUBJECTS = [f"sub-0{i}" for i in range(1, 8)]

for w in WINS:
    for sub in SUBJECTS:
        t0 = time.time()
        try:
            es = build_epochs(subject=sub, win=w, step=0.5)
            print(f"[ok] {sub} win={w} n={len(es)} ch={es.X.shape[1]} T={es.X.shape[2]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[FAIL] {sub} win={w}: {type(e).__name__}: {e}", flush=True)
print("DONE", flush=True)
