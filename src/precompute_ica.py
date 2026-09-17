"""Build the ICA-cleaned training-task caches for cohort A, in parallel.

ICA is the slow step and runs on CPU, so it is done here, ahead of and apart
from the GPU arm that consumes the caches. Each worker handles one subject and
is pinned to a small thread count: picard's BLAS calls would otherwise
oversubscribe the CPUs across workers.

Usage: python src/precompute_ica.py [n_workers]
"""
import os
import sys
import time
from multiprocessing import Pool

N_THREADS = "2"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, N_THREADS)

SUBJECTS = ["sub-0%d" % i for i in range(1, 8)]


def one(sub):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import warnings
    warnings.filterwarnings("ignore")
    from dataio import build_epochs
    t = time.time()
    try:
        es = build_epochs(subject=sub, win=4.0, step=0.5, zscore=False, ica=True)
        return "%s ok  %d windows  %.0fs" % (sub, len(es.y), time.time() - t)
    except Exception as e:
        return "%s FAILED %s: %s" % (sub, type(e).__name__, str(e)[:120])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    t0 = time.time()
    print("ICA precompute: %d subjects, %d workers" % (len(SUBJECTS), n), flush=True)
    with Pool(n) as pool:
        for msg in pool.imap_unordered(one, SUBJECTS):
            print("  " + msg, flush=True)
    print("ICA PRECOMPUTE DONE in %.0fs" % (time.time() - t0), flush=True)
