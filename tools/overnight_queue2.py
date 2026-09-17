"""Second overnight queue: multi-seed, per-subject versions of the core comparison.

Starts when the first queue logs QUEUE DONE. Runs gate against align+gate
(alignment the single differing factor, the design the third-cohort
pre-registration uses) over three seeds with per-participant rows on cohorts A
and B. All three cohorts then share one design with paired tests, and the
headline arm stops being single-seed.

The align+gate arm has no transformer and is reported bit-identical across
repeats, so its seed-0 rerun should reproduce the existing 0.884 exactly: a
reproducibility check at no extra cost.

No job starts after the cutoff, so the morning's integration always has time.
Jobs are in priority order, most valuable first.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_queue import RES, gpu, say, wait_line  # noqa: E402

CUTOFF = (7, 0)          # no new job starts at or after 07:00

JOBS = [
    # Cohort B's data is not on this machine (data/mobi_treadmill and
    # data/cache_mobi are absent; its results were produced elsewhere), so the
    # planned cohort-B job would fail. It is replaced by a seed replication of
    # the pre-registered cohort-C test, which is fast and runs on local data.
    # The registered verdict stays seed 0; these seeds are reported separately.

    # 1. Cohort C replication, seeds 1-2. Small data, so it goes first.
    ("C_rep_m0.2", dict(CX_COHORT="eegbci", CX_ARMS="dn_gate,dn_noctx",
                        CX_SEEDS="1,2", DN_MOMENTUM=0.2,
                        CX_OUT="cohort3_rep_m0.2.json")),
    ("C_rep_m0.01", dict(CX_COHORT="eegbci", CX_ARMS="dn_noctx",
                         CX_SEEDS="1,2", DN_MOMENTUM=0.01,
                         CX_OUT="cohort3_rep_m0.01.json")),
    # 2. Headline multi-seed on cohort A, with per-subject rows.
    ("A_aligngate_3seed", dict(CX_FULL=1, CX_ICA=0, CX_ARMS="dn_noctx",
                               CX_SEEDS="0,1,2", DN_MOMENTUM=0.2,
                               CX_OUT="a_aligngate_3seed.json")),
    # 3. The no-alignment reference on cohort A, completing the paired design.
    ("A_gate_3seed", dict(CX_FULL=1, CX_ICA=0, CX_ARMS="dn_gate",
                          CX_SEEDS="0,1,2", DN_MOMENTUM=0.2,
                          CX_OUT="a_gate_3seed.json")),
]


def past_cutoff():
    n = datetime.now()
    return (n.hour, n.minute) >= CUTOFF and n.hour < 12


def main():
    say("queue2 started, waiting for queue 1")
    wait_line(RES / "overnight_queue.log", r"QUEUE DONE", 400)
    for label, env in JOBS:
        if past_cutoff():
            say("queue2 skip %s: past %02d:%02d cutoff" % ((label,) + CUTOFF))
            continue
        gpu(label, **env)
    say("QUEUE2 DONE")


if __name__ == "__main__":
    main()
