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
    # 1. Headline multi-seed on cohort A, with per-subject rows.
    ("A_aligngate_3seed", dict(CX_FULL=1, CX_ICA=0, CX_ARMS="dn_noctx",
                               CX_SEEDS="0,1,2", DN_MOMENTUM=0.2,
                               CX_OUT="a_aligngate_3seed.json")),
    # 2. The inversion on cohort B, same factor isolation, three seeds.
    ("B_gate_aligngate_3seed", dict(CX_COHORT="mobi", CX_ARMS="dn_gate,dn_noctx",
                                    CX_SEEDS="0,1,2", DN_MOMENTUM=0.2,
                                    CX_OUT="b_gate_aligngate_3seed.json")),
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
