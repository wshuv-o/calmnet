"""Queue for the second machine (RTX 5080), which holds cohort B's data.

Runs only what this machine cannot: cohort B under the current estimator, with
per-participant rows and three seeds. A cohort-A job runs last, only if cohort
A's data is present. Jobs whose data is missing are skipped, and jobs whose
result keys already exist resume where they stopped.

    git pull origin representation-beats-architecture
    python tools/remote_5080_queue.py

Progress: results/overnight_queue.log (job starts and ends) and
results/job_<label>.log (one line per participant).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_queue import ROOT, gpu, say  # noqa: E402

BASELINES = ["EEGNeX", "ShallowFBCSPNet", "EEGNet", "EEGConformer",
             "TSception", "FBLightConvNet", "Deep4Net", "EEGTCNet"]
HAS_B = (ROOT / "data" / "mobi_treadmill").is_dir()
HAS_A = (ROOT / "data" / "ds007788").is_dir()

JOBS = [
    # 1. Alignment on cohort B, paired over three seeds. The existing -0.211 is
    #    one seed, the old estimator and no per-participant rows.
    ("B_gate_aligngate_3seed", HAS_B, dict(
        CX_MODEL="driftnet", CX_COHORT="mobi", CX_ARMS="dn_gate,dn_noctx",
        CX_SEEDS="0,1,2", DN_MOMENTUM=0.2, CX_OUT="b_gate_aligngate_3seed.json")),
    # 2. Slowed adaptation on cohort B over three seeds: the +0.090 recovery.
    ("B_aligngate_m001_3seed", HAS_B, dict(
        CX_MODEL="driftnet", CX_COHORT="mobi", CX_ARMS="dn_noctx",
        CX_SEEDS="0,1,2", DN_MOMENTUM=0.01, CX_OUT="b_aligngate_m001_3seed.json")),
    # 3. Published decoders on cohort B, pipeline C, three seeds.
    ("B_baselines_3seed", HAS_B, dict(
        CX_MODEL="bd:" + "+".join(BASELINES), CX_COHORT="mobi",
        CX_SEEDS="0,1,2", CX_OUT="bd_cohort_b.json")),
    # 4. Cohort A ablation arms, seeds 1-2 (seed 0 exists). Only if A's data is here.
    ("A_ablation_s12", HAS_A, dict(
        CX_MODEL="driftnet", CX_ARMS="dn_full,dn_noalign,dn_nogate,dn_stem",
        CX_SEEDS="1,2", DN_MOMENTUM=0.2, CX_FULL=1, CX_ICA=0,
        CX_OUT="a_ablation_s12.json")),
]

if __name__ == "__main__":
    say("remote queue: cohort B data %s, cohort A data %s"
        % ("found" if HAS_B else "MISSING", "found" if HAS_A else "missing"))
    for label, ok, env in JOBS:
        if not ok:
            say("skip %s: data not on this machine" % label)
            continue
        gpu(label, **env)
    say("REMOTE QUEUE DONE")
