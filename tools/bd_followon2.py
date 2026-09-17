"""Replacement for bd_followon.py with EEGNeX moved last for cohort A seeds 1-2.

Waits for the cohort C job already running (its PID is passed in) to exit, then
resumes cohort C (completed keys are skipped, so this only fills any gap) and
runs cohort A seeds 1-2 with the slowest model, EEGNeX, at the end.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_queue import gpu, say  # noqa: E402

C_ORDER = ["EEGNeX", "ShallowFBCSPNet", "EEGNet", "EEGConformer",
           "TSception", "FBLightConvNet", "Deep4Net", "EEGTCNet"]
A_ORDER = ["ShallowFBCSPNet", "EEGNet", "EEGConformer", "TSception",
           "FBLightConvNet", "Deep4Net", "EEGTCNet", "EEGNeX"]


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid], capture_output=True, text=True).stdout
    return str(pid) in out


if __name__ == "__main__":
    pid = int(sys.argv[1])
    say("bd follow-on 2: waiting for running cohort C job (pid %d)" % pid)
    while alive(pid):
        time.sleep(20)
    say("END   bd_cohortC  (pid %d exited)" % pid)
    gpu("bd_cohortC_resume", CX_COHORT="eegbci", CX_MODEL="bd:" + "+".join(C_ORDER),
        CX_SEEDS="0,1,2", CX_OUT="bd_cohort_c.json")
    gpu("bd_pipeC_s12", CX_MODEL="bd:" + "+".join(A_ORDER), CX_FULL=1, CX_ICA=0,
        CX_SEEDS="1,2", CX_OUT="bd_pipeline_c.json")
    say("BD FOLLOW-ON DONE")
