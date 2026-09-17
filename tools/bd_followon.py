"""Follow-on baseline queue, started after tools/bd_pipeline_c.py finishes.

1. Cohort C (EEGMMIDB, 20 participants): every baseline, seeds 0-2, pipeline C.
   Our align + gate and gate arms already exist there over the same seeds
   (cohort3_m0.2.json, cohort3_rep_m0.2.json). Fast, so it runs first.
2. Cohort A: seeds 1-2 for every baseline, so each matches our three-seed
   align + gate result. Closest competitors first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_queue import LOG, gpu, say, wait_line  # noqa: E402

MODELS = ["EEGNeX", "ShallowFBCSPNet", "EEGNet", "EEGConformer",
          "TSception", "FBLightConvNet", "Deep4Net", "EEGTCNet"]

if __name__ == "__main__":
    say("bd follow-on waiting for the pipeline C queue")
    wait_line(LOG, r"BD PIPELINE C DONE", 600)
    gpu("bd_cohortC", CX_COHORT="eegbci", CX_MODEL="bd:" + "+".join(MODELS),
        CX_SEEDS="0,1,2", CX_OUT="bd_cohort_c.json")
    gpu("bd_pipeC_s12", CX_MODEL="bd:" + "+".join(MODELS), CX_FULL=1, CX_ICA=0,
        CX_SEEDS="1,2", CX_OUT="bd_pipeline_c.json")
    say("BD FOLLOW-ON DONE")
