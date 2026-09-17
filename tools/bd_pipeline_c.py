"""Published decoders in pipeline C, seed 0, cohort A.

Each braindecode model runs under the wrapper and training loop the stock-ATCNet
`base` arm used (0.913), so all are comparable with it and with DriftNet.
Ordered by pipeline-F standing, then the three-participant screen.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_queue import gpu, say  # noqa: E402

MODELS = ["EEGNeX", "ShallowFBCSPNet", "EEGNet", "EEGConformer",
          "TSception", "FBLightConvNet", "Deep4Net", "EEGTCNet"]

if __name__ == "__main__":
    gpu("bd_pipeC", CX_MODEL="bd:" + "+".join(MODELS), CX_FULL=1, CX_ICA=0,
        CX_SEEDS="0", CX_OUT="bd_pipeline_c.json")
    say("BD PIPELINE C DONE")
