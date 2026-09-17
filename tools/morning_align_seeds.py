"""Morning job: align-only arm on cohort A, seeds 1-2.

Paired with a_aligngate_3seed.json it measures what the selective head adds to
accuracy over three seeds, the evidence behind the decision to remove it.
Settings match the seed-0 align_only.json run exactly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_queue import gpu, say  # noqa: E402

if __name__ == "__main__":
    gpu("A_align_s12", CX_FULL=1, CX_ICA=0, CX_ARMS="dn_align", CX_SEEDS="1,2",
        DN_MOMENTUM=0.2, CX_OUT="a_align_s12.json")
    say("MORNING DONE")
