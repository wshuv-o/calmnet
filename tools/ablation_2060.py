"""Cohort A ablation arms dn_stem and dn_noalign, seeds 1-2, on the RTX 2060.

Split of the 5080 queue's job 4 (A_ablation_s12): this machine takes dn_stem
and dn_noalign; the 5080 takes dn_full and dn_nogate. Settings identical to
job 4. Separate output file so the two machines never write the same JSON.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_queue import gpu, say  # noqa: E402

if __name__ == "__main__":
    gpu("A_ablation_s12_2060", CX_MODEL="driftnet", CX_ARMS="dn_stem,dn_noalign",
        CX_SEEDS="1,2", DN_MOMENTUM=0.2, CX_FULL=1, CX_ICA=0,
        CX_OUT="a_ablation_s12_2060.json")
    say("ABLATION 2060 DONE")
