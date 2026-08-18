"""Download NeuroRex (OpenNeuro ds007788) subject subsets.

Usage:
    python src/download.py                       # all sessions of sub-01
    python src/download.py sub-01/ses-01         # one session
    python src/download.py sub-01 sub-02         # two subjects
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import openneuro
from pathlib import Path

DATASET = "ds007788"
TAG = "1.0.1"
target = Path("./data/ds007788")
target.mkdir(parents=True, exist_ok=True)

include = sys.argv[1:] or ["sub-01"]
print(f"Downloading {include} of {DATASET} (v{TAG}) into {target.resolve()} ...", flush=True)

openneuro.download(
    dataset=DATASET,
    tag=TAG,
    target_dir=target,
    include=include,
    max_concurrent_downloads=8,
)

print("\nDone. File tree:", flush=True)
total = 0
for p in sorted(target.rglob("*")):
    if p.is_file():
        mb = p.stat().st_size / 1e6
        total += mb
        print(f"  {p.relative_to(target)}  ({mb:.2f} MB)")
print(f"\nTotal: {total:.1f} MB")
