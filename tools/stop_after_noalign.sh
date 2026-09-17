#!/usr/bin/env bash
# Stop the 2060 ablation job once dn_noalign seed 2 is saved, so dn_stem (moved
# to the RTX 5080) is never started here. Nothing is lost: results are written
# per completed arm-seed, and dn_stem has not begun.
cd "$(dirname "$0")/.."
until python -c "import json,sys; d=json.load(open('results/a_ablation_s12_2060.json')); sys.exit(0 if 'dn_noalign|s2' in d else 1)" 2>/dev/null; do sleep 30; done
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'exp_calmnetx|ablation_2060' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
echo "[$(date +%H:%M:%S)] stopped 2060 ablation after dn_noalign seed 2; dn_stem runs on the 5080" | tee -a results/overnight_queue.log
