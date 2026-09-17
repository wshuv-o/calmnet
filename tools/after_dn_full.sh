#!/usr/bin/env bash
cd "$(dirname "$0")/.."
# Wait for dn_full seeds 1 and 2 to be written, then stop the queue and restart
# on dn_nogate and dn_stem only. dn_noalign is being run on the RTX 2060, so
# this machine must not duplicate it.
#
# exp_calmnetx writes per completed arm-seed and skips keys already present, so
# stopping between arms loses nothing.
echo "waiting for dn_full|s1 and dn_full|s2 ..."
while true; do
  n=$(python -c "
import json,io,os
p='results/a_ablation_s12.json'
d=json.load(io.open(p,encoding='utf-8')) if os.path.exists(p) else {}
print(sum(1 for k in d if k.startswith('dn_full|')))" 2>/dev/null || echo 0)
  [ "$n" -ge 2 ] && break
  sleep 60
done
echo "dn_full done; stopping the queue"
pkill -f remote_5080_queue 2>/dev/null
powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force" 2>/dev/null
sleep 5
echo "restarting on dn_nogate,dn_stem seeds 1,2"
export CX_MODEL=driftnet CX_ARMS=dn_nogate,dn_stem CX_SEEDS=1,2
export DN_MOMENTUM=0.2 CX_FULL=1 CX_ICA=0 CX_OUT=a_ablation_s12.json
date; python -u src/exp_calmnetx.py; echo A_ABLATION_5080_DONE; date
