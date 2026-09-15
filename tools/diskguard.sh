#!/usr/bin/env bash
# Kill the named python job if free space on D: drops below the threshold.
# np.savez does not write atomically, so a disk that fills mid-write leaves
# truncated .npz caches that later fail with BadZipFile -- which is how seven
# trial caches were silently corrupted. Stopping the writer early is cheaper
# than finding the corruption afterwards.
PATTERN="${1:-exp_globalnorm}"; MIN_MB="${2:-1500}"
while true; do
  free_mb=$(df -m /d | tail -1 | awk '{print $4}')
  if [ "$free_mb" -lt "$MIN_MB" ]; then
    echo "DISKGUARD: only ${free_mb}MB free (<${MIN_MB}MB) - stopping $PATTERN"
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | Where-Object { \$_.CommandLine -like '*$PATTERN*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
    exit 1
  fi
  sleep 20
done
