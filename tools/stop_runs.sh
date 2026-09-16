#!/usr/bin/env bash
# Targeted stop: only python jobs running the named script, and only the
# overnight wrapper. Deliberately does NOT pattern-match on generic strings --
# a broad pattern previously matched the shell executing the kill itself.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | Where-Object { \$_.CommandLine -like '*exp_calmnetx*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like '%bash%'\" | Where-Object { \$_.CommandLine -like '*overnight.sh*' -or \$_.CommandLine -like '*run_queue.sh*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
exit 0
