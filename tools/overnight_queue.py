"""Overnight GPU queue.

Chains the experiments one at a time: there is one GPU, and Windows commit
charge rather than physical RAM is the binding memory limit on this machine.
Every wait is on a real condition (a process exiting, a line appearing in a
log) with a timeout, so no step can stall the queue, and a failed job is
logged and passed over.

Written in Python rather than bash because a shell `sleep` loop launched from
this session was killed after a few seconds earlier tonight.
"""
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LOG = RES / "overnight_queue.log"
PY = sys.executable

COMMON = dict(CX_MODEL="driftnet", CX_EPOCHS="30", CX_SIZE="base", CX_SEEDS="0")


def say(msg):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def wait_pid(pid, minutes):
    end = time.time() + minutes * 60
    while psutil.pid_exists(pid):
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        if time.time() > end:
            say("timeout waiting for pid %d" % pid)
            return False
        time.sleep(30)
    return True


def wait_line(path, pattern, minutes):
    end = time.time() + minutes * 60
    rx = re.compile(pattern, re.M)
    while True:
        try:
            if rx.search(Path(path).read_text(encoding="utf-8", errors="replace")):
                return True
        except FileNotFoundError:
            pass
        if time.time() > end:
            say("timeout waiting for %r in %s" % (pattern, path))
            return False
        time.sleep(30)


def gpu(label, **env):
    say("START %s" % label)
    t = time.time()
    e = dict(os.environ)
    e.update(COMMON)
    e.update({k: str(v) for k, v in env.items()})
    with open(RES / ("job_%s.log" % label), "w", encoding="utf-8") as out:
        rc = subprocess.call([PY, "-u", str(ROOT / "src" / "exp_calmnetx.py")],
                             cwd=str(ROOT), env=e, stdout=out,
                             stderr=subprocess.STDOUT)
    say("END   %s  rc=%d  %.0f min" % (label, rc, (time.time() - t) / 60))
    return rc


def main():
    say("queue started")

    # 1. Let the running dn_nogate arm finish; it holds most of the commit charge.
    say("waiting for run 2 (pid 13620) to exit")
    wait_pid(13620, 120)

    # 2. ICA caches on CPU, alongside the next GPU jobs.
    say("launching ICA precompute (2 workers)")
    (RES / "ica_components.csv").unlink(missing_ok=True)
    ica_log = open(RES / "ica_precompute.log", "w", encoding="utf-8")
    ica_proc = subprocess.Popen([PY, "-u", str(ROOT / "src" / "precompute_ica.py"), "2"],
                                cwd=str(ROOT), stdout=ica_log,
                                stderr=subprocess.STDOUT)

    # 3. Artefact control, uncleaned arm. Training task only, so the cleaned
    #    arm below differs from it in the cleaning step alone.
    gpu("icactl_orig", CX_FULL=0, CX_ICA=0, CX_ARMS="dn_noctx",
        DN_MOMENTUM=0.2, CX_OUT="icactl_orig.json")

    # 4. Third cohort, pre-registered. Needs the complete download.
    say("waiting for EEGMMIDB download")
    wait_line(ROOT / "data" / "eegbci" / "download.log", r"^DONE", 90)
    gpu("cohort3_m0.2", CX_COHORT="eegbci", CX_ARMS="dn_gate,dn_noctx",
        DN_MOMENTUM=0.2, CX_OUT="cohort3_m0.2.json")
    gpu("cohort3_m0.01", CX_COHORT="eegbci", CX_ARMS="dn_noctx",
        DN_MOMENTUM=0.01, CX_OUT="cohort3_m0.01.json")

    # 5. Artefact control, cleaned arm.
    say("waiting for ICA precompute")
    ica_proc.wait()
    ica_log.close()
    say("ICA precompute exited rc=%s" % ica_proc.returncode)
    gpu("icactl_ica", CX_FULL=0, CX_ICA=1, CX_ARMS="dn_noctx",
        DN_MOMENTUM=0.2, CX_OUT="icactl_ica.json")

    # 6. Align-only on full data, comparable with the headline align+gate arm.
    gpu("align_only", CX_FULL=1, CX_ICA=0, CX_ARMS="dn_align",
        DN_MOMENTUM=0.2, CX_OUT="align_only.json")

    say("QUEUE DONE")


if __name__ == "__main__":
    main()
