"""heartbeat.py - print the REAL state of a long local job every N minutes, for a Monitor.

    python modal_scripts/heartbeat.py --match 19_ens_member_preds --glob "data/19_member_preds/*.npz" \
        --log <file> --every 600

Each beat prints: whether a python process whose command line contains --match is alive, how many
files match --glob, and the last meaningful line of --log. If the process is gone, it prints a
STOPPED line and exits, so the watcher itself ends and wakes whoever is waiting.
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import psutil

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def alive(match):
    import os
    for p in psutil.process_iter(["cmdline"]):
        try:
            cl = p.info["cmdline"] or []
            if p.pid == os.getpid() or any("heartbeat.py" in c for c in cl):
                continue                          # never count the watcher itself as the job
            if any(match in c for c in cl):
                return p.pid
        except Exception:                                            # noqa: BLE001
            continue
    return None


def last_line(log):
    if not log or not Path(log).exists():
        return ""
    txt = Path(log).read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    lines = [x.strip() for x in txt.splitlines() if x.strip()]
    return lines[-1][:220] if lines else ""


ap = argparse.ArgumentParser()
ap.add_argument("--match", required=True)
ap.add_argument("--glob", default="")
ap.add_argument("--log", default="")
ap.add_argument("--every", type=int, default=600)
a = ap.parse_args()
root = Path(__file__).resolve().parent.parent
while True:
    try:
        pid = alive(a.match)
        n = len(glob.glob(str(root / a.glob))) if a.glob else 0
        stamp = time.strftime("%H:%M")
        if pid is None:
            print(f"[{stamp}] STOPPED: no process matching '{a.match}'; files {n}; last: {last_line(a.log)}", flush=True)
            break
        print(f"[{stamp}] alive (pid {pid}); files {n}; last: {last_line(a.log)}", flush=True)
    except Exception as e:                                               # noqa: BLE001
        print(f"[{time.strftime('%H:%M')}] heartbeat error (still watching): {type(e).__name__}: {e}", flush=True)
    time.sleep(a.every)
