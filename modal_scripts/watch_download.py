"""
watch_download.py - keep a workspace's archive download alive until it finishes.

WHY THIS EXISTS. modal_download.py already retries: `retries=10`, and every retry resumes by
HTTP Range from the committed size so nothing is re-fetched. That covers a failing FUNCTION.
It does NOT cover the failure we actually keep hitting, which is Modal cancelling the whole
ephemeral APP:

    "Received a cancellation signal while processing input" -> app state `stopped`

When the app stops, there is no function left to retry. The job simply ends, the archive sits
at whatever size it reached, an orphaned .download.lock is left behind, and nothing restarts
it. Session 23 died that way twice, at 112 GiB and at 152 GiB, and each death cost however
long it took a human to notice. That dead time - not bandwidth - is why these downloads feel
slow: the transfer itself runs near 90 MB/s.

So this polls, and relaunches when the app is gone and the archive is not finished.

RELAUNCHING IS SAFE. modal_download.py refuses to start a second writer while a lock is held
AND the archive grew recently (STALE_AFTER_S), and it refuses to touch an archive that
already has its .download.done sentinel. So a spurious relaunch is rejected by the
downloader itself rather than corrupting anything.

    python modal_scripts/watch_download.py 21
    python modal_scripts/watch_download.py 21 23 --poll 300 --hours 12
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# line_buffering: without it Python buffers stdout when redirected to a file, and a
# supervisor you cannot observe is a supervisor you cannot trust - the first run of this
# sat at 0 bytes for 40 minutes while a download it was meant to watch stayed dead.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = "rsna-knee-abnormality-detection.zip"
DONE = ".download.done"
FULL_GIB = 246.82


def env_for(session: int) -> dict:
    e = dict(os.environ)
    e["MODAL_CONFIG_PATH"] = str(Path.home() / (".modal_session%d.toml" % session))
    e["REQUESTS_CA_BUNDLE"] = str(Path.home() / ".certs" / "ca-bundle.pem")
    e["SSL_CERT_FILE"] = e["REQUESTS_CA_BUNDLE"]
    e["PYTHONIOENCODING"] = "utf-8"
    e.pop("MSYS_NO_PATHCONV", None)          # Git Bash mangles MODAL_CONFIG_PATH otherwise
    return e


def modal(session: int, *args, timeout=300):
    try:
        p = subprocess.run([sys.executable, "-m", "modal"] + list(args),
                           env=env_for(session), cwd=str(ROOT),
                           capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ""
    return ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")


def volume_state(session: int):
    """(gib_or_None, done_sentinel_present). None size means the call failed."""
    out = modal(session, "volume", "ls", "knee-data", "--json")
    try:
        rows = json.loads(out[out.index("["):out.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None, False
    names = {r.get("Filename"): r for r in rows}
    gib = None
    z = names.get(ARCHIVE)
    if z:
        raw = str(z.get("Size", ""))
        try:                                   # "152.0 GiB" or a raw byte count
            gib = (float(raw.split()[0]) if "GiB" in raw
                   else float(raw) / 1024 ** 3)
        except (ValueError, IndexError):
            gib = None
    return gib, DONE in names


def download_app_live(session: int) -> bool:
    """Is a download actually RUNNING, not merely listed?

    An app can sit in the list as `ephemeral` with ZERO tasks: the container died, the app
    record lingers, and nothing is downloading. Session 23 sat exactly like that at 216.3 GiB
    for over an hour while this returned True and the archive never moved. So parse the task
    COUNT out of the row and require it to be non-zero.

    The row looks like:  | ap-xxxx | knee-download | ephemeral | 1 | 2026-09-14 ... |
    """
    out = modal(session, "app", "list")
    for line in out.splitlines():
        if "knee-down" not in line:
            continue
        cells = [c.strip() for c in line.replace("│", "|").split("|")]
        tasks = 0
        for c in cells:
            if c.isdigit():                    # the task count is the only bare integer
                tasks = int(c)
                break
        state_ok = any("ephemeral" in c or "running" in c.lower() for c in cells)
        if state_ok and tasks > 0:
            return True
    return False


def relaunch(session: int) -> bool:
    """Start a detached download. Returns True if the launch command was accepted.

    A TIMEOUT HERE IS NOT A FAILURE. `modal run --detach` submits the job and then keeps
    printing until the client detaches, so a slow image build makes it exceed any timeout we
    set while the job is already running happily. The first version reported "relaunch
    FAILED" for session 21 at the exact 600s mark, and the archive grew from 136 to 152 GiB
    regardless. So say UNCONFIRMED rather than FAILED, and let the next poll settle it by
    looking at the app list, which is the real evidence either way.
    """
    out = modal(session, "run", "--detach", "modal_scripts/modal_download.py",
                "--skip-csvs", timeout=420)
    if "View run at" in out or "Initialized" in out:
        print("      relaunch submitted")
        return True
    if not out:                                    # our timeout fired, not modal's
        print("      relaunch UNCONFIRMED (client still printing at timeout); "
              "next poll will confirm from the app list")
        return True
    print("      relaunch FAILED:\n%s" % out[-500:])
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", type=int)
    ap.add_argument("--poll", type=int, default=300, help="seconds between checks")
    ap.add_argument("--hours", type=float, default=12.0)
    a = ap.parse_args()

    pending = list(a.sessions)
    relaunches = {s: 0 for s in pending}
    last_size, flat = {}, {s: 0 for s in pending}
    deadline = time.time() + a.hours * 3600
    print("watching %s  (poll %ds, giving up after %.1fh)"
          % (", ".join(map(str, pending)), a.poll, a.hours))

    while pending and time.time() < deadline:
        for s in list(pending):
            gib, done = volume_state(s)
            stamp = datetime.now().strftime("%H:%M:%S")
            size = "?" if gib is None else "%.1f GiB (%.0f%%)" % (gib, 100 * gib / FULL_GIB)
            if done:
                print("  %s  session %-3d %-20s DONE" % (stamp, s, size))
                pending.remove(s)
                continue
            live = download_app_live(s)
            # A SECOND liveness test: the archive must actually GROW. The volume commits
            # every 8 GiB, so one flat poll means nothing, but three in a row (15 min) with
            # the app claiming to be alive means the container is wedged.
            prev = last_size.get(s)
            flat[s] = flat[s] + 1 if (prev is not None and gib == prev) else 0
            last_size[s] = gib
            stalled = flat[s] >= 3
            print("  %s  session %-3d %-20s app %s%s"
                  % (stamp, s, size, "live" if live else "GONE",
                     "  STALLED (%d flat polls)" % flat[s] if stalled else ""))
            if not live or stalled:
                flat[s] = 0
                # the app died and nothing retries an app-level cancellation, so restart it.
                # modal_download.py rejects the launch itself if a real writer is running.
                relaunches[s] += 1
                print("      not downloading (app %s, %s) -> relaunch #%d"
                      % ("gone" if not live else "0 tasks or wedged",
                         "archive flat" if stalled else "unfinished", relaunches[s]))
                relaunch(s)
        if pending:
            time.sleep(a.poll)

    for s in pending:
        print("STILL UNFINISHED: session %d (%d relaunches)" % (s, relaunches[s]))
    for s in a.sessions:
        if s not in pending:
            print("finished: session %d after %d relaunch(es)" % (s, relaunches[s]))
    return 0 if not pending else 1


if __name__ == "__main__":
    raise SystemExit(main())
