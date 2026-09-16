r"""provision_session.py - take a fresh Modal workspace from empty to training-ready, unattended.

Runs the whole chain in order and does not stop for anything that can be retried:

    csvs -> archive (2 GiB parts) -> verify -> verify_deep (CRC sample) -> preprocess -> count shards

WHY A DRIVER AND NOT FIVE COMMANDS. The archive step is the one that historically goes wrong,
and it goes wrong in ways a human watching a terminal cannot fix: containers get preempted,
and the old single-file layout stalled at 0.2 MB/s with volume commits growing past five
minutes (see archive_parts.py). Everything that made session 102 painful is now either fixed
in the downloader or retried here:

  * bounded 2 GiB range requests, so Kaggle never stalls an open-ended multi-GiB response
  * one commit per part, so commit cost stays flat instead of growing with the archive
  * parts written .tmp and renamed, so a killed container cannot leave a short part
  * `modal run` is launched WITHOUT a timeout kill - killing the client cancels the app
  * each step retried, and the archive step resumes from whatever parts already landed

Progress is printed as single lines with timestamps so a background task log stays readable.

    python modal_scripts/provision_session.py 103
    python modal_scripts/provision_session.py 103 --skip-deep
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()


def log(msg):
    print("  %s  %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


def env_for(session):
    e = dict(os.environ)
    e["MODAL_CONFIG_PATH"] = str(HOME / (".modal_session%d.toml" % session))
    ca = HOME / ".certs" / "ca-bundle.pem"
    if ca.exists():
        e["REQUESTS_CA_BUNDLE"] = str(ca)
        e["SSL_CERT_FILE"] = str(ca)
    e["PYTHONIOENCODING"] = "utf-8"
    e.pop("MSYS_NO_PATHCONV", None)
    return e


def run(session, args, stream=False, timeout=None):
    """Run a modal command. `stream` echoes interesting lines as they arrive."""
    cmd = [sys.executable, "-u", "-m", "modal"] + list(args)
    if not stream:
        p = subprocess.run(cmd, env=env_for(session), cwd=str(ROOT),
                           capture_output=True, timeout=timeout)
        return p.returncode, ((p.stdout or b"") + (p.stderr or b"")).decode("utf-8", "replace")

    # streamed: never killed on a timeout - cancelling the client cancels the Modal app,
    # which is what turned one stalled download into 54 relaunches
    p = subprocess.Popen(cmd, env=env_for(session), cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, bufsize=1, text=True,
                         encoding="utf-8", errors="replace")
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-40]
        if any(k in line for k in ("[zip]", "[csv]", "[verify]", "[deep]", "[launch]",
                                   "Error", "Traceback", "error:")):
            if "GiB (" in line or "part " in line or line.startswith(("[csv]", "[verify]",
                                                                      "[deep]", "[launch]")) \
                    or "Error" in line or "Traceback" in line:
                log(line.strip())
    p.wait()
    return p.returncode, "\n".join(tail)


def step(session, label, args, attempts=3, stream=True):
    for i in range(1, attempts + 1):
        log("%s (attempt %d/%d)" % (label, i, attempts))
        rc, out = run(session, args, stream=stream)
        if rc == 0:
            log("%s OK" % label)
            return out
        log("%s failed rc=%d; last lines:" % (label, rc))
        for line in out.splitlines()[-6:]:
            log("    | %s" % line)
        if i < attempts:
            time.sleep(20)
    raise SystemExit("  %s FAILED after %d attempts - stopping" % (label, attempts))


def shard_count(session):
    rc, out = run(session, ["volume", "ls", "knee-shards", "--json"], timeout=600)
    try:
        return len(json.loads(out[out.index("["):out.rindex("]") + 1]))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--skip-deep", action="store_true",
                    help="skip the CRC sample (minutes, but it is the real integrity test)")
    a = ap.parse_args()
    s = a.session

    toml = HOME / (".modal_session%d.toml" % s)
    if not toml.exists():
        raise SystemExit("no %s - set the token first" % toml)

    t0 = time.time()
    rc, out = run(s, ["profile", "current"], timeout=180)
    log("session %d -> workspace %s" % (s, out.strip().splitlines()[-1] if rc == 0 else "?"))

    step(s, "1/5 csvs", ["run", "modal_scripts/modal_download.py", "--csvs-only"])
    step(s, "2/5 archive (2 GiB parts, resumable)",
         ["run", "--detach", "modal_scripts/modal_download.py", "--skip-csvs"], attempts=6)
    step(s, "3/5 verify (central directory)",
         ["run", "modal_scripts/modal_download.py::verify"])
    if not a.skip_deep:
        step(s, "4/5 verify_deep (CRC sample)",
             ["run", "modal_scripts/modal_download.py::verify_deep"])
    else:
        log("4/5 verify_deep SKIPPED")
    step(s, "5/5 preprocess -> shards", ["run", "--detach", "modal_scripts/modal_preprocess.py"],
         attempts=3)

    n = shard_count(s)
    mins = (time.time() - t0) / 60
    log("=" * 62)
    log("SESSION %d %s: %d shards in %.1f min"
        % (s, "READY" if n == 4407 else "INCOMPLETE (expected 4407)", n, mins))
    log("=" * 62)
    if n != 4407:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
