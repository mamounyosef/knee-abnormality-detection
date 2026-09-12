"""
wait_download.py - block until each workspace's archive download finishes.

Polls the archive size on the `knee-data` volume of every session given, and returns only
when all of them have stopped growing at (or near) the full size, or when one has clearly
stalled. The volume commits every 8 GiB, so the size moves in steps, not continuously:
"unchanged since the last poll" is NOT a stall, and the stall test has to be much longer
than one commit interval.

    python modal_scripts/wait_download.py 21 22
    python modal_scripts/wait_download.py 21 --poll 120 --stall-min 25
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ARCHIVE = "rsna-knee-abnormality-detection.zip"
FULL_GIB = 246.82


def _env(session: int) -> dict:
    e = dict(os.environ)
    e["MODAL_CONFIG_PATH"] = str(Path.home() / (".modal_session%d.toml" % session))
    e["PYTHONIOENCODING"] = "utf-8"
    e["REQUESTS_CA_BUNDLE"] = str(Path.home() / ".certs" / "ca-bundle.pem")
    e["SSL_CERT_FILE"] = e["REQUESTS_CA_BUNDLE"]
    # Git Bash rewrites anything that looks like a path; it mangles MODAL_CONFIG_PATH
    e.pop("MSYS_NO_PATHCONV", None)
    return e


def size_gib(session: int):
    """Archive size in GiB, or None when the call fails or the file is absent."""
    r = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "knee-data", "--json"],
                       env=_env(session), capture_output=True)
    out = r.stdout.decode("utf-8", "replace")
    try:
        start = out.index("[")
        rows = json.loads(out[start:out.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    for row in rows:
        if row.get("Filename") == ARCHIVE:
            v, unit = row["Size"].split()
            v = float(v)
            return v if unit == "GiB" else v / 1024 if unit == "MiB" else v / 1024**2
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", type=int)
    ap.add_argument("--poll", type=int, default=180, help="seconds between polls")
    ap.add_argument("--stall-min", type=int, default=25,
                    help="minutes without growth before calling it stalled; must be well "
                         "above one 8 GiB commit interval (~4 min) or a normal step looks "
                         "like a stall")
    a = ap.parse_args()

    last = {s: (None, time.time()) for s in a.sessions}
    done, stalled = set(), set()

    while len(done | stalled) < len(a.sessions):
        for s in a.sessions:
            if s in done or s in stalled:
                continue
            g = size_gib(s)
            prev, since = last[s]
            now = time.time()
            if g is None:
                print("[wait] session %d: could not read the volume" % s, flush=True)
                continue
            if prev is None or g > prev:
                last[s] = (g, now)
                print("[wait] session %2d  %7.2f GiB  (%.1f%%)"
                      % (s, g, 100 * g / FULL_GIB), flush=True)
            if g >= FULL_GIB - 0.5:
                done.add(s)
                print("[wait] session %d COMPLETE at %.2f GiB" % (s, g), flush=True)
            elif now - since > a.stall_min * 60:
                stalled.add(s)
                print("[wait] session %d STALLED at %.2f GiB, no growth for %d min"
                      % (s, g, a.stall_min), flush=True)
        if len(done | stalled) < len(a.sessions):
            time.sleep(a.poll)

    print("\n[wait] complete: %s   stalled: %s"
          % (sorted(done) or "none", sorted(stalled) or "none"))
    return 1 if stalled else 0


if __name__ == "__main__":
    sys.exit(main())
