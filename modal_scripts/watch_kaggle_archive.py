r"""watch_kaggle_archive.py - wait for Kaggle to rebuild the bulk archive, then provision.

The host re-uploaded the competition files on 2026-09-17 22:14 UTC. The CONTENT is unchanged
(all 4,407 studies, all 24,371 series, all 4,407 reports byte-identical to what we already
hold), but the re-upload invalidated Kaggle's prepared archive zip, so:

    GET /competitions/data/download-all/rsna-knee-abnormality-detection
    404 {"code":404,"message":"no gcs url"}

while the same endpoint serves titanic and digit-recognizer with 206. Nothing on our side can
fix that - Kaggle rebuilds these on their own schedule. So poll, and the moment it serves
bytes, run the normal provisioning chain unattended.

    python modal_scripts/watch_kaggle_archive.py 104
    python modal_scripts/watch_kaggle_archive.py 104 --every 1800 --max-hours 48
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
COMP = "rsna-knee-abnormality-detection"
URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/%s" % COMP


def log(msg):
    print("  %s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def serving():
    """(is it serving?, short status string). One tiny ranged request, nothing downloaded."""
    cert = HOME / ".certs" / "ca-bundle.pem"
    if cert.exists():
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(cert))
        os.environ.setdefault("SSL_CERT_FILE", str(cert))
    import requests
    c = json.loads((HOME / ".kaggle" / "kaggle.json").read_text())
    try:
        r = requests.get(URL, auth=(c["username"], c["key"]),
                         headers={"Range": "bytes=0-1023"}, stream=True,
                         timeout=(30, 60), allow_redirects=True)
        code = r.status_code
        body = "" if code in (200, 206) else r.text[:80].replace("\n", " ")
        r.close()
        return code in (200, 206), "%s %s" % (code, body)
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, str(e)[:60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=int)
    ap.add_argument("--every", type=int, default=1800, help="seconds between checks")
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--no-provision", action="store_true",
                    help="just report when it comes back, do not launch anything")
    a = ap.parse_args()

    log("watching %s for session %d, every %ds, giving up after %.0f h"
        % (COMP, a.session, a.every, a.max_hours))
    deadline = time.time() + a.max_hours * 3600
    n = 0
    while time.time() < deadline:
        n += 1
        ok, status = serving()
        if ok:
            log("SERVING after %d check(s) - the archive is back" % n)
            if a.no_provision:
                return 0
            log("launching provision_session.py %d" % a.session)
            r = subprocess.run([sys.executable,
                                str(ROOT / "modal_scripts" / "provision_session.py"),
                                str(a.session)], cwd=str(ROOT))
            log("provisioning exited %d" % r.returncode)
            return r.returncode
        if n == 1 or n % 6 == 0:            # first check, then hourly at the default rate
            log("still down (%s), check %d" % (status, n))
        time.sleep(a.every)
    log("gave up after %.0f h - the archive never came back" % a.max_hours)
    return 1


if __name__ == "__main__":
    sys.exit(main())
