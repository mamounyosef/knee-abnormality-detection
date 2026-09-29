"""poll_score.py - print one Kaggle submission's status every 2 min until it is scored.

    python runs/others/ensembling/poll_score.py <submission_ref> [<ref> ...]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
pending = set(sys.argv[1:])
for _ in range(240):                                   # 8 h at most
    out = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submissions", "-c",
                          "rsna-knee-abnormality-detection", "--csv"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    for ref in sorted(pending):
        row = next((l for l in out.splitlines() if l.startswith(ref + ",")), "")
        tail = row.split(",")[-3:] if row else ["not found"]
        print(f"[{time.strftime('%H:%M')}] {ref}: {tail}", flush=True)
        if "COMPLETE" in row or "ERROR" in row:
            pending.discard(ref)
    if not pending:
        break
    time.sleep(120)
