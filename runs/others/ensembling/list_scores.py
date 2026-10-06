"""list_scores.py - print the latest Kaggle submissions: ref, date, status, public score, start of description.

    python runs/others/ensembling/list_scores.py [n]
"""
import csv
import io
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
out = subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submissions", "-c",
                      "rsna-knee-abnormality-detection", "--csv"],
                     capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
for r in list(csv.DictReader(io.StringIO(out)))[:n]:
    print(r["ref"], r["date"][:16], r["status"].split(".")[-1], r["publicScore"] or "-", "|", r["description"][:70])
