"""kaggle_prune_datasets.py [--apply] [--before YYYY-MM-DD] - delete OUR old knee-* Kaggle datasets (cached submission
bundles) last updated before a date, to free the private-dataset quota. Scored submissions keep their scores; a deleted
bundle is rebuilt from its checkpoints by submit_ensemble.py if it is ever needed again. Non-knee datasets are never
touched. Dry run by default.

    python runs/others/ensembling/kaggle_prune_datasets.py --before 2026-10-01           # list
    python runs/others/ensembling/kaggle_prune_datasets.py --before 2026-10-01 --apply   # delete
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
import kaggle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--before", default="2026-10-01")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

rows, page = [], 1
while True:
    ds = kaggle.api.dataset_list(user="mamounhussam", page=page)
    if not ds:
        break
    rows += ds
    page += 1
old = [d for d in rows if d.ref.split("/")[1].startswith("knee-") and str(d.last_updated)[:10] < a.before]
print(f"{len(old)} knee-* datasets last updated before {a.before}, {sum(d.total_bytes or 0 for d in old) / 1e9:.1f} GB")
freed = 0
for d in sorted(old, key=lambda d: str(d.last_updated)):
    print(f"   {str(d.last_updated)[:10]}  {(d.total_bytes or 0) / 1e9:5.2f} GB  {d.ref}", flush=True)
    if a.apply:
        r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "delete", d.ref, "-y"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok = r.returncode == 0
        freed += (d.total_bytes or 0) if ok else 0
        print(f"      {'deleted' if ok else 'FAILED: ' + (r.stderr or r.stdout).strip()[-200:]}", flush=True)
if a.apply:
    print(f"freed {freed / 1e9:.1f} GB")
