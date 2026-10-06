"""resubmit_blend_weight.py - resubmit an already-built combined notebook with a different public-side blend weight.

The weights dataset is NOT touched: only the notebook's `PUBLIC_W = ...` line changes, the notebook is pushed as a new
kernel (its own slug), Kaggle's commit run must finish COMPLETE with a submission.csv and a blend receipt carrying the
new weight, and only then is it submitted. PUBLIC_W = weight of the PUBLIC side in the per-finding rank blend
(0.5 = 50/50; 0.4 = 60% ours / 40% public).

    python runs/others/ensembling/resubmit_blend_weight.py <notebook_dir> <public_w> <new_slug> "<message>" [--no-submit]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("REQUESTS_CA_BUNDLE", str(Path.home() / ".certs" / "ca-bundle.pem"))
os.environ.setdefault("SSL_CERT_FILE", os.environ["REQUESTS_CA_BUNDLE"])
args = [a for a in sys.argv[1:] if a != "--no-submit"]
SUBMIT = "--no-submit" not in sys.argv
src, w, slug, msg = Path(args[0]), float(args[1]), args[2], args[3]
COMP = "rsna-knee-abnormality-detection"


def kaggle(*a, check=True):
    p = subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and p.returncode:
        raise SystemExit(f"kaggle {' '.join(a[:3])} failed: {p.stdout[-400:]} {p.stderr[-400:]}")
    return p.stdout


work = Path("C:/temp1") / f"reblend_{slug}"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
meta = json.loads((src / "kernel-metadata.json").read_text(encoding="utf-8"))
nb = (src / meta["code_file"]).read_text(encoding="utf-8")
n = len(re.findall(r"PUBLIC_W = [0-9.]+", nb))
assert n == 1, f"expected exactly one PUBLIC_W line, found {n}"
nb = re.sub(r"PUBLIC_W = [0-9.]+", f"PUBLIC_W = {w}", nb)
user = meta["id"].split("/")[0]
meta["id"] = f"{user}/{slug}"
meta["title"] = slug.replace("-", " ")
(work / meta["code_file"]).write_text(nb, encoding="utf-8")
(work / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"pushing {meta['id']} with PUBLIC_W = {w} (datasets unchanged: {meta['dataset_sources'][0]} ...)", flush=True)
print(kaggle("kernels", "push", "-p", str(work))[-300:], flush=True)
t0 = time.time()
while True:
    time.sleep(60)
    out = kaggle("kernels", "status", meta["id"], check=False)
    st = out.strip().split('"')[-2] if '"' in out else out.strip()
    print(f"  {(time.time() - t0) / 60:5.1f} min  {st}", flush=True)
    if "COMPLETE" in st.upper():
        break
    if "ERROR" in st.upper() or "CANCEL" in st.upper():
        raise SystemExit(f"commit run ended {st}: not submitting")
od = work / "output"
kaggle("kernels", "output", meta["id"], "-p", str(od))
sub = od / "submission.csv"
rec = next(iter(od.rglob("blend_receipt.json")), None)
assert sub.exists(), "no submission.csv in the output: not submitting"
r = json.loads(rec.read_text(encoding="utf-8")) if rec else {}
print("blend receipt:", {k: r.get(k) for k in ("mode", "weight", "public_status", "error")}, flush=True)
assert abs(float(r.get("weight", -1)) - w) < 1e-9, f"blend receipt weight {r.get('weight')} != {w}: not submitting"
assert r.get("mode") == "blended" and "error" not in r, f"blend did not run cleanly ({r.get('mode')}): not submitting"
if not SUBMIT:
    print("--no-submit: stopping here")
    sys.exit(0)
print(kaggle("competitions", "submit", "-c", COMP, "-k", meta["id"], "-v", "1", "-f", "submission.csv", "-m", msg))
