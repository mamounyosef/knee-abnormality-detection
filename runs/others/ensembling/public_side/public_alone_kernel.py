"""public_alone_kernel.py - the prize-safe public pipeline ALONE as a submittable notebook (its public_side.csv becomes
submission.csv), to learn its leaderboard score without RadImageNet. Parallel CoAt readers on, cap 8 h.

    python runs/others/ensembling/public_side/public_alone_kernel.py            # push + wait + check, NO submit
    python runs/others/ensembling/public_side/public_alone_kernel.py submit <V>  # submit kernel version V (ask the user first)
"""
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER, SLUG, DSET = "mamounhussam", "knee-public-alone", "knee-combined-b2cut"


def kag(*a):
    env = dict(os.environ)
    env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(Path.home() / ".certs" / "ca-bundle.pem")
    env["PYTHONUTF8"] = "1"
    r = subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True, env=env)
    return (r.stdout or "") + (r.stderr or "")


if len(sys.argv) > 2 and sys.argv[1] == "submit":
    print(kag("competitions", "submit", "-c", "rsna-knee-abnormality-detection", "-f", "submission.csv",
              "-k", f"{USER}/{SLUG}", "-v", sys.argv[2], "-m", "Public pipeline ALONE, prize-safe (Stage 3 RadImageNet removed)"))
    sys.exit(0)

src = (HERE.parents[2] / "shared_code.py").read_text(encoding="utf-8")
cell = next(n.value.value for n in ast.parse(src).body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_PUBLIC_SIDE_CELL")
pre = r"""import sys, glob, json, subprocess, traceback
from pathlib import Path
W = Path(sorted(glob.glob('/kaggle/input/**/public_pipeline.ipynb', recursive=True))[0]).parent
COMP = Path(sorted(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True))[0]).parent
import pandas as pd
pd.read_csv(COMP / 'sample_submission.csv').to_csv('/kaggle/working/submission.csv', index=False)   # 0.5 until replaced"""
pub = (cell.replace("__STRESS__", "0").replace("__PUBLIC_W__", "1.0").replace("__PUBLIC_CAP_H__", "8.0")
       .replace("__PUBLIC_ENV__", repr({"RSNA_PARALLEL_COAT_READERS": "1"})).replace("__PUBLIC_OVERLAP__", "False"))
post = r"""import pandas as pd
sub = pd.read_csv(COMP / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
if PUB_OK:
    p = pd.read_csv(PUB_CSV, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(sub.StudyInstanceUID)
    assert p.notna().all().all()
    p.reset_index().to_csv('/kaggle/working/submission.csv', index=False)
else:
    sub.to_csv('/kaggle/working/submission.csv', index=False)
print('PUBLIC ALONE', _pub['status'])"""
nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": c.splitlines(True), "outputs": [], "execution_count": None}
                for c in (pre, pub, post)],
      "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
txt = (HERE.parent / "configs" / "combined_public_b2.py").read_text(encoding="utf-8")
datasets = re.findall(r'"([\w-]+/[\w-]+)"', txt[txt.index('"datasets"'):txt.index('"models"')])
d = HERE / "alone_nb"
d.mkdir(exist_ok=True)
(d / "public_alone.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{USER}/{SLUG}", "title": SLUG.replace("-", " "), "code_file": "public_alone.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{USER}/{DSET}"] + datasets,
    "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [],
    "model_sources": ["metaresearch/dinov2/PyTorch/small/1"]}, indent=2), encoding="utf-8")
out = kag("kernels", "push", "-p", str(d))
print(out.strip()[-200:])
ver = (re.search(r"version\s+(\d+)", out, re.I) or [None, "?"])[1]
t0 = time.time()
while True:
    st = kag("kernels", "status", f"{USER}/{SLUG}").lower()
    if any(k in st for k in ("complete", "error", "cancel")):
        break
    print(f"  {int(time.time() - t0) // 60} min: running", flush=True)
    time.sleep(60)
print("kernel ended:", st.strip()[-80:])
o = HERE / "alone_out"
o.mkdir(exist_ok=True)
for f in o.glob("*"):
    if f.is_file():
        f.unlink()
kag("kernels", "output", f"{USER}/{SLUG}", "-p", str(o))
r = json.loads((o / "public_receipt.json").read_text()) if (o / "public_receipt.json").exists() else {}
import pandas as pd  # noqa: E402
s = pd.read_csv(o / "submission.csv") if (o / "submission.csv").exists() else None
ok = r.get("status") == "ok" and s is not None and float((s.iloc[:, 1:] == 0.5).to_numpy().mean()) < 0.5
print(f"receipt {r.get('status')} | rows {None if s is None else len(s)} | READY TO SUBMIT: {ok} (version {ver})")
print(f"  submit with: python runs/others/ensembling/public_side/public_alone_kernel.py submit {ver}")
