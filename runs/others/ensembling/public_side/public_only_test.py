"""public_only_test.py - run ONLY the public side of the combined notebook on Kaggle (no ensemble), to learn fast whether
the prize-safe public pipeline works on our image and how long it takes. Uses the already-uploaded combined dataset
(it holds public_pipeline.ipynb). Never submits.

    python runs/others/ensembling/public_side/public_only_test.py [STRESS] [DATASET_SLUG] [EXTRA_ENV_JSON]

STRESS 0 = the 3 public studies; 128 = fake 128-study cohort, as in the stress run.
Outputs land in runs/others/ensembling/public_side/test_out/<kernel version>/.
"""
import ast
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2] / "shared_code.py"
STRESS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DSET = sys.argv[2] if len(sys.argv) > 2 else "knee-combined-b2"
USER, SLUG = "mamounhussam", "knee-public-side-test"
src = ENGINE.read_text(encoding="utf-8")
cell = next(n.value.value for n in ast.parse(src).body if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_PUBLIC_SIDE_CELL")
pre = r"""import sys, glob, json, subprocess, traceback
from pathlib import Path
hits = sorted(glob.glob('/kaggle/input/**/public_pipeline.ipynb', recursive=True))
assert hits, 'public_pipeline.ipynb not mounted'
W = Path(hits[0]).parent
COMP = Path(sorted(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True))[0]).parent
print('W', W, '| COMP', COMP)"""
pub = cell.replace("__STRESS__", str(STRESS)).replace("__PUBLIC_W__", "0.5").replace("__PUBLIC_CAP_H__", "3.0").replace("__PUBLIC_ENV__", "{}")
EXTRA_ENV = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}      # e.g. '{"RSNA_PARALLEL_COAT_READERS": "1"}'
if EXTRA_ENV:                                   # its own kernel, so it can run beside a plain test
    SLUG += "-env"
_hook = "    _penv['RSNA_TIME_BUDGET_S'] = str(int(PUBLIC_CAP_H * 3600))\n"
assert pub.count(_hook) == 1
pub = pub.replace(_hook, _hook + "    _penv.update(%r)\n" % (EXTRA_ENV,))
nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": c.splitlines(True), "outputs": [], "execution_count": None}
                for c in (pre, pub)],
      "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
# the combined config's PUBLIC_SIDE sits after its exec line, so read the lists from it directly
txt = (HERE.parent / "configs" / "combined_public_b2.py").read_text(encoding="utf-8")
datasets = re.findall(r'"([\w-]+/[\w-]+)"', txt[txt.index('"datasets"'):txt.index('"models"')])
d = HERE / "test_nb"
d.mkdir(exist_ok=True)
(d / "public_test.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{USER}/{SLUG}", "title": SLUG.replace("-", " "), "code_file": "public_test.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [f"{USER}/{DSET}"] + datasets, "competition_sources": ["rsna-knee-abnormality-detection"],
    "kernel_sources": [], "model_sources": ["metaresearch/dinov2/PyTorch/small/1"]}, indent=2), encoding="utf-8")
print(f"datasets: {len(datasets)} public + {DSET}; STRESS {STRESS}; extra env {EXTRA_ENV}")



def kag(*a):
    import os
    env = dict(os.environ)
    ca = Path.home() / ".certs" / "ca-bundle.pem"
    env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(ca)
    env["PYTHONUTF8"] = "1"
    r = subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True, env=env)
    return (r.stdout or "") + (r.stderr or "")


out = kag("kernels", "push", "-p", str(d))
print(out.strip()[-300:])
ver = (re.search(r"version\s+(\d+)", out, re.I) or [None, "?"])[1]
t0 = time.time()
while True:
    st = kag("kernels", "status", f"{USER}/{SLUG}").lower()
    if any(k in st for k in ("complete", "error", "cancel")):
        break
    print(f"  {int(time.time() - t0) // 60} min: running", flush=True)
    time.sleep(60)
print(f"kernel ended: {st.strip()[-120:]} after {(time.time() - t0) / 60:.1f} min")
o = HERE / "test_out" / f"v{ver}"
o.mkdir(parents=True, exist_ok=True)
kag("kernels", "output", f"{USER}/{SLUG}", "-p", str(o))
rp = o / "public_receipt.json"
if rp.exists():
    r = json.loads(rp.read_text(encoding="utf-8"))
    r.pop("log_tail", None) if r.get("status") == "ok" else None
    print("RECEIPT", json.dumps(r, indent=1)[:3000])
    if r.get("status") == "ok" and STRESS:
        print(f"public side: {r['seconds'] / 60:.1f} min for {STRESS} studies -> {r['seconds'] / STRESS * 946 / 3600:.2f} h "
              f"projected on 946 (includes one-off loading)")
else:
    print("no public_receipt.json; files:", [p.name for p in o.iterdir()])
print("outputs in", o)
