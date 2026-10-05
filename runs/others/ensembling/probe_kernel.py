"""probe_kernel.py - a 1-minute Kaggle kernel with the SAME sources and machine as a saved stress-stamp notebook, that
only lists /kaggle/input and the GPUs. Tells a setup failure (a dataset that will not mount, no GPU) from a code failure.

    python runs/others/ensembling/probe_kernel.py <stamp fingerprint prefix>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
fp = sys.argv[1]
nbdir = next(p for p in (HERE / "stress_passes").glob(fp + "*_notebook"))
meta = json.loads((nbdir / "kernel-metadata.json").read_text(encoding="utf-8"))
user = meta["id"].split("/")[0]
slug = "knee-probe"
cell = ("import os, subprocess, glob\n"
        "print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)\n"
        "for d in sorted(glob.glob('/kaggle/input/**/', recursive=False) + glob.glob('/kaggle/input/*/*/')):\n"
        "    print(d, len(os.listdir(d)))\n"
        "open('/kaggle/working/probe_ok.txt', 'w').write('ok')\n")
nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": cell.splitlines(True), "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
d = HERE / "probe_nb"
d.mkdir(exist_ok=True)
(d / "probe.ipynb").write_text(json.dumps(nb), encoding="utf-8")
meta.update(id=f"{user}/{slug}", title=slug.replace("-", " "), code_file="probe.ipynb")
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def kag(*a):
    env = dict(os.environ)
    env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(Path.home() / ".certs" / "ca-bundle.pem")
    env["PYTHONUTF8"] = "1"
    r = subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True, env=env)
    return (r.stdout or "") + (r.stderr or "")


print(kag("kernels", "push", "-p", str(d)).strip()[-200:])
t0 = time.time()
while True:
    st = kag("kernels", "status", f"{user}/{slug}").lower()
    if any(k in st for k in ("complete", "kernelworkerstatus.error", "cancel")):
        break
    print(f"  {int(time.time() - t0)} s: {st.strip()[-40:]}", flush=True)
    time.sleep(20)
print("ended:", st.strip()[-60:], f"after {int(time.time() - t0)} s")
o = d / "out"
o.mkdir(exist_ok=True)
kag("kernels", "output", f"{user}/{slug}", "-p", str(o))
for p in o.glob("*.log"):
    txt = p.read_text(errors="replace")
    try:
        txt = "".join(x.get("data", "") for x in json.loads(txt))
    except Exception:
        pass
    print(txt[-3000:])
