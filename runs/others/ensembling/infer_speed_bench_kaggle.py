"""infer_speed_bench_kaggle.py - run infer_speed_bench.py on a Kaggle T4 against an uploaded ensemble dataset. Never submits.

    python runs/others/ensembling/infer_speed_bench_kaggle.py [DATASET_SLUG] [MEMBER_INDICES]
Output: runs/others/ensembling/speed_bench_out/speed_bench.json
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DSET = sys.argv[1] if len(sys.argv) > 1 else "knee-combined-b2"
IDX = sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3,4,5,8,9"
USER, SLUG = "mamounhussam", "knee-speed-bench"
bench = (HERE / "infer_speed_bench.py").read_text(encoding="utf-8")
cell = ("import glob, subprocess, sys\nfrom pathlib import Path\n"
        "W = Path(sorted(glob.glob('/kaggle/input/**/shared_code.py', recursive=True))[0]).parent\n"
        f"Path('/kaggle/working/infer_speed_bench.py').write_text({bench!r})\n"
        f"r = subprocess.run([sys.executable, '/kaggle/working/infer_speed_bench.py', str(W), {IDX!r}], "
        "capture_output=True, text=True, cwd='/kaggle/working')\n"
        "print(r.stdout[-20000:]); print(r.stderr[-5000:])\n")
nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": cell.splitlines(True), "outputs": [], "execution_count": None}],
      "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
d = HERE / "speed_bench_nb"
d.mkdir(exist_ok=True)
(d / "bench.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(d / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{USER}/{SLUG}", "title": SLUG.replace("-", " "), "code_file": "bench.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{USER}/{DSET}"],
    "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": []}, indent=2), encoding="utf-8")


def kag(*a):
    env = dict(os.environ)
    env["REQUESTS_CA_BUNDLE"] = env["SSL_CERT_FILE"] = str(Path.home() / ".certs" / "ca-bundle.pem")
    env["PYTHONUTF8"] = "1"
    r = subprocess.run([sys.executable, "-m", "kaggle", *a], capture_output=True, text=True, env=env)
    return (r.stdout or "") + (r.stderr or "")


print(kag("kernels", "push", "-p", str(d)).strip()[-200:])
t0 = time.time()
while True:
    st = kag("kernels", "status", f"{USER}/{SLUG}").lower()
    if any(k in st for k in ("complete", "error", "cancel")):
        break
    print(f"  {int(time.time() - t0) // 60} min: running", flush=True)
    time.sleep(60)
print("kernel ended:", st.strip()[-80:])
o = HERE / "speed_bench_out"
o.mkdir(exist_ok=True)
for f in o.iterdir():
    f.unlink()
kag("kernels", "output", f"{USER}/{SLUG}", "-p", str(o))
f = o / "speed_bench.json"
if f.exists():
    j = json.loads(f.read_text(encoding="utf-8"))
    print(j["gpu"], j["torch"])
    for r in j["rows"]:
        print(json.dumps(r))
else:
    print("no speed_bench.json; files:", [p.name for p in o.iterdir()])
    for p in o.glob("*.log"):
        print(p.read_text(errors="replace")[-3000:])
