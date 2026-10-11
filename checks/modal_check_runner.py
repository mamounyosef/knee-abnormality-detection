"""modal_check_runner.py - run a checks/*.py script on a Modal CPU container instead of this PC (heavy checks only).

The image is the engine's own (runs/ code + small data CSVs), plus checks/. The shards volume is mounted at /shards
and linked to /root/data/shards, and sc.load_config is wrapped with sc.remote_cfg so every config the check loads
points at the container paths. CPU only: torch.cuda stays unavailable.

    (select the Modal profile first)
    python -m modal run checks/modal_check_runner.py --name raptor96_check.py
"""
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "runs"), "/root/runs"):      # local repo, or the container (entrypoint lands in /root)
    if Path(_p).exists():
        sys.path.insert(0, _p)
import shared_code as sc  # noqa: E402

app = modal.App("check-runner")
image = sc.modal_image().add_local_dir(str(ROOT / "checks"), "/root/checks", ignore=["__pycache__"])
shards = modal.Volume.from_name("knee-shards")


@app.function(image=image, cpu=8.0, memory=32768, timeout=3600, volumes={"/shards": shards}, retries=0)
def run_check(name: str, args: str = ""):
    import os
    import runpy

    os.makedirs("/root/data", exist_ok=True)
    if not os.path.exists("/root/data/shards"):
        os.symlink("/shards", "/root/data/shards")
    sys.path.insert(0, "/root/runs")
    import shared_code as rsc
    _orig = rsc.load_config
    rsc.load_config = lambda *a, **k: rsc.remote_cfg(_orig(*a, **k))
    sys.argv = [f"/root/checks/{name}"] + args.split()
    runpy.run_path(f"/root/checks/{name}", run_name="__main__")


@app.local_entrypoint()
def main(name: str, args: str = ""):
    run_check.remote(name, args)
