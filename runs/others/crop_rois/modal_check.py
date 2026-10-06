"""modal_check.py - run checks/joint_zoom_check.py for a run on a cheap Modal T4 (real shards, the run's own config),
so a pre-launch check never competes with a training job on this PC's GPU.

    python -m modal run runs/others/crop_rois/modal_check.py --run convnext_base_25d_v5_joint_zoom_030_6v30sl
"""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else Path("/root")   # inside the container the script sits in /root
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

cfg = sc.load_config(ROOT / "runs" / "convnext_base_25d_18slices_v5_aug_base", verbose=False)   # volume names only
app = modal.App("knee-zoom-check")
image = sc.modal_image().add_local_dir(str(ROOT / "checks"), "/root/checks", ignore=["*.png", "__pycache__"])
shards = modal.Volume.from_name(cfg["modal"]["shards_volume"])
runs = modal.Volume.from_name(cfg["modal"]["runs_volume"])


@app.function(image=image, gpu="T4", cpu=4.0, memory=24 * 1024, timeout=3600,
              volumes={cfg["modal"]["shards_mount"]: shards, cfg["modal"]["runs_mount"]: runs})
def check(run: str) -> str:
    import os
    import subprocess
    env = dict(os.environ, KNEE_REMOTE="1", PYTHONUNBUFFERED="1")
    p = subprocess.run([sys.executable, "/root/checks/joint_zoom_check.py", run], env=env, capture_output=True, text=True)
    return p.stdout[-6000:] + ("\nSTDERR:\n" + p.stderr[-3000:] if p.returncode else "")


@app.local_entrypoint()
def main(run: str):
    print(check.remote(run))
