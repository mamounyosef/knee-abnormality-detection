"""modal_head_sweep.py - run seq_head_sweep.py (pooling heads on cached slice features) on a cheap Modal T4, so it never
competes with a training job for this PC's RAM or GPU. The 0.85 GB feature cache travels in the image.

    python -m modal run runs/others/modal_head_sweep.py --heads "mean lse vieww smooth_top2 top4" --seeds 3
"""
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/root")   # inside the container the script sits in /root
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

app = modal.App("knee-head-sweep")
image = sc.modal_image().add_local_dir(str(ROOT / "runs" / "others" / "features" / "convnext_base_relabel"),
                                       "/root/runs/others/features/convnext_base_relabel")


@app.function(image=image, gpu="T4", cpu=4.0, memory=16 * 1024, timeout=3600)
def sweep(heads: str, seeds: int, epochs: int) -> str:
    import subprocess
    p = subprocess.run([sys.executable, "/root/runs/others/seq_head_sweep.py", "--heads", *heads.split(),
                        "--seeds", str(seeds), "--epochs", str(epochs)], capture_output=True, text=True, cwd="/root")
    return p.stdout[-8000:] + ("\nSTDERR:\n" + p.stderr[-3000:] if p.returncode else "")


@app.local_entrypoint()
def main(heads: str = "mean lse vieww smooth_top2 top4", seeds: int = 3, epochs: int = 30):
    print(sweep.remote(heads, seeds, epochs))
