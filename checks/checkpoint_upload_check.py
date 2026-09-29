"""checkpoint_upload_check.py - does output.upload_checkpoints really land checkpoints on Modal?

Uses the engine's own write path (_write_checkpoint -> CheckpointUploader) with the given Modal
config: writes small step / top / best / last checkpoints into a temp results folder, DELETES the
step file straight away (the rolling prune racing the upload), waits for the uploads, then lists
the remote folder and compares sizes. last.pt must NOT be uploaded (not in the patterns).
Cleans the remote test folder afterwards.

    python checks/checkpoint_upload_check.py [<modal config path>]
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

cfg_path = sys.argv[1] if len(sys.argv) > 1 else sc.resolve_upload_modal_config("local")
RUN = "zz_upload_check"
res = Path(tempfile.mkdtemp()) / "results"
ck = res / "checkpoints"
ck.mkdir(parents=True)
sc._UPLOADER = sc.CheckpointUploader({"modal_config": cfg_path, "patterns": ["step*.pt", "top_step*.pt", "best.pt"]},
                                     RUN, res, "results/checkpoints", "knee-runs")
payload = {"model": {"w": torch.randn(256, 256)}, "step": 1}
sizes = {}
for n in ("step500.pt", "top_step500.pt", "best.pt", "last.pt"):
    sc._write_checkpoint(payload, ck / n)
    sizes[n] = (ck / n).stat().st_size
(ck / "step500.pt").unlink()                 # the prune deletes it before the upload runs
sc._UPLOADER.join()
sc._UPLOADER = None

env = dict(os.environ, MODAL_CONFIG_PATH=str(Path(os.path.expanduser(cfg_path))), PYTHONUTF8="1")
r = subprocess.run([sys.executable, "-m", "modal", "volume", "ls", "--json", "knee-runs",
                    f"/{RUN}/results/checkpoints"], capture_output=True, text=True, env=env)
import json  # noqa: E402
remote = {Path(e["Filename"]).name: e["Size"] for e in json.loads(r.stdout or "[]")}
print("remote:", remote)
fails = 0
for n in ("step500.pt", "top_step500.pt", "best.pt"):
    ok = n in remote
    fails += not ok
    print(f"  {'OK  ' if ok else 'FAIL'} {n} uploaded (local {sizes[n]} bytes, remote {remote.get(n)})")
ok = "last.pt" not in remote
fails += not ok
print(f"  {'OK  ' if ok else 'FAIL'} last.pt NOT uploaded (not in the patterns)")
staged_left = list((res / "_upload_staging").glob("*"))
fails += bool(staged_left)
print(f"  {'OK  ' if not staged_left else 'FAIL'} staging folder emptied after upload ({len(staged_left)} left)")
subprocess.run([sys.executable, "-m", "modal", "volume", "rm", "-r", "knee-runs", f"/{RUN}"],
               capture_output=True, text=True, env=env)
print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
