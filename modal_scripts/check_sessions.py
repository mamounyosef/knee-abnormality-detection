"""
check_sessions.py - is a workspace ready to train?

A run needs three things on its workspace, and discovering a missing one AFTER launching
costs a container start and, worse, can look like a training bug:
    knee-shards   the preprocessed studies. Without it there is nothing to train on.
    knee-runs     where checkpoints and logs are written.
    _pretrained/  trunks that do not come from timm (MRI-CORE and its MAE variant).
                  timm backbones download at run time and need nothing here.

    python modal_scripts/check_sessions.py 18 19 20
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECT_SHARDS = 4407          # studies, from other_md/outline.md


def modal(session, *args):
    e = dict(os.environ)
    e["MODAL_CONFIG_PATH"] = str(Path.home() / (".modal_session%d.toml" % session))
    e["PYTHONIOENCODING"] = "utf-8"
    e.pop("MSYS_NO_PATHCONV", None)
    # modal prints box-drawing characters; this PC's console codec is cp1252 and dies on
    # them, so decode as utf-8 and replace anything undecodable rather than raising
    r = subprocess.run([sys.executable, "-m", "modal"] + list(args),
                       env=e, cwd=str(ROOT), capture_output=True)
    return type("R", (), {"stdout": (r.stdout or b"").decode("utf-8", "replace"),
                          "stderr": (r.stderr or b"").decode("utf-8", "replace")})()


def check(session):
    tok = Path.home() / (".modal_session%d.toml" % session)
    if not tok.exists():
        return (session, "NO TOKEN", "-", "-", "-", "no config file")
    vols = modal(session, "volume", "list").stdout
    if "knee" not in vols and "Name" not in vols:
        return (session, "AUTH?", "-", "-", "-", (vols or "").strip()[:40])
    has_shards = "knee-shards" in vols
    has_runs = "knee-runs" in vols

    n_shards = "-"
    if has_shards:
        r = modal(session, "volume", "ls", "knee-shards")
        n = len([l for l in r.stdout.splitlines() if ".npz" in l])
        n_shards = str(n) if n else "0"

    pre = "-"
    if has_runs:
        r = modal(session, "volume", "ls", "knee-runs", "_pretrained")
        names = [l.split("/")[-1] for l in r.stdout.splitlines() if ".pth" in l]
        pre = ",".join(sorted(n.replace("MRI_CORE_", "").replace(".pth", "")
                              for n in names)) or "none"

    ready = has_shards and has_runs and n_shards not in ("-", "0")
    verdict = "READY" if ready else ("needs shards" if not has_shards
                                     else "needs knee-runs" if not has_runs else "empty shards")
    return (session, verdict, "yes" if has_shards else "NO",
            "yes" if has_runs else "NO", n_shards, pre)


sessions = [int(a) for a in sys.argv[1:]] or [18, 19, 20]
print("%-9s%-16s%-9s%-10s%-10s%s" % ("session", "verdict", "shards", "runs-vol",
                                     "n_shards", "_pretrained"))
for s in sessions:
    row = check(s)
    print("%-9d%-16s%-9s%-10s%-10s%s" % row)
print("\nn_shards may be a page, not the total; anything in the thousands is fine.")
print("timm backbones (SigLIP, MaxViT, ConvNeXt) need NOTHING in _pretrained.")
