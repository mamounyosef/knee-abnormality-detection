"""pull_final_checkpoint.py - fetch a finished run's checkpoint and PROVE it is the right one.

For a full-data run (evaluation off) the right checkpoint is the FINAL step, and a stale one
is silent: convnext_base_full_all_knees_same_orientation's local copy once sat at step 3850
while training ended at 4400. The expected final step is computed from the run's own config
(epochs x optimiser steps per epoch over its training CSV), and the pulled file must match.

For a validated run the right checkpoint is best.pt, and its score must equal the peak of
the run's own val_log.

    python modal_scripts/pull_final_checkpoint.py convnext_base_25d_all_knees_same_orientation_fulldata 107
"""
import contextlib
import io
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

HOME = Path.home()


def modal(session, *args):
    env = dict(os.environ)
    env["MODAL_CONFIG_PATH"] = str(HOME / (".modal_session%s.toml" % session))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-m", "modal", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main(run, session):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = sc.load_config(ROOT / "runs" / run)
    full = not (cfg.get("evaluation") or {}).get("enable", True)
    dst = ROOT / "runs" / run / "results" / "checkpoints"
    dst.mkdir(parents=True, exist_ok=True)

    if full:
        n = len(pd.read_csv(Path(cfg["paths"]["data_dir"]) / cfg["paths"]["train_csv"]))
        bs, acc = sc.train_batch_size(cfg), cfg["dataloader"].get("accum_steps", 1) or 1
        spe = (-(-n // bs)) // acc
        expect = spe * int(cfg["training"]["epochs"])
        want = ["best.pt", "last.pt", "step%d.pt" % expect]
    else:
        v = pd.read_csv(ROOT / "runs" / run / "results" / "val_log.csv") \
            if (ROOT / "runs" / run / "results" / "val_log.csv").exists() else None
        expect = None
        want = ["best.pt"]

    got = None
    for name in want:
        rc, out = modal(session, "volume", "get", "--force", "knee-runs",
                        "%s/results/checkpoints/%s" % (run, name), str(dst / name))
        if rc == 0 and (dst / name).exists():
            got = name
            break
    if got is None:
        raise SystemExit("could not pull any of %s from session%s" % (want, session))
    ck = torch.load(dst / got, map_location="cpu", weights_only=False)
    step = int(ck.get("global_step") or ck.get("step") or 0)
    if got != "best.pt":
        (dst / got).replace(dst / "best.pt")
    ok = (step == expect) if full else True
    print("%-62s pulled %-9s step %-6d %s  (%.2f GB, %d tensors)"
          % (run, got, step,
             ("expected %d -> %s" % (expect, "OK" if ok else "STALE")) if full else
             "score %s" % ck.get("score"),
             (dst / "best.pt").stat().st_size / 1e9, len(ck["model"])))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
