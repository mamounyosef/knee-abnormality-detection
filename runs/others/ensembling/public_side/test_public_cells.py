"""test_public_cells.py - run the combined notebook's public-side and blend cells (shared_code._PUBLIC_SIDE_CELL,
_PUBLIC_BLEND_CELL) on Linux against a MOCK competition and a MOCK public notebook, through the real nbconvert.

Scenarios: stress (fake study IDs + symlinks + dry-run blend), normal (blended submission), public notebook raises
(ours kept), public notebook hangs past the cap (process group killed, ours kept), ours all 0.5 (public alone).
Needs Linux (symlinks, process groups): in WSL with a venv holding pandas, numpy, nbconvert, ipykernel:

    wsl -e ~/pubtest/bin/python "/mnt/d/My Projects/knee_abnormality_detection/runs/others/ensembling/public_side/test_public_cells.py"
"""
import ast
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[3] / "shared_code.py"
consts = {}
for node in ast.parse(ENGINE.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
            "_PUBLIC_SIDE_CELL", "_PUBLIC_BLEND_CELL"):
        consts[node.targets[0].id] = node.value.value
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ROOT = Path("/tmp/pubcells")
shutil.rmtree(ROOT, ignore_errors=True)
COMP, W, WORK = ROOT / "comp", ROOT / "w", ROOT / "work"
for d in (COMP / "test_series", W, WORK):
    d.mkdir(parents=True)
ids = ["1.2.840.111", "1.2.840.222", "1.2.840.333"]
pd.DataFrame({"StudyInstanceUID": ids}).to_csv(COMP / "test.csv", index=False)
pd.DataFrame({"StudyInstanceUID": ["9"] * 5}).to_csv(COMP / "train.csv", index=False)
rows = []
for u in ids:
    for s in ("11", "22"):
        (COMP / "test_series" / u / f"{u}.{s}").mkdir(parents=True)
        (COMP / "test_series" / u / f"{u}.{s}" / "1.dcm").write_text("x")
        rows.append({"StudyInstanceUID": u, "SeriesInstanceUID": f"{u}.{s}", "Anatomical_Plane": "Sagittal"})
pd.DataFrame(rows).to_csv(COMP / "test_series.csv", index=False)
sub = pd.DataFrame({"StudyInstanceUID": ids, **{t: 0.5 for t in T}})
sub.to_csv(COMP / "sample_submission.csv", index=False)

MOCK = r'''
import os, time, numpy as np, pandas as pd
from pathlib import Path
mode = os.environ['MOCK_MODE']
if mode == 'raise':
    raise RuntimeError('mock public failure')
if mode == 'hang':
    time.sleep(120)
root = Path(os.environ.get('RSNA_COMP_ROOT') or os.environ['MOCK_COMP'])
t = pd.read_csv(root / 'test.csv', dtype={'StudyInstanceUID': str})
for u in t.StudyInstanceUID:                      # every study folder must resolve to real series
    assert len(list((root / 'test_series' / u).iterdir())) == 2, u
assert float(os.environ['RSNA_TIME_BUDGET_S']) > 0
lab = [c for c in pd.read_csv(root / 'sample_submission.csv').columns if c != 'StudyInstanceUID']
rng = np.random.default_rng(0)
out = pd.DataFrame(rng.random((len(t), len(lab))), columns=lab)
out.insert(0, 'StudyInstanceUID', t.StudyInstanceUID)
out.to_csv(Path(os.environ['MOCK_WORK']) / 'public_side.csv', index=False)
print('mock public wrote', len(out))
'''
nb = {"cells": [{"cell_type": "code", "metadata": {}, "source": MOCK.splitlines(True), "outputs": [],
                 "execution_count": None}],
      "metadata": {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
(W / "public_pipeline.ipynb").write_text(json.dumps(nb), encoding="utf-8")


def run(mode, stress, w=0.5, cap=3.0, ours_neutral=False):
    import os
    os.environ.update(MOCK_MODE=mode, MOCK_COMP=str(COMP), MOCK_WORK=str(WORK))
    for f in WORK.iterdir():
        f.unlink() if f.is_file() else shutil.rmtree(f)
    pub = (consts["_PUBLIC_SIDE_CELL"].replace("/kaggle/working", str(WORK)).replace("__STRESS__", str(stress))
           .replace("__PUBLIC_W__", repr(w)).replace("__PUBLIC_CAP_H__", repr(cap)).replace("__PUBLIC_ENV__", "{}").replace("+ 600", "+ 2"))
    blend = consts["_PUBLIC_BLEND_CELL"]
    g = {"sys": sys, "json": json, "subprocess": subprocess, "traceback": traceback, "Path": Path,
         "W": W, "COMP": COMP, "pd": pd}
    cwd = os.getcwd()
    os.chdir(WORK)
    try:
        exec(compile(pub, "public_cell", "exec"), g)
        g["sub"], g["TASKS"] = sub.copy(), T
        rng = np.random.default_rng(1)
        if stress:
            todo = pd.concat([sub[["StudyInstanceUID"]]] * 43, ignore_index=True).head(stress)
            g["pred"] = todo.assign(**{t: rng.random(len(todo)) for t in T})
            sub.to_csv("submission.csv", index=False)               # bail() file
        else:
            ours = sub[["StudyInstanceUID"]].assign(**{t: (0.5 if ours_neutral else rng.random(3)) for t in T})
            ours.to_csv("submission.csv", index=False)
        exec(compile(blend, "blend_cell", "exec"), g)
        r = json.loads((WORK / "public_receipt.json").read_text())
        b = json.loads((WORK / "blend_receipt.json").read_text())
        s = pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str})
        return r, b, s, g
    finally:
        os.chdir(cwd)


fails = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {info}"))
    if not cond:
        fails.append(name)


r, b, s, g = run("ok", 128)
check("stress: public ok", r["status"] == "ok" and r["rows"] == 128, r)
check("stress: dry-run blend", b["mode"] == "stress dry run" and b["rows"] == 3, b)
check("stress: submission untouched (0.5 bail)", (s[T].to_numpy() == 0.5).all())
check("stress: fake root cleaned", not Path("/tmp/pub_stress_root").exists())
check("stress: preview written", (WORK / "stress_blend_preview.csv").exists())

r, b, s, g = run("ok", 0, w=0.5)
ours = pd.read_csv(WORK / "submission_ours.csv", dtype={"StudyInstanceUID": str})
pub = pd.read_csv(WORK / "public_side.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ours.StudyInstanceUID]
exp = 0.5 * ours[T].rank(pct=True).to_numpy() + 0.5 * pub[T].rank(pct=True).to_numpy()
check("normal: blended", b["mode"] == "blended", b)
check("normal: values = 0.5 rank(ours) + 0.5 rank(public)", np.allclose(s[T].to_numpy(), exp))
check("normal: row order kept", list(s.StudyInstanceUID) == ids)

wd = {t: (0.8 if t == "ACL" else 0.3) for t in T}
r, b, s, g = run("ok", 0, w=wd)
ours = pd.read_csv(WORK / "submission_ours.csv", dtype={"StudyInstanceUID": str})
pub = pd.read_csv(WORK / "public_side.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ours.StudyInstanceUID]
exp = np.stack([(1 - wd[t]) * ours[t].rank(pct=True).to_numpy() + wd[t] * pub[t].rank(pct=True).to_numpy() for t in T], 1)
check("per-finding weights", b["mode"] == "blended" and np.allclose(s[T].to_numpy(), exp), b)

r, b, s, g = run("raise", 0)
ours = pd.read_csv(WORK / "submission_ours.csv", dtype={"StudyInstanceUID": str}) if (WORK / "submission_ours.csv").exists() else None
check("raise: public failed", r["status"].startswith("failed"), r)
check("raise: ours kept", b["mode"] == "ours only" and ours is None, b)

r, b, s, g = run("hang", 0, cap=0.0005)
check("hang: timed out and killed", r.get("returncode") == "timeout" and r["seconds"] < 30, r)
check("hang: ours kept", b["mode"] == "ours only", b)
leftover = subprocess.run(["pgrep", "-f", "public_pipeline.ipynb"], capture_output=True, text=True).stdout.strip()
check("hang: no process left", leftover == "", leftover)

r, b, s, g = run("ok", 0, ours_neutral=True)
check("ours all 0.5: public alone", b["mode"].startswith("public only"), b)

print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
