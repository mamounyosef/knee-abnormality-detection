"""ttt_sim_ensemble.py - does the test-time-trained student lift fit-10's core? (7 Oct)

convnext_base_25d_v5_ttt_sim = the parent member fine-tuned ONLY on the 404 stand-in test studies with the teacher's
soft labels (data_code/26_make_ttt_pseudo.py). Here, at every saved step, against the REAL labels:
  solo           student alone vs the parent's 0.9174 val-404 / 0.9209 audit
  swap parent    fit-10 core (S6 + z030 on joint) with the parent replaced by the student
  add            the student as one more vote
Surfaces: audit-58 (panel), clean-121 (audit + 63 v5-val studies no member trained on). Paired bootstrap share.

    python runs/others/ttt_sim_ensemble.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
src = (HERE / "zoom_ensemble_test.py").read_text(encoding="utf-8")
z = {"__file__": str(HERE / "zoom_ensemble_test.py"), "__name__": "zoom_scaffold"}
exec(compile(src[:src.index("for bname, base in")], "zoom_ensemble_test.py", "exec"), z)
g, T, S6, BASE, JM, data, blend, full_auc, auc_cols, R, PARENT = (z[k] for k in (
    "g", "T", "S6", "BASE", "JM", "data", "blend", "full_auc", "auc_cols", "R", "PARENT"))
RUN = "convnext_base_25d_v5_ttt_sim"
core = [m for m in S6 if m in BASE]
Z = [("z030", 1, JM.astype(float))]
steps = sorted({int(s) for s in re.findall(r"_audit_s(\d+)\.npz", " ".join(
    f.name for f in (R / RUN / "results" / "predictions").glob("*.npz")))})
print("student steps:", steps)


def score(mem, extra):
    out = []
    for s, (P, y, I) in data.items():
        p = blend(P, mem, extra)
        out.append((full_auc(y, p), auc_cols(y, p, I)))
    return out


ref = score(core, Z)
print(f"fit-10 core: audit {ref[0][0]:.4f} | clean-121 {ref[1][0]:.4f}")
for st in steps:
    g["STEP"][RUN] = st
    for s, (P, y, I) in data.items():
        ids = g["SURF"][s][0]
        P["ttt"] = g["get"](RUN, ids).loc[ids, T].astype(float).to_numpy()
    solo = [full_auc(y, P["ttt"]) for s, (P, y, I) in data.items()]
    line = f"step {st:4d}: solo audit {solo[0]:.4f} clean {solo[1]:.4f}"
    for name, (mem, extra) in {"swap parent": ([m for m in core if m != PARENT] + ["ttt"], Z),
                               "add": (core + ["ttt"], Z)}.items():
        r = score(mem, extra)
        d = [(a - ra, ((c - rc) > 0).mean()) for (a, c), (ra, rc) in zip(r, ref)]
        line += f" | {name}: audit {r[0][0]:.4f} ({d[0][0]:+.4f}, {d[0][1]:.0%}) clean {r[1][0]:.4f} ({d[1][0]:+.4f}, {d[1][1]:.0%})"
    print(line)
