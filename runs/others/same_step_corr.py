"""same_step_corr.py - is Qwen 'different' only because it is half-trained? (7 Oct)

Compares prediction correlation of the fit-10 core members (+ z030 zoom) with each other and with
qwen35_2b_vlm_sys, twice: (a) every member at its CONVERGED step (the one the ensemble uses), (b) every member at the
SAME training step as Qwen (nearest saved step; all use 8 studies per update, so the same number of studies seen).
Correlation: per-finding Spearman over clean-121 (audit-58 + 63 v5-val studies no member trained on), averaged over
the 12 findings. Also each member's solo macro AUROC at that step.

    python runs/others/same_step_corr.py [qwen_step]
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
src = (HERE / "zoom_ensemble_test.py").read_text(encoding="utf-8")
z = {"__file__": str(HERE / "zoom_ensemble_test.py"), "__name__": "zoom_scaffold"}
exec(compile(src[:src.index("data = {}")], "zoom_ensemble_test.py", "exec"), z)
g, T, S6, BASE, full_auc = z["g"], z["T"], z["S6"], z["BASE"], z["full_auc"]
R = g["R"]
Q = "qwen35_2b_vlm_sys"
qv = pd.read_csv(R / Q / "results" / "val_log.csv").drop_duplicates(["run_id", "step"], keep="last")
QSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else int(qv.step.max())
MEM = [m for m in S6 if m in BASE] + [z["ZOOM"]["z030"], Q]
SHORT = {m: m.replace("convnext_base_", "cnb_").replace("medimageinsight_davit_huge_336_v3_f0", "mii_f0")
         .replace("_25d_v5_joint_zoom_030", "_zoom030")[:26] for m in MEM}
ids, Y = g["SURF"]["clean121"]
y = Y.loc[ids, T].to_numpy().astype(int)


def steps_of(run):
    d = R / run / "results" / "predictions"
    a = {int(x) for x in re.findall(r"_audit_s(\d+)\.npz", " ".join(f.name for f in d.glob("*.npz")))}
    v = {int(x) for x in re.findall(r"_val_s(\d+)\.npz", " ".join(f.name for f in d.glob("*.npz")))}
    return sorted(a & v)


def preds_at(run, step):
    """Predictions on clean-121 at a forced step (None = the member's converged step)."""
    runs = [run] + ([f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)] if "_v3_f" in run else [])
    old = {r: g["STEP"].get(r) for r in runs}
    for r in runs:
        if step is None:
            g["STEP"].pop(r, None) if r not in z["steps"].values() else None
        else:
            av = steps_of(r)
            g["STEP"][r] = min(av, key=lambda s: abs(s - step))
    used = g["STEP"].get(run, "best")
    p = g["get"](run, ids).loc[ids, T].astype(float).to_numpy()
    for r, s in old.items():
        if s is None:
            g["STEP"].pop(r, None)
        else:
            g["STEP"][r] = s
    return p, used


def table(P, title):
    print(f"\n=== {title}")
    n = len(MEM)
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = np.nanmean([spearmanr(P[MEM[i]][:, k], P[MEM[j]][:, k])[0] for k in range(len(T))])
    print(" " * 28 + "".join(f"{SHORT[m][:9]:>10s}" for m in MEM))
    for i, m in enumerate(MEM):
        print(f"  {SHORT[m]:26s}" + "".join("        - " if i == j else f"{C[i, j]:10.3f}" for j in range(n)))
    core = [i for i, m in enumerate(MEM) if m != Q]
    for i, m in enumerate(MEM):
        others = [j for j in core if j != i]
        print(f"  mean corr of {SHORT[m]:26s} with the other non-Qwen members: {np.nanmean(C[i, others]):.3f}")
    return C


qstep = QSTEP
conv, same = {}, {}
print(f"Qwen step {qstep}")
for m in MEM:
    if m == Q:
        g["STEP"][Q] = int(qv.loc[qv.val_macro_auroc.idxmax(), "step"])
        conv[m] = g["get"](Q, ids).loc[ids, T].astype(float).to_numpy()
        g["STEP"][Q] = min(steps_of(Q), key=lambda s: abs(s - qstep))
        same[m] = g["get"](Q, ids).loc[ids, T].astype(float).to_numpy()
        cu, su, mx = "best-so-far", g["STEP"][Q], max(steps_of(Q))
    else:
        conv[m], cu = preds_at(m, None)
        same[m], su = preds_at(m, qstep)
        mx = max(steps_of(m))
    print(f"  {SHORT[m]:26s} converged step {str(cu):>11s}: solo {full_auc(y, conv[m]):.4f} | at step {su} "
          f"(of {mx} saved): solo {full_auc(y, same[m]):.4f}")
table(conv, "converged members (Qwen at its best step so far)")
table(same, f"all members at the same step as Qwen (~{qstep})")
