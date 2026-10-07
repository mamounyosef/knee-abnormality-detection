"""zoom_ensemble_test.py - do the joint-zoom runs help the ENSEMBLE? (6 Oct)

Members: the batch-3 scaffold fourteen (old_label_members_ensemble.py BASE, split analogues) and the 6-member core of
the 0.943 submission (S6) inside it. Flat logit mean per finding (count-normalised when a member votes on a subset).
Zoom runs, each at the step with the best val macro over the 5 JOINT findings (val only; audit never picks):
  dual  = convnext_base_25d_v5_joint_zoom_dual, z030 = convnext_base_25d_v5_joint_zoom_030,
  z020  = convnext_base_25d_v5_joint_zoom (stopped at step 1250)
Modes per run, on both bases:
  add12        added on all 12 findings
  addJ         added on the 5 joint findings only (ACL, Medial/Lateral Meniscus, Medial/Lateral OA)
  addJ w2/w3   same at weight 2 / 3
  swapP        replaces the 2.5D parent (convnext_base_25d_18slices_v5_aug_base) on all 12
  swapPJ       replaces the parent on the joint findings only (parent keeps the other 7)
plus combinations (dual + z030 on the joint findings, both modes).
Surfaces: audit-58 (panel labels) and clean-121 (audit + 63 v5-val studies no member trained on, v3 labels).
Paired bootstrap (2000 resamples, same studies for base and variant): delta and the share of resamples where it helps.

    python runs/others/zoom_ensemble_test.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
src = (HERE / "old_label_members_ensemble.py").read_text(encoding="utf-8")
g = {"__file__": str(HERE / "old_label_members_ensemble.py"), "__name__": "olm_scaffold"}
exec(compile(src[:src.index("pool = sorted(")], "old_label_members_ensemble.py", "exec"), g)
T, BASE, SURF, R = g["T"], g["BASE"], g["SURF"], g["R"]
JOINT = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
JM = np.array([t in JOINT for t in T])
PARENT = "convnext_base_25d_18slices_v5_aug_base"
S6 = ["convnext_base_v5", "dinov3_convnext_base_v5", PARENT, "convnext_base_25d_poolattn_v5",
      "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip", "medimageinsight_davit_huge_336_v3_f0"]
ZOOM = {"dual": "convnext_base_25d_v5_joint_zoom_dual", "z030": "convnext_base_25d_v5_joint_zoom_030",
        "z020": "convnext_base_25d_v5_joint_zoom", "zax": "convnext_base_25d_v5_joint_zoom_030_ax",
        "zdino": "convnext_base_25d_v5_joint_zoom_030_dinov3",
        "z6v": "convnext_base_25d_v5_joint_zoom_030_6v30sl"}
OVERALL = {"zax", "z6v"}          # these pick their step by overall val macro (they may vote on all 12), the rest by joint
rng = np.random.default_rng(0)


def zoom_preds(run):
    v = pd.read_csv(R / run / "results" / "val_log.csv").drop_duplicates("step", keep="last")
    v = v[v.step > 0]
    v["j"] = v[[f"val_auroc_{t}" for t in (T if run in [ZOOM[k] for k in OVERALL] else JOINT)]].mean(axis=1)
    st = int(v.loc[v.j.idxmax(), "step"])
    g["STEP"][run] = st
    return st


steps = {k: zoom_preds(r) for k, r in ZOOM.items()}
print("zoom steps (best val joint macro):", steps)
missing = [m for m in S6 if m not in BASE]
print("S6 members not in the scaffold base:", missing or "none")


def auc_cols(y, p, I):
    """Per-bootstrap macro AUROC: y, p (n, 12); I (B, n) resample indices -> (B,)."""
    out = np.zeros(len(I))
    cnt = np.zeros(len(I))
    for j in range(y.shape[1]):
        yy, pp = y[I, j], p[I, j]
        n1 = yy.sum(1); n0 = yy.shape[1] - n1
        ok = (n1 > 0) & (n0 > 0)
        r = rankdata(pp, axis=1)
        a = ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
        out += np.where(ok, a, 0); cnt += ok
    return out / cnt


def full_auc(y, p):
    return auc_cols(y, p, np.arange(len(y))[None])[0]


data = {}
for s, (ids, Y) in SURF.items():
    P = {m: g["get"](m, ids).loc[ids, T].astype(float).to_numpy() for m in BASE}
    for k, r in ZOOM.items():
        P[k] = g["get"](r, ids).loc[ids, T].astype(float).to_numpy()
    y = Y.loc[ids, T].to_numpy().astype(int)
    data[s] = (P, y, rng.integers(0, len(ids), (2000, len(ids))))


def blend(P, members, extra=()):
    """members: names at weight 1 on all 12; extra: (name, weight, mask) votes."""
    num = sum(P[m] for m in members)
    den = np.full(len(T), float(len(members)))
    for name, w, mask in extra:
        num = num + P[name] * (w * mask)[None]
        den = den + w * mask
    return num / den[None]


def variants(base):
    out = {}
    for k in ZOOM:
        out[f"{k} add12"] = (base, [(k, 1, np.ones(12))])
        for w in (1, 2, 3):
            out[f"{k} addJ w{w}"] = (base, [(k, w, JM.astype(float))])
        out[f"{k} addO (other 7)"] = (base, [(k, 1, (~JM).astype(float))])
        if PARENT in base:
            nb = [m for m in base if m != PARENT]
            out[f"{k} swapP"] = (nb, [(k, 1, np.ones(12))])
            out[f"{k} swapPJ"] = (nb, [(k, 1, JM.astype(float)), (PARENT, 1, (~JM).astype(float))])
    out["dual+z030 addJ"] = (base, [("dual", 1, JM.astype(float)), ("z030", 1, JM.astype(float))])
    out["z030 addJ + zax add12"] = (base, [("z030", 1, JM.astype(float)), ("zax", 1, np.ones(12))])
    out["z030 addJ + zax addO"] = (base, [("z030", 1, JM.astype(float)), ("zax", 1, (~JM).astype(float))])
    out["z030+zax addJ"] = (base, [("z030", 1, JM.astype(float)), ("zax", 1, JM.astype(float))])
    out["z030+zax addJ + zax addO"] = (base, [("z030", 1, JM.astype(float)), ("zax", 1, np.ones(12))])
    if PARENT in base:
        nb = [m for m in base if m != PARENT]
        out["zax swapP + z030 addJ"] = (nb, [("zax", 1, np.ones(12)), ("z030", 1, JM.astype(float))])
    out["z030+zdino addJ"] = (base, [("z030", 1, JM.astype(float)), ("zdino", 1, JM.astype(float))])
    out["z030 addJ + zdino add12"] = (base, [("z030", 1, JM.astype(float)), ("zdino", 1, np.ones(12))])
    out["z030 addJ + zdino addO"] = (base, [("z030", 1, JM.astype(float)), ("zdino", 1, (~JM).astype(float))])
    out["z030+zdino addJ w2"] = (base, [("z030", 2, JM.astype(float)), ("zdino", 2, JM.astype(float))])
    out["z030 addJ + z6v add12"] = (base, [("z030", 1, JM.astype(float)), ("z6v", 1, np.ones(12))])
    out["z030 addJ + z6v addO"] = (base, [("z030", 1, JM.astype(float)), ("z6v", 1, (~JM).astype(float))])
    out["z030+z6v addJ"] = (base, [("z030", 1, JM.astype(float)), ("z6v", 1, JM.astype(float))])
    if PARENT in base:
        nb = [m for m in base if m != PARENT]
        out["z6v swapP + z030 addJ"] = (nb, [("z6v", 1, np.ones(12)), ("z030", 1, JM.astype(float))])
    out["dual add12 + z030 addJ"] = (base, [("dual", 1, np.ones(12)), ("z030", 1, JM.astype(float))])
    if PARENT in base:
        nb = [m for m in base if m != PARENT]
        out["dual swapP + z030 addJ"] = (nb, [("dual", 1, np.ones(12)), ("z030", 1, JM.astype(float))])
    return out


for bname, base in (("all 14", BASE), ("S6 (0.943 core)", [m for m in S6 if m in BASE])):
    print(f"\n=== base: {bname} ({len(base)} members)")
    ref = {}
    for s, (P, y, I) in data.items():
        p0 = blend(P, base)
        ref[s] = (full_auc(y, p0), auc_cols(y, p0, I))
    print(f"  base: audit {ref['audit58'][0]:.4f} | clean-121 {ref['clean121'][0]:.4f}")
    rows = []
    for name, (mem, extra) in variants(base).items():
        r = []
        for s, (P, y, I) in data.items():
            p = blend(P, mem, extra)
            d = auc_cols(y, p, I) - ref[s][1]
            r.append((full_auc(y, p), full_auc(y, p) - ref[s][0], (d > 0).mean()))
        rows.append((name, r))
    rows.sort(key=lambda x: -(x[1][0][1] + x[1][1][1]))
    for name, ((a, da, pa), (c, dc, pc)) in rows:
        flag = "  <- both up" if da > 0 and dc > 0 else ""
        print(f"  {name:26s} audit {a:.4f} ({da:+.4f}, {pa:4.0%}) | clean {c:.4f} ({dc:+.4f}, {pc:4.0%}){flag}")
