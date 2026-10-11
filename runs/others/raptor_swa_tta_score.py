"""raptor_swa_tta_score.py - score the logits from raptor_swa_tta_modal.py on val-346 (v3), audit-58 and clean-121, with
paired bootstrap vs s9000 (the submitted checkpoint). TTA = mean logit of a variant and its mirror.

    python runs/others/raptor_swa_tta_score.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "runs/others/raptor_swa_tta"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
L = {f.stem: np.load(f, allow_pickle=True) for f in D.glob("*.npz")}
ids = [str(s) for s in L["s9000"]["ids"]]
P = {k: pd.DataFrame(v["logits"], index=ids, columns=T) for k, v in L.items()}
P["s9000_tta"] = (P["s9000"] + P["s9000_mirror"]) / 2
P["swa3_tta"] = (P["swa3"] + P["swa3_mirror"]) / 2
val = pd.read_csv(ROOT / "data/23_val_v3_audit.csv").set_index("StudyInstanceUID")
aud = set(pd.read_csv(ROOT / "data/05_audit58.csv").StudyInstanceUID)
clean = pd.read_csv(ROOT / "data/24_val_clean121.csv").set_index("StudyInstanceUID")
SURF = {"val-346": val.loc[[s for s in ids if s not in aud], T],
        "audit-58": val.loc[[s for s in ids if s in aud], T],
        "clean-121": clean.loc[[s for s in clean.index if s in P["s9000"].index], T]}
rng = np.random.default_rng(0)


def auc_cols(y, p, I):
    out = np.zeros(len(I)); cnt = np.zeros(len(I))
    for j in range(y.shape[1]):
        yy, pp = y[I, j], p[I, j]
        n1 = yy.sum(1); n0 = yy.shape[1] - n1
        ok = (n1 > 0) & (n0 > 0)
        r = rankdata(pp, axis=1)
        out += np.where(ok, ((r * yy).sum(1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1), 0); cnt += ok
    return out / cnt


order = ["s9000", "s9000_mirror", "s9000_tta", "swa3", "swa3_mirror", "swa3_tta"]
res = {}
for sname, Ydf in SURF.items():
    y = (Ydf.values >= 0.5).astype(int)
    I = rng.integers(0, len(y), (2000, len(y)))
    full = np.arange(len(y))[None]
    base = P["s9000"].loc[Ydf.index].values
    b0, bI = auc_cols(y, base, full)[0], auc_cols(y, base, I)
    for k in order:
        p = P[k].loc[Ydf.index].values
        a = auc_cols(y, p, full)[0]
        res.setdefault(k, []).append((a, a - b0, (auc_cols(y, p, I) - bI > 0).mean()))
print(f"{'variant':14s} " + " | ".join(f"{s:>24s}" for s in SURF))
for k in order:
    print(f"{k:14s} " + " | ".join(f"{a:.4f} ({d:+.4f}, {p:4.0%})" for a, d, p in res[k]))
