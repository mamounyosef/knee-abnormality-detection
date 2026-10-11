"""jev_labels_audit.py - the public JEV-Knee HYBRID labels (riadmohamed42) vs our v3 / v4 labels on audit-58 (11 Oct).

Caveat: HYBRID picked its per-target source ON the 58 (and 16 more), so its audit number is optimistic; ours are not.
Blends in logit space, w = share of JEV.

    python runs/others/jev_labels_audit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = (aud.loc[u, ["official_" + t for t in T]].values >= 0.5).astype(int)
jev = pd.read_csv(Path(r"C:\temp1\pubnb\jevdata\HYBRID_labels_scores.csv")).set_index("StudyInstanceUID")
# v3 on the audit studies = Qwen + MedGemma on the 5 MedGemma findings (as data_code/22_make_convnext_mii_labels.py)
q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, T].values
mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, T].values
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
bi = [T.index(t) for t in ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]]
v3 = q.copy(); v3[:, bi] = sg(0.5 * lg(q[:, bi]) + 0.5 * lg(mg[:, bi]))
J = jev.loc[u, T].astype(float)
print("JEV NaN cells on the 58:", int(J.isna().sum().sum()), "-> filled with 0.5 (no read)")
J = J.fillna(0.5).values


def per(P):
    return [roc_auc_score(Y[:, j], P[:, j]) for j in range(12)]


print(f"audit-58 macro AUROC: v3 {np.mean(per(v3)):.4f} | JEV HYBRID {np.mean(per(J)):.4f}")
for w in (0.1, 0.2, 0.3, 0.5):
    b = sg((1 - w) * lg(v3) + w * lg(J))
    print(f"  v3 + JEV at w {w:.1f}: {np.mean(per(b)):.4f}")
print("\nper finding: v3 | JEV")
for t, a, b in zip(T, per(v3), per(J)):
    print(f"  {t:17s} {a:.4f} | {b:.4f}")

# v4 (what every current member trains on) = 0.5 v3 + 0.5 OOF teachers; on the 58 its audit-side twin is the
# w50 blend of v3 with the round-1 teacher means (data_code/22_make_convnext_mii_labels.py reproduces 0.9453)
import sys as _sys
_sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
import sweep_new_runs as sw  # noqa: E402


def _fam(pat):
    out = []
    for k in range(5):
        z = np.load(sw.best_npz(pat.format(k), "audit"), allow_pickle=True)
        out.append(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[u].values)
    return np.mean(out, 0)


teach = 0.5 * _fam("convnext_base_25d_18slices_f{}") + 0.5 * _fam("medimageinsight_davit_huge_336_v3_f{}")
v4 = sg(0.5 * lg(v3) + 0.5 * teach)
print(f"\nv4 on audit-58: {np.mean(per(v4)):.4f} (expected 0.9453)")
for w in (0.05, 0.1, 0.15, 0.2, 0.3):
    b = sg((1 - w) * lg(v4) + w * lg(J))
    print(f"  v4 + JEV at w {w:.2f}: {np.mean(per(b)):.4f}")
