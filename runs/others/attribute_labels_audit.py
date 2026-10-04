"""attribute_labels_audit.py - can the graded report attributes build labels closer to the PANEL than our v3 labels?

The panel reads images with explicit criteria (data_code/03_label_reports.py notes: OA = substantial cartilage LOSS, not
osteophytes; meniscus = a TEAR). The attributes grade exactly those things. For each finding, a FIXED composite (mean of
its relevant attributes, each scaled to [0, 1] by its maximum grade; silent = 0), no weights fitted on the 58:
scored alone and blended 50/50 (logit space) with the v3 label, against the panel on audit-58.

    python runs/others/attribute_labels_audit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path(__file__).resolve().parent.parent.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMP = {
    "ACL": ["acl"], "MCL": ["mcl"],
    "Medial Meniscus": ["med_meniscus", "med_meniscus_complex"], "Lateral Meniscus": ["lat_meniscus", "lat_meniscus_complex"],
    "Medial OA": ["med_cartilage", "med_osteophytes", "med_subchondral", "med_oa_stated", "jsn_med"],
    "Lateral OA": ["lat_cartilage", "lat_osteophytes", "lat_subchondral", "lat_oa_stated", "jsn_lat"],
    "PF OA": ["pf_cartilage", "pf_osteophytes", "pf_subchondral", "pf_oa_stated"],
    "Effusion": ["effusion"], "Synovitis": ["synovitis_stated", "synovial_thickening", "hoffa", "effusion"],
    "Baker's": ["baker"], "Contusion": ["contusion_lfc", "contusion_ltp", "contusion_mfc", "contusion_mtp", "contusion_patella"],
    "Fracture": ["fracture_traumatic", "insufficiency_fracture", "osteochondral_lesion"],
}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))   # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))   # noqa: E731
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
A = pd.read_csv(D / "25_report_attributes.csv").set_index("StudyInstanceUID")
A = (A.drop(columns=["model", "bad"]).astype(float) / A.drop(columns=["model", "bad"]).astype(float).max()).fillna(0).loc[u]
q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, T]
mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, T]
v3 = q.copy()
for t in ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]:
    v3[t] = sg(0.5 * lg(q[t].values) + 0.5 * lg(mg[t].values))
rng = np.random.default_rng(0)
print(f"{'finding':17s} {'pos':>4s} {'v3 label':>9s} {'attr comp':>10s} {'50/50':>8s} {'delta':>8s} {'better':>7s}")
rows = []
for t in T:
    y = aud.loc[u, "official_" + t].astype(int).values
    comp = A[COMP[t]].mean(1).values
    c01 = (comp - comp.min()) / (comp.max() - comp.min() + 1e-9)
    bl = sg(0.5 * lg(v3[t].values) + 0.5 * lg(0.05 + 0.9 * c01))
    a, b, c = roc_auc_score(y, v3[t]), roc_auc_score(y, comp), roc_auc_score(y, bl)
    d = []
    for _ in range(2000):
        i = rng.integers(0, 58, 58)
        if 0 < y[i].sum() < 58:
            d.append(roc_auc_score(y[i], bl[i]) - roc_auc_score(y[i], v3[t].values[i]))
    rows.append((a, b, c))
    print(f"{t:17s} {y.sum():4d} {a:9.4f} {b:10.4f} {c:8.4f} {c - a:+8.4f} {np.mean(np.array(d) > 0):7.0%}")
m = np.array(rows).mean(0)
print(f"{'macro':17s} {'':4s} {m[0]:9.4f} {m[1]:10.4f} {m[2]:8.4f} {m[2] - m[0]:+8.4f}")
