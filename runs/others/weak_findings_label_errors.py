"""weak_findings_label_errors.py - where do the label errors on the four weak findings come from, and would more teacher
weight on just those four fix them?

For Lateral OA, Synovitis, PF OA, Lateral Meniscus on audit-58 (panel labels):
  1. label AUROC as the teacher weight w grows (labels = sigmoid((1-w) logit(v3) + w * teacher logit), the v4 recipe;
     teacher = 50/50 ConvNeXt v2 folds + MII v3 folds, blind on the 58), w = 0 ... 1. v4 uses w = 0.5 for all 12.
  2. the panel-positive studies the v3 labels miss (label < 0.5): does the report mention the finding at all
     (keyword search in the English report), and does the teacher catch it (teacher prob > 0.5)?

    python runs/others/weak_findings_label_errors.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
D = HERE.parent.parent / "data"
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEAK = ["Lateral OA", "Synovitis", "PF OA", "Lateral Meniscus"]
KEYS = {"Lateral OA": ["lateral", "osteoarth", "chondr", "cartilage", "joint space", "osteophyte", "degenerat"],
        "Synovitis": ["synovi", "synovial thicken", "hoffa", "inflamm"],
        "PF OA": ["patell", "trochle", "chondromalac", "femoropatell", "patellofemoral"],
        "Lateral Meniscus": ["lateral menisc", "meniscus", "menisc"]}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))   # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))   # noqa: E731

aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = pd.DataFrame(aud.loc[u, ["official_" + t for t in T]].values.astype(int), index=u, columns=T)
q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, T]
mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, T]
v3 = q.copy()
for t in ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]:
    v3[t] = sg(0.5 * lg(q[t].values) + 0.5 * lg(mg[t].values))


def fold_mean(prefix):
    out = []
    for k in range(5):
        z = np.load(sw.best_npz(f"{prefix}{k}", "audit"), allow_pickle=True)
        out.append(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[u])
    return sum(out) / 5


teach = 0.5 * fold_mean("convnext_base_25d_18slices_f") + 0.5 * fold_mean("medimageinsight_davit_huge_336_v3_f")
rep = pd.read_csv(D / "02_reports_en.csv")
tcol = next(c for c in rep.columns if c.lower() in ("report_en", "report", "text", "report_text", "findings"))
rep = rep.set_index("StudyInstanceUID")[tcol].reindex(u).fillna("").str.lower()

print("1. label AUROC on audit-58 as the teacher weight grows (v4 = w 0.5)")
ws = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
print(f"{'finding':17s} " + " ".join(f"w{w:<4}" for w in ws))
for t in WEAK + ["(macro of the 4)"]:
    if t.startswith("("):
        vals = [np.mean([roc_auc_score(Y[x], sg((1 - w) * lg(v3[x].values) + w * teach[x].values)) for x in WEAK]) for w in ws]
    else:
        vals = [roc_auc_score(Y[t], sg((1 - w) * lg(v3[t].values) + w * teach[t].values)) for w in ws]
    print(f"{t:17s} " + " ".join(f"{v:.4f}" for v in vals))

print("\n2. panel-positive studies the v3 labels miss (label < 0.5)")
for t in WEAK:
    pos = [s for s in u if Y.loc[s, t] == 1]
    miss = [s for s in pos if v3.loc[s, t] < 0.5]
    ment = [s for s in miss if any(k in rep[s] for k in KEYS[t])]
    caught = [s for s in miss if sg(teach.loc[s, t]) > 0.5]
    fp = [s for s in u if Y.loc[s, t] == 0 and v3.loc[s, t] >= 0.5]
    print(f"  {t:17s} positives {len(pos):2d}, missed {len(miss):2d} (report mentions a keyword: {len(ment)}, "
          f"teacher catches: {len(caught)}) | false positives {len(fp)}")
