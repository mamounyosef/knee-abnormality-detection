"""label_vs_model_per_finding.py - per finding on audit-58: how good are the TRAINING LABELS (v3 report labels, the
v4 blend) against the panel, versus the batch-2 ensemble (split analogues)? If a finding's labels are weak, the models
trained on them hit that ceiling no matter how specialised they are.

    python runs/others/label_vs_model_per_finding.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
D = HERE.parent.parent / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402
from ten_mii_folds_audit import TEN_SPLIT  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))   # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))   # noqa: E731
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = aud.loc[u, ["official_" + t for t in T]].values.astype(int)
q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, T].values
mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, T].values
bi = [T.index(t) for t in ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]]
v3 = q.copy()
v3[:, bi] = sg(0.5 * lg(q[:, bi]) + 0.5 * lg(mg[:, bi]))


def P(run, step=None):
    d = HERE.parent / run / "results" / "predictions"
    f = next(d.glob(f"*_audit_s{step}.npz")) if step else Path(sw.best_npz(run, "audit"))
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[u].values


B12 = {"dinov3_convnext_base": "dinov3_convnext_base_v5", "convnext_base_relabel": "convnext_base_v5",
       "convnext_base_25d_all_knees_same_orientation": "convnext_base_25d_18slices_v5_aug_base",
       "convnext_base_25d_poolattn": "convnext_base_25d_poolattn_v5",
       "mricore_vitb_cross_attention": "mricore_vitb_cross_attention_v5",
       "mricore_vitb_mae_cross_attention": "mricore_vitb_mae_cross_attention_v5"}
base = [B12.get(TEN_SPLIT[n], TEN_SPLIT[n]) for n in TEN_SPLIT if n != "medimageinsight_davit_huge_336_fulldata"] + \
       [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]
ens = np.mean([P(r, 3700 if r == "convnext_base_25d_poolattn_v5" else None) for r in base], 0)
mii = np.mean([P(f"medimageinsight_davit_huge_336_v3_f{k}") for k in range(5)], 0)
print(f"{'finding':17s} {'positives':>9s} {'v3 labels':>10s} {'ensemble':>9s} {'MII folds':>10s}  (audit-58 AUROC vs the panel)")
rows = []
for j, t in enumerate(T):
    a, b, c = (roc_auc_score(Y[:, j], x[:, j]) for x in (v3, ens, mii))
    rows.append((t, Y[:, j].sum(), a, b, c))
for t, n, a, b, c in sorted(rows, key=lambda r: r[2]):
    print(f"{t:17s} {n:9d} {a:10.4f} {b:9.4f} {c:10.4f}")
print(f"{'macro':17s} {'':9s} {np.mean([r[2] for r in rows]):10.4f} {np.mean([r[3] for r in rows]):9.4f} {np.mean([r[4] for r in rows]):10.4f}")
