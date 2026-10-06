"""silence_label_test.py - does treating "report silent" per finding fix our labels? (forum thread 733932, 6 Oct)

Report state per study x finding from the pilkwang LLM labels (CC0): YES / NO / UNK (not addressed). On the 58 audit
studies (panel labels) this prints, per finding: share of silent reports, panel-positive rate among silent ones, and
the label-key AUROC of:
  v3       our report labels (05_audit58.csv)
  teacher  mean logit of the 10 out-of-fold teachers the v4 labels used (5 ConvNeXt + 5 MII fold models; the audit
           studies are in no fold, so every fold model is out of sample there)
  v4       0.5 logit(v3) + 0.5 teacher  (what our members train on now)
  silentT  v4, but silent cells take the teacher alone (silence = "nobody looked")
  silentN  v4, but silent cells pushed to "absent" (silence = "normal")
  perF     per finding the better of silentT / v4 by the forum's rule: silence = unknown for Synovitis, PF OA,
           Effusion, Fracture, Contusion, Lateral OA; = absent for the rest (rule fixed BEFORE looking, from thread 733932)

    python runs/others/silence_label_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
UNKNOWN_IF_SILENT = {"Synovitis", "PF OA", "Effusion", "Fracture", "Contusion", "Lateral OA"}
au = pd.read_csv(ROOT / "data/05_audit58.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
pk = pd.read_csv(ROOT / "data/external/pilkwang_llm_labels/report_labels_v2.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
au = au[au.index.isin(pk.index)]
ids = list(au.index)
Y = au[["official_" + t for t in T]].to_numpy() > 0.5
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
v3 = lg(au[T].to_numpy())
tea = []
for name in ["convnext_base_25d_18slices_f{}", "medimageinsight_davit_huge_336_v3_f{}"]:
    for k in range(5):
        f = sw.best_npz(name.format(k), "audit")
        z = np.load(f, allow_pickle=True)
        df = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T)
        tea.append(df.loc[ids, T].to_numpy())
teacher = np.mean(tea, 0)
silent = (pk.loc[ids, [f"{t}__verdict" for t in T]].to_numpy() == "UNK")
v4 = 0.5 * v3 + 0.5 * teacher
cands = {"v3": v3, "teacher": teacher, "v4": v4,
         "silentT": np.where(silent, teacher, v4),
         "silentN": np.where(silent, np.minimum(v4, v4.min(0, keepdims=True)), v4)}
rule = np.array([t in UNKNOWN_IF_SILENT for t in T])[None]
cands["perF"] = np.where(silent & rule, teacher, v4)
print(f"{len(ids)} audit studies\n")
print(f"{'finding':17s} silent  pos|silent  " + " ".join(f"{k:>8s}" for k in cands))
macro = {k: [] for k in cands}
for j, t in enumerate(T):
    s = silent[:, j]
    pr = Y[s, j].mean() if s.any() else float("nan")
    row = []
    for k, p in cands.items():
        a = roc_auc_score(Y[:, j], p[:, j]); macro[k].append(a); row.append(a)
    print(f"{t:17s} {s.mean():5.0%}   {pr:6.0%}     " + " ".join(f"{a:8.3f}" for a in row) + ("   (rule: unknown)" if t in UNKNOWN_IF_SILENT else ""))
print(f"{'MACRO':17s}                  " + " ".join(f"{np.mean(v):8.4f}" for v in macro.values()))
