"""sex_signal.py - does PatientSex (kept in the DICOM headers, never used) add anything beyond the image models? (8 Oct)

PatientSex per study from data/stats/patient_ids.csv (first series' header). Also field strength / vendor as controls.
1. prevalence per finding by sex (v3 labels, 4,342 fold studies; panel labels on audit-58)
2. incremental value: per finding, logistic regression label ~ model logit (+ sex), 5-fold CV on the 4,342
   out-of-fold studies, graded by v3 labels (soft label >= 0.5). Model logit = mean of the round-2 ConvNeXt OOF and
   the MII OOF (each study predicted only by the fold that held it out).
3. audit-58: the same two regressions fit on the 4,342, applied to the 5-fold-mean audit logits, panel labels.
4. is sex already visible to the model? AUROC of the model's penultimate-free proxy: how well the OOF logits
   (12-dim) predict sex (logistic, CV). High = the image already encodes it, little left to add.

    python runs/others/sex_signal.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "data"
src2 = (HERE / "round2_teacher_audit.py").read_text(encoding="utf-8")
g2 = {"__file__": str(HERE / "round2_teacher_audit.py"), "__name__": "r2_audit"}
exec(compile(src2[:src2.index("\ndef logit(p):")], "round2_teacher_audit.py", "exec"), g2)
best_npz, T = g2["best_npz"], g2["TASKS"]


def npz_df(f):
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T)


def fam(pat, split):
    parts = [npz_df(best_npz(pat.format(k), split)) for k in range(5)]
    return pd.concat(parts) if split == "val" else sum(parts) / 5


CNX, MII = "convnext_base_25d_18slices_v4_flip_f{}", "medimageinsight_davit_huge_336_v3_f{}"
oof = (fam(CNX, "val") + fam(MII, "val").loc[fam(CNX, "val").index]) / 2
aud_p = (fam(CNX, "audit") + fam(MII, "audit")) / 2

pid = pd.read_csv(D / "stats/patient_ids.csv").drop_duplicates("study").set_index("study")
sex = pid.patient_sex.map({"F": 0.0, "M": 1.0})
print("sex values:", pid.patient_sex.value_counts(dropna=False).to_dict())
v3 = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")

ids = [s for s in oof.index if pd.notna(sex.get(s))]
au = [s for s in aud.index if pd.notna(sex.get(s))]
print(f"fold studies with sex {len(ids)}/{len(oof)}, audit {len(au)}/58")
Y = (v3.loc[ids, T].values >= 0.5).astype(int)
Ya = (aud.loc[au, ["official_" + t for t in T]].values >= 0.5).astype(int)
S, Sa = sex.loc[ids].values, sex.loc[au].values

print("\n1. prevalence (v3 >= 0.5) by sex, and sex-alone AUROC")
print(f"   {'finding':17s} {'F':>6s} {'M':>6s}  AUROC(sex)  audit F/M")
for j, t in enumerate(T):
    pf, pm = Y[S == 0, j].mean(), Y[S == 1, j].mean()
    a = roc_auc_score(Y[:, j], S)
    af = Ya[Sa == 0, j].mean() if (Sa == 0).any() else np.nan
    am = Ya[Sa == 1, j].mean() if (Sa == 1).any() else np.nan
    print(f"   {t:17s} {pf:6.3f} {pm:6.3f}  {a:.3f}      {af:.2f}/{am:.2f}")

print("\n2. 5-fold CV on the 4,342 (v3 labels): AUROC model-only vs model + sex")
cv = StratifiedKFold(5, shuffle=True, random_state=0)
gain, rows = [], []
X0 = oof.loc[ids, T].values
for j, t in enumerate(T):
    a = X0[:, [j]]
    b = np.column_stack([X0[:, j], S])
    pa = cross_val_predict(LogisticRegression(C=1.0), a, Y[:, j], cv=cv, method="decision_function")
    pb = cross_val_predict(LogisticRegression(C=1.0), b, Y[:, j], cv=cv, method="decision_function")
    ra, rb = roc_auc_score(Y[:, j], pa), roc_auc_score(Y[:, j], pb)
    gain.append(rb - ra)
    # fit on all 4,342, apply to audit (panel)
    ma = LogisticRegression(C=1.0).fit(a, Y[:, j]); mb = LogisticRegression(C=1.0).fit(b, Y[:, j])
    xa = aud_p.loc[au, T].values
    if 0 < Ya[:, j].sum() < len(au):
        qa = roc_auc_score(Ya[:, j], ma.decision_function(xa[:, [j]]))
        qb = roc_auc_score(Ya[:, j], mb.decision_function(np.column_stack([xa[:, j], Sa])))
    else:
        qa = qb = np.nan
    rows.append((t, ra, rb, qa, qb, mb.coef_[0][1]))
    print(f"   {t:17s} {ra:.4f} -> {rb:.4f} ({rb - ra:+.4f})   audit {qa:.4f} -> {qb:.4f} ({qb - qa:+.4f})"
          f"   sex coef {mb.coef_[0][1]:+.2f}")
r = np.array([x[1:5] for x in rows], float)
print(f"   macro            {r[:, 0].mean():.4f} -> {r[:, 1].mean():.4f} ({r[:, 1].mean() - r[:, 0].mean():+.4f})"
      f"   audit {np.nanmean(r[:, 2]):.4f} -> {np.nanmean(r[:, 3]):.4f} ({np.nanmean(r[:, 3]) - np.nanmean(r[:, 2]):+.4f})")

print("\n4. can the 12 model logits predict sex? (CV AUROC; 0.5 = sex invisible to the model)")
ps = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), X0, S.astype(int), cv=cv, method="decision_function")
print(f"   {roc_auc_score(S, ps):.3f}")
