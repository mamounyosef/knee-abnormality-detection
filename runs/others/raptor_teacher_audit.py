"""raptor_teacher_audit.py - do the 5 RAPTOR96 folds (13 Oct, v4 labels, flipped) make better labels?

Same audit-58 estimate as data_code/22_make_convnext_mii_labels.py and round2_teacher_audit.py: a teacher's audit
prediction is its fold mean (the 58 are in no fold), blended in logit space into the v3 report labels.
To beat: the CURRENT labels w50 = 0.5 v3 + 0.5 (old ConvNeXt + MII) / 2, audit-58 0.9453.
Tests
  teachers alone              old ConvNeXt, round-2 ConvNeXt (v4_flip), MII, raptor
  raptor as the only teacher  v3 + raptor at w
  raptor replaces a teacher   v3 + (raptor + MII) / 2, v3 + (raptor + old ConvNeXt) / 2
  raptor as a third teacher   v3 + (old ConvNeXt + MII + raptor) / 3
  on top of the current       logit(w50) + a * raptor (the current labels moved toward raptor)
Echo check: out-of-fold rank correlation with the v3 labels on all training studies (a teacher that only repeats v3
adds nothing). Paired bootstrap of the best variant against w50.

    python runs/others/raptor_teacher_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "runs/others"))
import round2_teacher_audit as r2  # noqa: E402  (prints its own report first; reused for its loaders)

TASKS = r2.TASKS
FAM = {"old ConvNeXt": "convnext_base_25d_18slices_f{}",
       "round-2 ConvNeXt": "convnext_base_25d_18slices_v4_flip_f{}",
       "MII": "medimageinsight_davit_huge_336_v3_f{}",
       "raptor": "convnext_base_v5_raptor96_f{}"}
logit, sigmoid, mac, per, Y, u, v3 = r2.logit, r2.sigmoid, r2.mac, r2.per, r2.Y, r2.u, r2.v3

print("\n" + "=" * 90 + "\nRAPTOR96 FOLDS AS A LABEL TEACHER\n" + "=" * 90)
A = {}
for name, pat in FAM.items():
    fold = []
    for k in range(5):
        f = r2.best_npz(pat.format(k), "audit")
        z = np.load(f, allow_pickle=True)
        fold.append(pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=TASKS).loc[u].values)
    A[name] = np.mean(fold, 0)
    if name == "raptor":
        print("  raptor fold files:", ", ".join(Path(r2.best_npz(pat.format(k), "audit")).name for k in range(5)))

cur = 0.5 * logit(v3) + 0.5 * (0.5 * A["old ConvNeXt"] + 0.5 * A["MII"])
CUR = mac(sigmoid(cur))
print(f"\naudit-58: v3 labels {mac(v3):.4f} | CURRENT w50 labels {CUR:.4f} (expected 0.9453)")
assert abs(CUR - 0.9453) < 5e-4
for n in FAM:
    print(f"  teacher alone, {n:17s} {mac(A[n]):.4f}")

V = {}
print(f"\n  {'w':>4}  {'v3+raptor':>9}  {'v3+(rap+MII)/2':>14}  {'v3+(rap+oldCNX)/2':>17}  {'v3+(old,MII,rap)/3':>18}"
      f"  {'v3+(old,MII,rap,MII)':>20}")
for w in (0.3, 0.4, 0.5, 0.6, 0.7):
    t = {"v3 + raptor": A["raptor"],
         "v3 + (raptor + MII)/2": 0.5 * A["raptor"] + 0.5 * A["MII"],
         "v3 + (raptor + old ConvNeXt)/2": 0.5 * A["raptor"] + 0.5 * A["old ConvNeXt"],
         "v3 + (old ConvNeXt + MII + raptor)/3": (A["old ConvNeXt"] + A["MII"] + A["raptor"]) / 3,
         "v3 + 0.25 old + 0.5 MII + 0.25 raptor": 0.25 * A["old ConvNeXt"] + 0.5 * A["MII"] + 0.25 * A["raptor"]}
    r = []
    for k, x in t.items():
        p = (1 - w) * logit(v3) + w * x
        V[f"{k} @w{w:.1f}"] = p
        r.append(mac(sigmoid(p)))
    print(f"  {w:4.1f}  {r[0]:9.4f}  {r[1]:14.4f}  {r[2]:17.4f}  {r[3]:18.4f}  {r[4]:20.4f}")

print("\n  current w50 labels moved toward raptor: (1 - a) logit(w50) + a raptor")
for a in (0.1, 0.2, 0.3, 0.4, 0.5):
    p = (1 - a) * cur + a * A["raptor"]
    V[f"w50 + raptor a{a:.1f}"] = p
    print(f"  a {a:.1f}  {mac(sigmoid(p)):.4f}  ({mac(sigmoid(p)) - CUR:+.4f})")

best = max(V, key=lambda k: mac(sigmoid(V[k])))
bp = sigmoid(V[best])
print(f"\nbest variant: {best}  {mac(bp):.4f} ({mac(bp) - CUR:+.4f} vs current w50)")
print("per finding, current -> best:")
for t, a, b in zip(TASKS, per(sigmoid(cur)), per(bp)):
    print(f"  {t:17s} {a:.4f} -> {b:.4f} ({b - a:+.4f})")
print("per finding, teacher alone: raptor | old ConvNeXt | MII | v3")
for t, a, b, m, l in zip(TASKS, per(A["raptor"]), per(A["old ConvNeXt"]), per(A["MII"]), per(v3)):
    print(f"  {t:17s} {a:.4f} | {b:.4f} | {m:.4f} | {l:.4f}")

rng = np.random.default_rng(0)
sc = sigmoid(cur)
for name in [best, "w50 + raptor a0.2", "v3 + (old ConvNeXt + MII + raptor)/3 @w0.5"]:
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(u), len(u))
        try:
            d.append(np.mean([roc_auc_score(Y[i, j], sigmoid(V[name])[i, j]) - roc_auc_score(Y[i, j], sc[i, j])
                              for j in range(12)]))
        except ValueError:
            continue
    d = np.array(d)
    print(f"bootstrap {name} - current: mean {d.mean():+.4f}, 90% [{np.percentile(d, 5):+.4f}, "
          f"{np.percentile(d, 95):+.4f}], better in {np.mean(d > 0):.0%}")

O = {n: r2.oof(p) for n, p in FAM.items()}
ids = [s for s in O["MII"].index if all(s in O[n].index for n in FAM)]
lab = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
ids = [s for s in ids if s in lab.index]
L = logit(lab.loc[ids, TASKS].values)
rk = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(12)])  # noqa: E731
print(f"\nout-of-fold, {len(ids)} training studies: mean rank correlation per finding")
for n in FAM:
    x = O[n].loc[ids].values
    print(f"  {n:17s} with v3 labels {rk(x, L):.3f}   with MII {rk(x, O['MII'].loc[ids].values):.3f}   "
          f"with old ConvNeXt {rk(x, O['old ConvNeXt'].loc[ids].values):.3f}")
