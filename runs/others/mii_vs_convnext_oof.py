"""mii_vs_convnext_oof.py - whose out-of-fold predictions make the better teacher labels?

Two 5-fold sets on the same studies (13_fold{k} = 21_fold{k} membership):
  ConvNeXt  convnext_base_25d_18slices_f{k}       (trained on v2 labels)
  MII       medimageinsight_davit_huge_336_v3_f{k} (trained on v3 labels)
  ViT       dinov2_vits14_25d_f{k}                 (DINOv2 ViT-S/14, v2 labels)
Each at its own best-val checkpoint.

1. audit-58 (panel labels, in no fold, so every fold's audit prediction is blind), the judge
   used to pick the 18_* labels: v3 labels blended 0.7 / 0.3 with the teacher's logits.
   Per fold (as 18_make_oof_labels.py reported it) and with the 5-fold mean logit; also both
   teachers together, and each teacher alone. Paired bootstrap MII vs ConvNeXt.
2. the 4,342 out-of-fold studies against v2 and v3 labels. Biased: each family trained on one
   of those label sets, so read it only as a sanity check.

    python runs/others/mii_vs_convnext_oof.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FAM = {"ConvNeXt": "convnext_base_25d_18slices_f{}", "MII": "medimageinsight_davit_huge_336_v3_f{}",
       "ViT": "dinov2_vits14_25d_f{}"}
W = 0.3


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def mac(Y, P):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


def load(run, split):
    f = sw.best_npz(run, split)
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T), f.stem


def main():
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = (aud.loc[u, ["official_" + t for t in T]].values >= 0.5).astype(int)
    v3 = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
    v2 = pd.read_csv(D / "08_train_full.csv").set_index("StudyInstanceUID")
    # v3 recipe on the audit studies (the full-data file holds the panel labels there)
    q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, T].values
    mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, T].values
    bi = [T.index(t) for t in ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]]
    c = q.copy()
    c[:, bi] = sig(0.5 * lg(q[:, bi]) + 0.5 * lg(mg[:, bi]))
    base = mac(Y, c)

    A, O = {}, {}
    for fam, pat in FAM.items():
        A[fam], parts = [], []
        for k in range(5):
            a, sa = load(pat.format(k), "audit")
            o, so = load(pat.format(k), "val")
            held = pd.read_csv(D / f"13_fold{k}_val.csv").StudyInstanceUID.astype(str).tolist()
            assert set(o.index) == set(held), f"{pat.format(k)} val != its held-out fold"
            A[fam].append(a.loc[u].values)
            parts.append(o)
            print(f"  {pat.format(k):40s} {so:22s} {sa}")
        O[fam] = pd.concat(parts)
        assert O[fam].index.is_unique and len(O[fam]) == 4342

    print(f"\n1. audit-58 (panel labels). v3 labels alone: {base:.4f}")
    print(f"   {'teacher':22s} {'per fold (min-max, mean)':34s} {'5-fold mean logit':>18s} {'teacher alone':>14s}")
    blend = {}
    for fam in FAM:
        per = [mac(Y, sig((1 - W) * lg(c) + W * a)) for a in A[fam]]
        m = np.mean(A[fam], 0)
        blend[fam] = sig((1 - W) * lg(c) + W * m)
        print(f"   {fam:22s} {min(per):.4f}-{max(per):.4f} ({np.mean(per):.4f} {np.mean(per) - base:+.4f})"
              f"   {mac(Y, blend[fam]):.4f} ({mac(Y, blend[fam]) - base:+.4f})   {mac(Y, m):.4f}")
    # every combination of two or more families, equal weight per family
    from itertools import combinations
    for n in range(2, len(FAM) + 1):
        for combo in combinations(FAM, n):
            m = np.mean([np.mean(A[f], 0) for f in combo], 0)
            key = " + ".join(combo)
            blend[key] = sig((1 - W) * lg(c) + W * m)
            print(f"   {key:30s} {'':26s} {mac(Y, blend[key]):.4f} ({mac(Y, blend[key]) - base:+.4f})"
                  f"   {mac(Y, m):.4f}")

    rng = np.random.default_rng(0)
    pairs = [("MII", "ConvNeXt"), ("ConvNeXt + MII", "ConvNeXt"), ("ConvNeXt + MII", "MII")]
    if "ViT" in FAM:
        pairs += [("ConvNeXt + MII + ViT", "ConvNeXt + MII"), ("ConvNeXt + MII + ViT", "ConvNeXt")]
    for x, y in pairs:
        d = []
        for _ in range(2000):
            i = rng.integers(0, len(u), len(u))
            if (Y[i].sum(0) == 0).any() or (Y[i].sum(0) == len(i)).any():
                continue
            d.append(mac(Y[i], blend[x][i]) - mac(Y[i], blend[y][i]))
        d = np.array(d)
        print(f"   bootstrap {x} minus {y}: {d.mean():+.4f}, 95% CI [{np.percentile(d, 2.5):+.4f}, "
              f"{np.percentile(d, 97.5):+.4f}], P(>0) {100 * (d > 0).mean():.0f}%")

    print("\n2. out-of-fold, 4,342 studies (biased toward the label set each family trained on)")
    ids = O["MII"].index
    for name, lab in (("v2 labels (ConvNeXt's)", v2), ("v3 labels (MII's)", v3)):
        Yl = (lab.loc[ids, T].values >= 0.5).astype(int)
        r = {f: mac(Yl, O[f].loc[ids].values) for f in FAM}
        r["all"] = mac(Yl, np.mean([O[f].loc[ids].values for f in FAM], 0))
        print(f"   vs {name:24s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
