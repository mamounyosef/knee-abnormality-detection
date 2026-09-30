"""oof_seedfix_ensemble_test.py - does convnext_base_25d_18slices_oof_seedfix help the ensemble?

Stand-in for the 0.938 ten: every member's SPLIT-data version (val and audit never trained on).
convnext_base_full_all_knees_same_orientation has no split version with the flip, so its
unflipped split parent convnext_base_relabel stands in. Candidates are added (11th member) or
swapped in for each member in turn. Flat mean of logits, like the shipped ensemble.

Truth is FIXED per split so runs trained on different label sets are comparable:
  val (704)  against the v2 labels (05_val_qwen24t-med) AND the v3 labels (17_val_qwen-mg-raw)
  audit (58) against the expert panel (the leaderboard's kind of label)
Paired bootstrap over studies (2000 resamples) for every change vs the ten.

    python runs/others/ensembling/oof_seedfix_ensemble_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_new_runs as sw  # noqa: E402  (best_npz: each run's best-val prediction file)

ROOT = Path(__file__).resolve().parents[3]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TEN = ["convnext_base_relabel", "mricore_vitb_cross_attention", "mricore_vitb_mae_cross_attention",
       "dinov3_convnext_base", "convnext_base_25d_all_knees_same_orientation", "swinv2_base_in22k",
       "convnext_base_25d_poolattn", "convnext_base_25d_perfinding_slots_and_pooling",
       "medimageinsight_davit_huge_336", "medimageinsight_davit_huge_perslice_480_warm"]
CANDS = {"oof_seedfix (LOCAL, oof labels)": "convnext_base_25d_18slices_oof_seedfix",
         "localcheck (LOCAL, v3)": "convnext_base_25d_18slices_mglabels_raw_localcheck",
         "mglabels_raw (Modal, v3)": "convnext_base_25d_18slices_mglabels_raw",
         "18slices (Modal, v2) control": "convnext_base_25d_18slices"}
BOOT = 2000


def truth(split):
    if split == "audit":
        a = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
        return {"panel": (a[["official_" + t for t in T]].values >= 0.5).astype(int), "ids": a.index}
    out = {}
    for name, f in (("v2", "05_val_qwen24t-med.csv"), ("v3", "17_val_qwen-mg-raw.csv")):
        d = pd.read_csv(ROOT / "data" / f).set_index("StudyInstanceUID")
        out[name] = d
    return out


def logits(run, split, ids):
    f = sw.best_npz(run, split)
    z = np.load(f, allow_pickle=True)
    df = pd.DataFrame(z["logits"].astype(float), index=[str(s) for s in z["study_ids"]])
    return df.loc[ids].values


def aucs(p, y):
    from scipy.stats import rankdata
    r = rankdata(p, axis=0)
    n1 = y.sum(0).astype(float)
    n0 = len(y) - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        a = ((r * y).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    a[(n1 == 0) | (n0 == 0)] = np.nan
    return a


def main():
    rng = np.random.default_rng(0)
    for split in ("val", "audit"):
        if split == "audit":
            t = truth("audit")
            ids = [str(s) for s in t["ids"]]
            ys = {"panel": t["panel"]}
        else:
            t = truth("val")
            ids = [str(s) for s in t["v2"].index]
            ys = {k: (d.loc[ids, T].values >= 0.5).astype(int) for k, d in t.items()}
        L = {m: logits(m, split, ids) for m in TEN + list(CANDS.values())}
        ten = np.mean([L[m] for m in TEN], axis=0)
        variants = {"THE TEN (stand-in)": ten}
        for cn, cr in CANDS.items():
            variants[f"+ {cn}"] = np.mean([L[m] for m in TEN] + [L[cr]], axis=0)
            best = None
            for m in TEN:
                sw_ = np.mean([L[x] for x in TEN if x != m] + [L[cr]], axis=0)
                s = np.nanmean(aucs(sw_, ys[list(ys)[0]]))
                if best is None or s > best[0]:
                    best = (s, m, sw_)
            variants[f"{cn} REPLACES convnext_base_25d_all_knees"] = np.mean(
                [L[x] for x in TEN if x != "convnext_base_25d_all_knees_same_orientation"] + [L[cr]], axis=0)
            variants[f"{cn} replaces best member ({best[1]}) [in-sample pick]"] = best[2]
        variants["+ oof_seedfix + localcheck"] = np.mean(
            [L[m] for m in TEN] + [L[CANDS["oof_seedfix (LOCAL, oof labels)"]], L[CANDS["localcheck (LOCAL, v3)"]]], axis=0)
        n = len(ids)
        for yn, y in ys.items():
            base = np.nanmean(aucs(ten, y))
            print(f"\n=== {split.upper()} ({n} studies) vs {yn} labels: the ten {base:.4f} ===")
            print("  solo:", ", ".join(f"{cn.split(' ')[0]} {np.nanmean(aucs(L[cr], y)):.4f}" for cn, cr in CANDS.items()))
            for vn, P in variants.items():
                if vn.startswith("THE TEN"):
                    continue
                s = np.nanmean(aucs(P, y))
                g = []
                for _ in range(BOOT):
                    i = rng.integers(0, n, n)
                    if (y[i].sum(0) == 0).any() or (y[i].sum(0) == n).any():
                        continue
                    g.append(np.nanmean(aucs(P[i], y[i])) - np.nanmean(aucs(ten[i], y[i])))
                g = np.array(g)
                print(f"  {vn:78s} {s:.4f} ({s - base:+.4f})  CI [{np.percentile(g, 2.5):+.4f}, "
                      f"{np.percentile(g, 97.5):+.4f}]  P>0 {100 * (g > 0).mean():.0f}%")


if __name__ == "__main__":
    main()
