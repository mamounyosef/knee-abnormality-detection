"""
22_make_convnext_mii_labels.py - blend TWO out-of-fold teachers into the v3 labels.

    new logit = 0.7 * logit(v3 label) + 0.3 * (0.5 * ConvNeXt OOF logit + 0.5 * MII OOF logit)

v3 label = data/17_*_qwen-mg-raw.csv. Each teacher's logit is the ONE fold model that did not
train on that study, at its own best-val checkpoint:
  ConvNeXt  convnext_base_25d_18slices_f{k}        (13_fold{k}, v2 labels), the 18_* teacher
  MII       medimageinsight_davit_huge_336_v3_f{k} (21_fold{k} = same studies, v3 labels)
The 58 audit studies are in no fold and keep the panel's labels, exactly as in 17_* and 18_*.

WHY (audit-58, 1 Oct, runs/others/mii_vs_convnext_oof.py): v3 alone 0.9311; + ConvNeXt 0.9400
(the 18_* recipe, twin LB 0.922); + MII 0.9397; + both half each 0.9414. The DINOv2 folds as a
third teacher hurt (0.9404). Weight 0.3 and the 50/50 split kept from 18_*:
runs/others/teacher_blend_search.py found no tuned setting that beats them out of sample
(global +0.0007, 64% of splits; per-class -0.0001).

    python data_code/22_make_convnext_mii_labels.py --w 0.3   # 22_*-w30: v3 70%, teachers 15% each
    python data_code/22_make_convnext_mii_labels.py --w 0.5   # 22_*-w50: v3 50%, teachers 25% each
    python data_code/22_make_convnext_mii_labels.py --w 0.6   # 22_*-w60: v3 40%, teachers 30% each (top of the label curve)
(audit-58: w=0.3 0.9414, w=0.5 0.9453 in sample; both trained so the leaderboard decides)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "runs/others/ensembling"))
import sweep_new_runs as sw  # noqa: E402

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
import argparse  # noqa: E402
_ap = argparse.ArgumentParser()
_ap.add_argument("--w", type=float, required=True)
W = _ap.parse_args().w
EXPECTED = {0.3: 0.9414, 0.5: 0.9453, 0.6: 0.9458}[W]   # audit-58 of the blended labels
_SUF = f"w{round(W * 100)}"
TEACHERS = {"ConvNeXt": ("convnext_base_25d_18slices_f{}", "13_fold{}_val.csv"),
            "MII": ("medimageinsight_davit_huge_336_v3_f{}", "21_fold{}_val.csv")}
OUT = {"17_train_qwen-mg-raw.csv": f"22_train_qwen-mg-convnext-mii-{_SUF}.csv",
       "17_val_qwen-mg-raw.csv": f"22_val_qwen-mg-convnext-mii-{_SUF}.csv",
       "17_train_full_qwen-mg-raw.csv": f"22_train_full_qwen-mg-convnext-mii-{_SUF}.csv"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    fails = []

    def check(cond, msg):
        print(("  OK    " if cond else "  FAIL  ") + msg)
        if not cond:
            fails.append(msg)

    print("out-of-fold predictions:")
    asg = pd.read_csv(D / "13_fold_assignment.csv")
    per_t, aud_t = {}, {}
    for t, (pat, val_csv) in TEACHERS.items():
        oof, audit_preds = [], []
        for k in range(5):
            run = pat.format(k)
            f = sw.best_npz(run, "val")
            z = np.load(f, allow_pickle=True)
            ids = [str(s) for s in z["study_ids"]]
            held = pd.read_csv(D / val_csv.format(k)).StudyInstanceUID.tolist()
            check(set(ids) == set(held) and len(ids) == len(held),
                  f"{run}: {Path(f).name} covers exactly its held-out fold ({len(ids)} studies)")
            check(set(held) == set(pd.read_csv(D / f"13_fold{k}_val.csv").StudyInstanceUID),
                  f"{run}: its held-out fold is 13_fold{k}'s")
            check(list(z["logits"].shape) == [len(ids), 12] and np.isfinite(z["logits"]).all(),
                  f"{run}: 12 finite logits per study")
            oof.append(pd.DataFrame(z["logits"], index=ids, columns=TASKS))
            a = np.load(sw.best_npz(run, "audit"), allow_pickle=True)
            audit_preds.append(pd.DataFrame(a["logits"], index=[str(s) for s in a["study_ids"]],
                                            columns=TASKS))
        o = pd.concat(oof)
        check(o.index.is_unique and set(o.index) == set(asg.StudyInstanceUID),
              f"{t}: one prediction per train+val study ({len(o)})")
        per_t[t], aud_t[t] = o, audit_preds
    O = 0.5 * per_t["ConvNeXt"] + 0.5 * per_t["MII"].loc[per_t["ConvNeXt"].index]
    check(np.isfinite(O.values).all(), f"two-teacher mean finite ({len(O)} studies)")

    cur = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
    llm_ids = cur.index[cur["source"] != "audit"]
    check(set(llm_ids) == set(O.index), f"every LLM-labelled study ({len(llm_ids)}) has one OOF prediction")
    new = cur.copy()
    new.loc[llm_ids, TASKS] = sigmoid((1 - W) * logit(cur.loc[llm_ids, TASKS].values)
                                      + W * O.loc[llm_ids, TASKS].values)
    v = new.loc[llm_ids, TASKS].values
    check(np.isfinite(v).all() and (v > 0).all() and (v < 1).all(), "new labels finite, inside (0, 1)")
    for t in TASKS:
        r = pd.Series(new.loc[llm_ids, t]).corr(cur.loc[llm_ids, t], method="spearman")
        check(0.6 < r < 0.9999, f"{t:17s} rank correlation with the current labels {r:.3f} "
                                f"(mean {cur.loc[llm_ids, t].mean():.3f} -> {new.loc[llm_ids, t].mean():.3f})")

    # the audit estimate, as far as it can be made: the 58 are in no fold, so each fold model's
    # blind audit prediction stands in for the one-model OOF prediction every training study gets
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = aud.loc[u, ["official_" + t for t in TASKS]].values
    q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, TASKS].values
    mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, TASKS].values
    B = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
    bi = [TASKS.index(t) for t in B]
    c = q.copy()
    c[:, bi] = sigmoid(0.5 * logit(q[:, bi]) + 0.5 * logit(mg[:, bi]))
    mac = lambda P: np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12)])   # noqa: E731
    two = 0.5 * np.mean([a.loc[u].values for a in aud_t["ConvNeXt"]], 0) \
        + 0.5 * np.mean([a.loc[u].values for a in aud_t["MII"]], 0)
    check(abs(mac(c) - 0.9311) < 5e-4, f"audit-58 current labels {mac(c):.4f} (expected 0.9311)")
    got = mac(sigmoid((1 - W) * logit(c) + W * two))
    check(abs(got - EXPECTED) < 5e-4, f"audit-58 with both teachers blended in at w={W}: {got:.4f} (expected {EXPECTED})")

    print("\nwriting the training files (same rows, columns and order as the 17_* ones):")
    for src, dst in OUT.items():
        s = pd.read_csv(D / src)
        ids = s["StudyInstanceUID"]
        llm = (s["source"] != "audit").values if "source" in s else np.ones(len(s), bool)
        o = s.copy()
        o.loc[llm, TASKS] = new.loc[ids[llm], TASKS].values
        o.to_csv(D / dst, index=False)
        back = pd.read_csv(D / dst)
        check(list(back.columns) == list(s.columns) and len(back) == len(s)
              and (back["StudyInstanceUID"] == ids).all(), f"{dst}: {len(back)} rows, same columns and order")
        check(np.allclose(back.loc[llm, TASKS].values, new.loc[ids[llm], TASKS].values),
              f"{dst}: the {llm.sum()} LLM-labelled rows carry the blended labels")
        other = [c_ for c_ in s.columns if c_ not in TASKS]
        check(back[other].astype(str).equals(s[other].astype(str)), f"{dst}: non-label columns unchanged")
        if (~llm).any():
            check(np.array_equal(back.loc[~llm, TASKS].values, s.loc[~llm, TASKS].values),
                  f"{dst}: the {(~llm).sum()} audit rows keep the panel's labels exactly")
    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
