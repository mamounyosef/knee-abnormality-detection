"""
20_make_ens_labels.py - blend the 0.938 ensemble's own predictions into the current labels.

    new logit = 0.7 * logit(current label) + 0.3 * ensemble logit          (all 12 findings)

current label = data/17_*_qwen-mg-raw.csv (the labels of the LB-0.916 twin). Ensemble logit = the
flat mean of the ten members' logits (the ensemble's own recipe: ensemble.py WEIGHTS="flat",
COMBINE_SPACE="logit"), from data/19_member_preds/*.npz (19_ens_member_preds.py). The 58 audit
studies keep the panel's own labels in the full-data file, exactly as in 17_* and 18_*.

Unlike the out-of-fold labels (18_*), these predictions are IN-SAMPLE for the training studies:
every member trained on them (six on the audit studies too). That is the point of the test: does
the ensemble's opinion still help as a label when it has seen the study? The twin
convnext_base_25d_18slices_ens answers it on audit-58 and the leaderboard.

    python data_code/20_make_ens_labels.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
sys.path.insert(0, str(ROOT / "data_code"))
TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
W = 0.3
MEMBERS = ["convnext_base_full_all_knees_same_orientation", "mricore_vitb_cross_attention",
           "mricore_vitb_mae_cross_attention_fulldata_flip", "dinov3_convnext_base",
           "convnext_base_25d_all_knees_same_orientation_fulldata", "swinv2_base_in22k",
           "convnext_base_25d_poolattn_fulldata_flip",
           "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip",
           "medimageinsight_davit_huge_336_fulldata", "medimageinsight_davit_huge_perslice_480_warm"]
OUT = {"17_train_qwen-mg-raw.csv": "20_train_qwen-mg-ens.csv",
       "17_val_qwen-mg-raw.csv": "20_val_qwen-mg-ens.csv",
       "17_train_full_qwen-mg-raw.csv": "20_train_full_qwen-mg-ens.csv"}


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

    cur = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
    ids = list(cur.index)
    print("member predictions:")
    L = []
    for m in MEMBERS:
        f = D / "19_member_preds" / f"{m}.npz"
        check(f.exists(), f"{m}: predictions saved")
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        df = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=list(z["tasks"]))
        check(list(z["tasks"]) == TASKS, f"{m}: 12 tasks in the standard order")
        check(set(ids) <= set(df.index) and df.index.is_unique, f"{m}: covers all {len(ids)} studies once")
        check(np.isfinite(df.values).all(), f"{m}: all logits finite (step {int(z['step'])})")
        L.append(df.loc[ids, TASKS].values)
    if len(L) != len(MEMBERS):
        print("\nmissing members - stopping")
        return 1
    E = np.mean(L, axis=0)
    llm = (cur["source"] != "audit").values
    new = cur.copy()
    new.loc[llm, TASKS] = sigmoid((1 - W) * logit(cur.loc[llm, TASKS].values) + W * E[llm])
    v = new.loc[llm, TASKS].values
    check(np.isfinite(v).all() and (v > 0).all() and (v < 1).all(), "new labels finite, inside (0, 1)")
    for j, t in enumerate(TASKS):
        r = pd.Series(new.loc[llm, t].values).corr(pd.Series(cur.loc[llm, t].values), method="spearman")
        print(f"      {t:17s} rank corr with current {r:.3f}; mean {cur.loc[llm, t].mean():.3f} -> {new.loc[llm, t].mean():.3f}")

    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    u = list(aud.index)
    Y = aud.loc[u, ["official_" + t for t in TASKS]].values
    pos = [ids.index(x) for x in u]
    q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, TASKS].values
    mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, TASKS].values
    B = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
    bi = [TASKS.index(t) for t in B]
    c = q.copy()
    c[:, bi] = sigmoid(0.5 * logit(q[:, bi]) + 0.5 * logit(mg[:, bi]))
    mac = lambda P: np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12)])   # noqa: E731
    blend = sigmoid((1 - W) * logit(c) + W * E[pos])
    check(abs(mac(c) - 0.9311) < 5e-4, f"audit-58 current labels {mac(c):.4f} (expected 0.9311)")
    print(f"      audit-58 ensemble alone {mac(E[pos]):.4f}; current + 30% ensemble {mac(blend):.4f} "
          f"({mac(blend) - mac(c):+.4f}) - INFLATED: six members trained on these 58")

    print("\nwriting the training files (same rows, columns and order as the 17_* ones):")
    for src, dst in OUT.items():
        s = pd.read_csv(D / src)
        sid = s["StudyInstanceUID"]
        m_ = (s["source"] != "audit").values if "source" in s else np.ones(len(s), bool)
        o = s.copy()
        o.loc[m_, TASKS] = new.loc[sid[m_], TASKS].values
        o.to_csv(D / dst, index=False)
        back = pd.read_csv(D / dst)
        check(list(back.columns) == list(s.columns) and len(back) == len(s)
              and (back["StudyInstanceUID"] == sid).all(), f"{dst}: {len(back)} rows, same columns and order")
        check(np.allclose(back.loc[m_, TASKS].values, new.loc[sid[m_], TASKS].values),
              f"{dst}: the {m_.sum()} LLM-labelled rows carry the blended labels")
        other = [x for x in s.columns if x not in TASKS]
        check(back[other].astype(str).equals(s[other].astype(str)), f"{dst}: non-label columns unchanged")
        if (~m_).any():
            check(np.array_equal(back.loc[~m_, TASKS].values, s.loc[~m_, TASKS].values),
                  f"{dst}: the {(~m_).sum()} audit rows keep the panel's labels exactly")
    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
