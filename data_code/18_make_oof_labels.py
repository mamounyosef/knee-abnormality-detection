"""
18_make_oof_labels.py - blend out-of-fold image-model predictions into the current labels.

    new logit = 0.7 * logit(current label) + 0.3 * OOF logit          (all 12 findings)

current label = data/17_*_qwen-mg-raw.csv (Qwen, with MedGemma 1.5 averaged in on five findings;
LB 0.916 for its twin). OOF logit = the prediction of the ONE convnext_base_25d_18slices fold model
that did not train on that study: fold k's model scored on data/13_fold{k}_val.csv at its own
best-val checkpoint (results/predictions/*_val_s<step>.npz). The five held-out sets are disjoint
and cover all 4,342 train + val studies; the 58 audit studies are in no fold and keep the panel's
own labels in the full-data file, exactly as in 17_*.

WHY (audit-58, 26 Sep): the fold models never saw the audit studies either, so their audit
predictions are the same kind of blind second opinion. Blending 30% of one fold model into the
current labels: 0.9383 to 0.9401 depending on the fold (current 0.9311), CI of the 5-fold average
[+0.0014, +0.0158]. 0.3 is the middle of the plateau (0.3-0.5), fixed before building. The ViT
folds were weaker (0.9377) and added nothing on top (0.9394); other labellers on top: nothing.

    python data_code/18_make_oof_labels.py
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
W = 0.3
FOLD_RUN = "convnext_base_25d_18slices_f{}"
OUT = {"17_train_qwen-mg-raw.csv": "18_train_qwen-mg-oof.csv",
       "17_val_qwen-mg-raw.csv": "18_val_qwen-mg-oof.csv",
       "17_train_full_qwen-mg-raw.csv": "18_train_full_qwen-mg-oof.csv"}


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
    oof, audit_preds = [], []
    for k in range(5):
        run = FOLD_RUN.format(k)
        f = sw.best_npz(run, "val")
        z = np.load(f, allow_pickle=True)
        ids = [str(s) for s in z["study_ids"]]
        held = pd.read_csv(D / f"13_fold{k}_val.csv").StudyInstanceUID.tolist()
        check(set(ids) == set(held) and len(ids) == len(held),
              f"{run}: {Path(f).name} covers exactly its held-out fold ({len(ids)} studies)")
        check(list(z["logits"].shape) == [len(ids), 12] and np.isfinite(z["logits"]).all(),
              f"{run}: 12 finite logits per study")
        oof.append(pd.DataFrame(z["logits"], index=ids, columns=TASKS))
        a = np.load(sw.best_npz(run, "audit"), allow_pickle=True)
        audit_preds.append(pd.DataFrame(a["logits"], index=[str(s) for s in a["study_ids"]], columns=TASKS))
    O = pd.concat(oof)
    check(O.index.is_unique, f"no study predicted twice ({len(O)} studies)")
    asg = pd.read_csv(D / "13_fold_assignment.csv")
    check(set(O.index) == set(asg.StudyInstanceUID), "the five folds together = every train+val study")

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
    per = [mac(sigmoid((1 - W) * logit(c) + W * a.loc[u].values)) for a in audit_preds]
    check(abs(mac(c) - 0.9311) < 5e-4, f"audit-58 current labels {mac(c):.4f} (expected 0.9311)")
    check(min(per) > mac(c), f"audit-58 with one fold model blended in: {min(per):.4f} to {max(per):.4f} "
                             f"(mean {np.mean(per):.4f}, {np.mean(per) - mac(c):+.4f})")

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
