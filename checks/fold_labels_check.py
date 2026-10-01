"""fold_labels_check.py - exactly which labels does each MedImageInsight fold run train and validate on?

For every runs/medimageinsight_davit_huge_336_v3_f{k}, reads the train/val CSVs its OWN config
resolves (through its train.py) and checks, row by row and finding by finding:
  1. the labels equal the v3 file 17_train_full_qwen-mg-raw.csv exactly
  2. the v3 file itself equals its recipe rebuilt from the sources: Qwen (03_report_labels_qwen24t-med)
     with MedGemma 1.5 (16_medgemma15_corpus_all) averaged in logit space on the MedGemma five
     (Effusion, Synovitis, Baker's, Lateral Meniscus, Lateral OA); Qwen alone on the other seven
  3. against the v2 labels (Qwen only): the other seven findings identical, the five changed
  4. study membership equals 13_fold{k} (the split the ConvNeXt folds used); train/val disjoint;
     the 5 val sets disjoint and together every non-audit study; no audit study anywhere

    python checks/fold_labels_check.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B5 = ["Effusion", "Synovitis", "Baker's", "Lateral Meniscus", "Lateral OA"]
O7 = [t for t in T if t not in B5]
bad = 0


def check(ok, msg):
    global bad
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'}  {msg}")


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


v3 = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
Q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID")
M = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID")
v2 = pd.read_csv(D / "08_train_full.csv").set_index("StudyInstanceUID")
audit = set(pd.read_csv(D / "05_audit58.csv").StudyInstanceUID)
nonaud = [s for s in v3.index if s not in audit]

print("v3 file vs its recipe (non-audit studies):")
rebuilt = Q.loc[nonaud, T].astype(float).copy()
rebuilt[B5] = 1 / (1 + np.exp(-(lg(Q.loc[nonaud, B5].values) + lg(M.loc[nonaud, B5].values)) / 2))
d = np.abs(rebuilt[T].values - v3.loc[nonaud, T].values)
check(d.max() < 1e-4, f"v3 = Qwen, with MedGemma averaged in on the five: max |diff| {d.max():.2e} over {len(nonaud)} studies")
d2 = np.abs(v3.loc[nonaud, O7].values - v2.loc[nonaud, O7].values).max()
d5 = np.abs(v3.loc[nonaud, B5].values - v2.loc[nonaud, B5].values).mean()
check(d2 == 0, f"other seven findings identical to v2 (max |diff| {d2:.2e})")
check(d5 > 0.05, f"MedGemma five changed from v2 (mean |diff| {d5:.3f})")

sys.path.insert(0, str(ROOT / "runs"))
vals = []
for k in range(5):
    run = f"medimageinsight_davit_huge_336_v3_f{k}"
    spec = importlib.util.spec_from_file_location(f"t{k}", ROOT / "runs" / run / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    cfg, sc = tm.cfg, tm.sc
    print(f"\n{run}  (experiment.name = {cfg['experiment']['name']})")
    check(cfg["experiment"]["name"] == run, "loads its own config")
    check(cfg["input"].get("laterality_flip") == "left", "left knees mirrored (flip on)")
    parts = {}
    for part, key in (("train", "train_csv"), ("val", "val_csv")):
        p = Path(sc.resolve(cfg["paths"][key], cfg))
        df = pd.read_csv(p).set_index("StudyInstanceUID")
        parts[part] = df
        check(p.name == f"21_fold{k}_{part}.csv", f"{part}: reads {p.name}")
        ids = list(df.index)
        dv = np.abs(df[T].values - v3.loc[ids, T].values).max()
        check(dv == 0, f"{part}: {len(ids)} rows, labels == v3 file exactly (max |diff| {dv:.1e})")
        check(not np.isnan(df[T].values).any() and df[T].values.min() >= 0 and df[T].values.max() <= 1,
              f"{part}: no NaN, all in [0, 1]")
        old = list(pd.read_csv(D / f"13_fold{k}_{part}.csv").StudyInstanceUID)
        check(ids == old, f"{part}: same studies, same order as 13_fold{k}_{part}")
        check(not (set(ids) & audit), f"{part}: no audit study")
    check(not (set(parts["train"].index) & set(parts["val"].index)), "train and val disjoint")
    vals.append(set(parts["val"].index))
print()
union = set().union(*vals)
check(sum(map(len, vals)) == len(union) and union == set(nonaud),
      f"5 val sets disjoint, together {len(union)} = every non-audit study")
print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
