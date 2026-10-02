"""relabel_batch_plan.py - which ensemble members matter most, and what each costs to retrain.

INFLUENCE: leave-one-out on the current 14 (the 0.938 ten's nine non-MII members + the 5 MII folds),
each full-data member replaced by its split-data analogue (ten_mii_folds_audit.TEN_SPLIT), which never
saw val or audit. Drop = ensemble score without the member minus with it (negative = it helps).
  val   : 704 studies, v3 LLM labels (17_val_qwen-mg-raw) as the one target; the MII folds are replaced by the split MII (folds trained on val)
  audit : 58 studies, panel labels, the actual 14 with the folds
COST: from each SHIPPED full-data run's own training_summary.json and config (labels, flip, GPU, time).

    python runs/others/relabel_batch_plan.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402
from ten_mii_folds_audit import TEN_SPLIT  # noqa: E402

ROOT = HERE.parent.parent
D = ROOT / "data"
RUNS = ROOT / "runs"
sys.path.insert(0, str(RUNS))
import shared_code as sc  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
NINE = [k for k in TEN_SPLIT if k != "medimageinsight_davit_huge_336_fulldata"]
FOLDS = [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]


def preds(run, split, ids):
    z = np.load(sw.best_npz(run, split), allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[ids].values


def macro(Y, P):
    return np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12)])


aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
ua = list(aud.index)
Ya = (aud.loc[ua, ["official_" + t for t in T]].values >= 0.5).astype(int)
val = pd.read_csv(D / "17_val_qwen-mg-raw.csv").set_index("StudyInstanceUID")   # v3 labels, one target for all
uv = list(val.index)
Yv = (val.loc[uv, T].values >= 0.5).astype(int)

A = {n: preds(TEN_SPLIT[n], "audit", ua) for n in NINE}
A.update({f: preds(f, "audit", ua) for f in FOLDS})
V = {n: preds(TEN_SPLIT[n], "val", uv) for n in NINE}
V["mii_split"] = preds("medimageinsight_davit_huge_336", "val", uv)
amem, vmem = NINE + FOLDS, NINE + ["mii_split"] * 5      # 14 votes in both, the MII family 5 of them
fa = macro(Ya, np.mean([A[n] for n in amem], 0))
fv = macro(Yv, np.mean([V[n] for n in vmem], 0))
print(f"all 14 (split analogues): val {fv:.4f}  audit {fa:.4f}\n")

rows = []
for n in NINE:
    da = macro(Ya, np.mean([A[x] for x in amem if x != n], 0)) - fa
    dv = macro(Yv, np.mean([V[x] for x in vmem if x != n], 0)) - fv
    cfg = sc.load_config(RUNS / n, verbose=False)
    sj = RUNS / n / "results" / "training_summary.json"
    s = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    hw = s.get("hardware") or {}
    rows.append({"member": n, "val_drop": dv, "audit_drop": da,
                 "labels": Path(str(cfg["paths"].get("train_csv"))).name,
                 "flip": cfg["input"].get("laterality_flip"),
                 "size": cfg["input"].get("size"), "slices": cfg["input"].get("n_slices"),
                 "as_ch": cfg["input"].get("slices_as_channels"),
                 "gpu": cfg.get("modal", {}).get("gpu"), "epochs": cfg["training"].get("epochs"),
                 "elapsed": s.get("elapsed"), "hw": hw.get("gpu") or hw.get("device") or hw})
df = pd.DataFrame(rows).sort_values("val_drop")
pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60)
print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
