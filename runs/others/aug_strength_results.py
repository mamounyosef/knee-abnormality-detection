"""aug_strength_results.py - which augmentation won? (convnext_base_25d_18slices_v5_aug_base / _strong / _strong_extra)

Same v5 val for all three (404 studies: 346 on v3 labels + the 58 audit on panel labels), so the
comparison is paired. Per arm: best val, its step and epoch, whether training stopped early or ran
out, audit-58 alone at that step, and the end of the curve (was it still climbing?). Then each arm
against the baseline at their own best checkpoints: paired bootstrap over the 404 val studies, the
346 alone and the 58 alone, plus per-finding val.

    python runs/others/aug_strength_results.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
sys.path.insert(0, str(HERE / "ensembling"))
import sweep_new_runs as sw  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ARMS = ["convnext_base_25d_18slices_v5_aug_base", "convnext_base_25d_18slices_v5_aug_strong",
        "convnext_base_25d_18slices_v5_aug_strong_extra"]
B = 2000
val = pd.read_csv(RUNS.parent / "data" / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
ids = list(val.index)
Y = (val.loc[ids, T].values > 0.5).astype(int)
src = val.loc[ids, "source"].values


def macro(y, p):
    return np.nanmean([roc_auc_score(y[:, j], p[:, j]) if 0 < y[:, j].sum() < len(y) else np.nan for j in range(12)])


P, rows = {}, []
for a in ARMS:
    d = RUNS / a / "results"
    v = pd.read_csv(d / "val_log.csv").drop_duplicates("step", keep="last").sort_values("step")
    i = v.val_macro_auroc.idxmax()
    f = sw.best_npz(a, "val")
    z = np.load(f, allow_pickle=True)
    P[a] = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[ids].values
    cap = (RUNS / a / "results" / "train_config")
    stop = "early stop" if any("early stop:" in t.read_text(encoding="utf-8", errors="replace")
                               for t in cap.glob("*.txt")) else "ran to the end"
    tail = v.val_macro_auroc.values[-3:]
    rows.append({"arm": a.replace("convnext_base_25d_18slices_v5_aug_", ""), "best val": v.loc[i, "val_macro_auroc"],
                 "step": int(v.loc[i, "step"]), "epoch": int(v.loc[i, "epoch"]),
                 "audit@best": v.loc[i, "audit_macro_auroc"], "best audit": v.audit_macro_auroc.max(),
                 "last step": int(v.step.max()), "last epoch": int(v.epoch.max()), "stopped": stop,
                 "last 3 val": " ".join(f"{x:.4f}" for x in tail), "npz": Path(f).name})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
for a in ARMS:
    m = macro(Y, P[a])
    assert abs(m - df.loc[ARMS.index(a), "best val"]) < 2e-3, f"{a}: rescored {m:.4f} != logged"

rng = np.random.default_rng(0)
base = ARMS[0]
print("\nvs the baseline, paired bootstrap at each arm's best checkpoint (2,000 resamples):")
for a in ARMS[1:]:
    for lab, sel in (("all 404", np.ones(len(ids), bool)), ("346 v3-labelled", src == "val"), ("58 audit", src == "audit")):
        idx = np.where(sel)[0]
        obs = macro(Y[idx], P[a][idx]) - macro(Y[idx], P[base][idx])
        d = []
        for _ in range(B):
            r = rng.choice(idx, len(idx))
            d.append(macro(Y[r], P[a][r]) - macro(Y[r], P[base][r]))
        d = np.array(d)
        print(f"  {a[-12:]:>12s} {lab:16s} {obs:+.4f}   90% interval [{np.nanpercentile(d, 5):+.4f}, {np.nanpercentile(d, 95):+.4f}]"
              f"   better in {np.nanmean(d > 0):.0%}")
print("\nper finding, all 404 val (base | strong | strong_extra):")
for j, t in enumerate(T):
    s = [roc_auc_score(Y[:, j], P[a][:, j]) for a in ARMS]
    print(f"  {t:17s} {s[0]:.4f} | {s[1]:.4f} ({s[1] - s[0]:+.4f}) | {s[2]:.4f} ({s[2] - s[0]:+.4f})")
