"""swin_v5_swap_audit.py - does swapping swinv2_base_in22k for its v5 warm start (swinv2_base_in22k_v5_warm, best clean-121
step 2000) help the batch-2 ensemble (ref 56824782)? Audit-58, split analogues, same method as relabel_batch2_swap_audit.py.
CAVEAT: the 58 sit inside the clean-121 val the new SwinV2 early-stopped on, so its numbers are slightly optimistic.

    python runs/others/swin_v5_swap_audit.py
"""
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

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B1 = {"dinov3_convnext_base": "dinov3_convnext_base_v5",
      "convnext_base_relabel": "convnext_base_v5",
      "convnext_base_25d_all_knees_same_orientation": "convnext_base_25d_18slices_v5_aug_base",
      "convnext_base_25d_poolattn": "convnext_base_25d_poolattn_v5"}
B1.update({"mricore_vitb_cross_attention": "mricore_vitb_cross_attention_v5",
           "mricore_vitb_mae_cross_attention": "mricore_vitb_mae_cross_attention_v5"})   # batch 2 = the base
SWAP = {"swinv2_base_in22k": "swinv2_base_in22k_v5_warm"}
STEP = {"convnext_base_25d_poolattn_v5": 3700, "swinv2_base_in22k_v5_warm": 2000}   # as submitted / best clean-121 step
for a in sys.argv[1:]:
    r, st = a.split("=")
    STEP[r] = int(st)

aud = pd.read_csv(HERE.parent.parent / "data" / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = (aud.loc[u, ["official_" + t for t in T]].values >= 0.5).astype(int)


def P(run):
    f = (next((HERE.parent / run / "results" / "predictions").glob(f"*_audit_s{STEP[run]}.npz"))
         if run in STEP else sw.best_npz(run, "audit"))
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[u].values, Path(f).name


def mac(p, idx=slice(None)):
    return np.mean([roc_auc_score(Y[idx, j], p[idx, j]) for j in range(12)])


old14 = [TEN_SPLIT[n] for n in TEN_SPLIT if n != "medimageinsight_davit_huge_336_fulldata"] + \
        [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]
base = [B1.get(m, m) for m in old14]
assert len(base) == 14 and all(s in base for s in SWAP), base
pred, names = {}, {}
for r in set(base) | set(SWAP.values()):
    pred[r], names[r] = P(r)
print("files:")
for r in sorted(pred):
    print(f"  {r:46s} {names[r]}")
print("\nmembers (audit-58 alone):")
for old, new in SWAP.items():
    print(f"  {old:40s} {mac(pred[old]):.4f}  ->  {new:40s} {mac(pred[new]):.4f}  ({mac(pred[new]) - mac(pred[old]):+.4f})")
ens = lambda ms: np.mean([pred[m] for m in ms], 0)   # noqa: E731
b = ens(base)
print(f"\nbatch-2 fourteen (split analogues)  {mac(b):.4f}")
for old, new in SWAP.items():
    s = ens([new if m == old else m for m in base])
    print(f"  swap only {new:40s} {mac(s):.4f} ({mac(s) - mac(b):+.4f})")
allswap = ens([SWAP.get(m, m) for m in base])
rng = np.random.default_rng(0)
d = []
for _ in range(2000):
    i = rng.integers(0, 58, 58)
    try:
        d.append(mac(allswap, i) - mac(b, i))
    except ValueError:
        pass
d = np.array(d)
print(f"\nSWIN SWAPPED                        {mac(allswap):.4f} ({mac(allswap) - mac(b):+.4f})   "
      f"90% [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}], better in {np.mean(d > 0):.0%}")
print("\nper finding (batch-2 fourteen -> SwinV2 swapped):")
for j, t in enumerate(T):
    x, y = roc_auc_score(Y[:, j], b[:, j]), roc_auc_score(Y[:, j], allswap[:, j])
    print(f"  {t:17s} {x:.4f} -> {y:.4f} ({y - x:+.4f})")
