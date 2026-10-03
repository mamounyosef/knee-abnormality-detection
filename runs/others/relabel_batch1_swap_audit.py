"""relabel_batch1_swap_audit.py - does swapping the relabelled (v4 labels, v5 split, flipped) members into the 0.942 ensemble
help? Scored on audit-58, the only fair surface:

  the 0.942 fourteen, each full-data member replaced by its split-data analogue (ten_mii_folds_audit.TEN_SPLIT; full-data
  runs trained on audit), vs the same fourteen with these four swapped for their relabelled versions:
      dinov3_convnext_base                         -> dinov3_convnext_base_v5
      convnext_base_relabel (analogue of convnext_base_full_all_knees_same_orientation) -> convnext_base_v5
      convnext_base_25d_all_knees_same_orientation (analogue of ..._fulldata) -> convnext_base_25d_18slices_v5_aug_base
      convnext_base_25d_poolattn (analogue of ..._fulldata_flip)            -> convnext_base_25d_poolattn_v5
  Each relabelled run at its best-val step. CAVEAT: the 58 are inside the v5 val those runs early-stopped on, so their
  audit numbers are slightly optimistic. The v5 val cannot be used: the old analogues trained on 283 of its 346 studies.
  Flat logit mean; paired bootstrap over the 58; each swap alone too.

    python runs/others/relabel_batch1_swap_audit.py
    python runs/others/relabel_batch1_swap_audit.py convnext_base_25d_poolattn_v5=3700   # a member at another step
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
SWAP = {"dinov3_convnext_base": "dinov3_convnext_base_v5",
        "convnext_base_relabel": "convnext_base_v5",
        "convnext_base_25d_all_knees_same_orientation": "convnext_base_25d_18slices_v5_aug_base",
        "convnext_base_25d_poolattn": "convnext_base_25d_poolattn_v5"}
aud = pd.read_csv(HERE.parent.parent / "data" / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
Y = (aud.loc[u, ["official_" + t for t in T]].values >= 0.5).astype(int)


STEP = {}                                   # run -> a step to use instead of its best-val step
for a in sys.argv[1:]:                      # e.g. convnext_base_25d_poolattn_v5=3700
    r, st = a.split("=")
    STEP[r] = int(st)


def P(run):
    f = (next((HERE.parent / run / "results" / "predictions").glob(f"*_audit_s{STEP[run]}.npz"))
         if run in STEP else sw.best_npz(run, "audit"))
    z = np.load(f, allow_pickle=True)
    return pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]], columns=T).loc[u].values, Path(f).name


def mac(p, idx=slice(None)):
    return np.mean([roc_auc_score(Y[idx, j], p[idx, j]) for j in range(12)])


base = [TEN_SPLIT[n] for n in TEN_SPLIT if n != "medimageinsight_davit_huge_336_fulldata"] + \
       [f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5)]
assert len(base) == 14 and all(s in base for s in SWAP)
pred = {}
for r in set(base) | set(SWAP.values()):
    pred[r], name = P(r)
print("members (audit-58 alone):")
for old, new in SWAP.items():
    print(f"  {old:46s} {mac(pred[old]):.4f}  ->  {new:40s} {mac(pred[new]):.4f}  ({mac(pred[new]) - mac(pred[old]):+.4f})")
ens = lambda ms: np.mean([pred[m] for m in ms], 0)   # noqa: E731
b = ens(base)
print(f"\nthe fourteen (split analogues)  {mac(b):.4f}")
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
print(f"\nALL FOUR SWAPPED                {mac(allswap):.4f} ({mac(allswap) - mac(b):+.4f})   "
      f"90% [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}], better in {np.mean(d > 0):.0%}")
print("\nper finding (fourteen -> all four swapped):")
for j, t in enumerate(T):
    x, y = roc_auc_score(Y[:, j], b[:, j]), roc_auc_score(Y[:, j], allswap[:, j])
    print(f"  {t:17s} {x:.4f} -> {y:.4f} ({y - x:+.4f})")
