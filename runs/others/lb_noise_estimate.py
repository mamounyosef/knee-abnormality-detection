"""lb_noise_estimate.py - how much can a public-leaderboard score move by chance alone?

The public board scores ~a fraction of ~1,300 test studies. Resampling subsets of that size from
our own 704 validation studies (their predictions at each run's shipped checkpoint, v2 labels)
gives the spread of ONE model's macro AUROC and, more to the point, of the DIFFERENCE between
two models scored on the same studies (paired, so their shared difficulty cancels).

    python runs/others/lb_noise_estimate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNS / "others"))
import rank_single_models as rk  # noqa: E402
import ensembling.sweep_new_runs as sw  # noqa: E402

D = RUNS.parent / "data"
val = pd.read_csv(D / "05_val_qwen24t-med.csv").set_index("StudyInstanceUID")
Y = (val[rk.T].values >= 0.5).astype(int)
MODELS = {"MII 336 (0.929)": "medimageinsight_davit_huge_336",
          "oof_seedfix (0.922)": "convnext_base_25d_18slices_oof_seedfix",
          "relabel (0.918)": "convnext_base_relabel",
          "25d all_knees (0.919)": "convnext_base_25d_all_knees_same_orientation"}
L = {}
for k, r in MODELS.items():
    z = np.load(sw.best_npz(r, "val"), allow_pickle=True)
    L[k] = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]]).loc[val.index].values
rng = np.random.default_rng(0)
for n in (300, 400, 500):
    single, pair = [], {}
    for _ in range(1000):
        i = rng.choice(len(Y), n, replace=True)
        if (Y[i].sum(0) == 0).any():
            continue
        s = {k: rk.macro(v[i], Y[i]) for k, v in L.items()}
        single.append(s["MII 336 (0.929)"])
        for k in list(MODELS)[1:]:
            pair.setdefault(k, []).append(s["MII 336 (0.929)"] - s[k])
    print(f"\npublic set of {n} studies:")
    print(f"  one model's score: +/- {np.std(single):.4f} (1 SD); 95% of draws within +/- {1.96 * np.std(single):.4f}")
    for k, g in pair.items():
        g = np.array(g)
        print(f"  MII minus {k:22s} SD {g.std():.4f}  -> a {0.929 - float(k.split('(')[1][:5]):+.3f} public gap is "
              f"{abs(0.929 - float(k.split('(')[1][:5])) / g.std():.1f} SD")
