"""head_refit_fit.py - step 2 of the head-refit test: retrain ONLY the final linear head, fp32, on saved features.

Reads runs/<run>/results/features_s<step>.npz (from head_refit_features.py). Normal gradient descent (AdamW,
full batch, binary cross-entropy on soft labels), starting from the trained head, with an L2 pull back toward the
original head weights. The backbone is never touched. Training labels: v4 (w50) for every study; scoring: the v5
val's own targets (v3 for the 346, panel for the 58 audit).

  A  original head                                   (must reproduce the run's logged val, 0.9174 on the 404)
  B  refit on TRAIN only (3,996, features the backbone has seen)
  C  refit on TRAIN + VAL: 5-fold cross-validation on the 346 val (fit on train + 4/5 of val, score the held-out
     fifth), audit-58 scored by a fit on train + all 346
  D  refit on VAL only (held-out features, as in "deep feature reweighting"), same 5 folds
  each for a small grid of (steps, pull strength); val numbers are over the 346, audit over the 58.

    python runs/others/head_refit_fit.py
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

RUNS = Path(__file__).resolve().parent.parent
D = RUNS.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="convnext_base_25d_18slices_v5_aug_base")
ap.add_argument("--step", default="5900")
a = ap.parse_args()
torch.manual_seed(0)

z = np.load(RUNS / a.run / "results" / f"features_s{a.step}.npz", allow_pickle=True)
ids, split = z["study_ids"].astype(str), z["split"].astype(str)
F = torch.tensor(z["feats"], dtype=torch.float32)
W0, B0 = torch.tensor(z["head_w"]), torch.tensor(z["head_b"])
v4 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
tgt = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
Ytr = torch.tensor(v4.loc[ids, T].values, dtype=torch.float32)             # training targets, v4 everywhere
tr, va, au = (split == "train"), (split == "val"), (split == "audit")
Yva = (tgt.loc[ids[va], T].values > 0.5).astype(int)
Yau = (tgt.loc[ids[au], T].values > 0.5).astype(int)
print(f"{F.shape[0]} studies x {F.shape[1]} features: train {tr.sum()}, val {va.sum()}, audit {au.sum()}")


def mac(Y, P):
    return np.nanmean([roc_auc_score(Y[:, j], P[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan for j in range(12)])


def head(W, B, m):
    return (F[m] @ W.T + B).numpy()


def fit(mask, steps, pull, lr=1e-3):
    """AdamW on the head only, fp32, full batch; L2 pull toward the trained head keeps it from drifting."""
    W, B = W0.clone().requires_grad_(True), B0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W, B], lr=lr, weight_decay=0.0)
    X, Y = F[mask], Ytr[mask]
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ W.T + B, Y)
        loss = loss + pull * ((W - W0) ** 2).sum()
        loss.backward()
        opt.step()
    return W.detach(), B.detach()


A_va, A_au = mac(Yva, head(W0, B0, va)), mac(Yau, head(W0, B0, au))
both = va | au
Yboth = (tgt.loc[ids[both], T].values > 0.5).astype(int)
print(f"A original head: val346 {A_va:.4f}  audit58 {A_au:.4f}  | the 404 together {mac(Yboth, head(W0, B0, both)):.4f} (logged 0.9174)\n")

rng = np.random.default_rng(0)
vi = np.where(va)[0]
folds = np.array_split(rng.permutation(vi), 5)
grid = list(itertools.product((50, 200, 800), (0.0, 1e-3, 1e-2)))
print(f"{'variant':6s} {'steps':>5s} {'pull':>6s} | {'val346':>7s} {'vs A':>8s} | {'audit58':>7s} {'vs A':>8s}")
for steps, pull in grid:
    W, B = fit(tr, steps, pull)                                   # B: train only
    bv, ba = mac(Yva, head(W, B, va)), mac(Yau, head(W, B, au))
    print(f"{'B':6s} {steps:5d} {pull:6.0e} | {bv:.4f} {bv - A_va:+.4f} | {ba:.4f} {ba - A_au:+.4f}")
    for name, base in (("C", tr), ("D", np.zeros_like(tr))):      # C: train + 4/5 val; D: 4/5 val only
        P = np.zeros((len(ids), 12))
        for f in folds:
            m = base.copy()
            m[np.setdiff1d(vi, f)] = True
            Wf, Bf = fit(m, steps, pull)
            P[f] = (F[f] @ Wf.T + Bf).numpy()
        cv = mac(Yva, P[va])
        m = base.copy(); m[vi] = True
        Wa, Ba = fit(m, steps, pull)
        ca = mac(Yau, head(Wa, Ba, au))
        print(f"{name:6s} {steps:5d} {pull:6.0e} | {cv:.4f} {cv - A_va:+.4f} | {ca:.4f} {ca - A_au:+.4f}")
