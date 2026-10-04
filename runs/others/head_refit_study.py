"""head_refit_study.py - head-only ideas on frozen features, measured without selection bias.

Features: runs/<run>/results/features_s<step>.npz (head_refit_features.py; default convnext_base_25d_18slices_v5_aug_base
step 5900, 1,024 features per study). Training targets v4 for every study; scoring on the v5 val targets (v3 for the
346, panel for the 58 audit). fp32, full-batch AdamW, the backbone never touched.

0. IS THE +0.0025 AUDIT GAIN REAL? Paired bootstrap of the 58 (and the 346) for the best settings of the first test,
   which were picked as the best of 9 (max of noisy numbers = optimistic).
1. LINEAR HEAD, ONE SETTING FOR ALL LABELS vs EACH LABEL ITS OWN SETTING ("12 heads"). A linear head with 12 outputs
   already IS 12 independent one-label heads (separate weights, separate loss terms); what a per-label head adds is
   per-label settings (steps, pull to the original head). Settings grid: steps {50, 200, 800} x pull {0, 1e-3, 1e-2}.
2. MLP HEAD 1024 -> hidden -> 12:  "fresh" (trained from scratch)  and  "residual" (original head + a small MLP
   correction whose last layer starts at zero, so it begins exactly at the trained head).

PROTOCOL (no number is picked after seeing it): nested cross-validation on the 346 val. Outer 5 folds: fit on train +
4/5 of val, score the held-out fifth. The setting is chosen INSIDE each outer fold by an inner 4-fold CV on that fold's
4/5 of val (per label for the per-label variant). The 58 audit: setting chosen by 5-fold CV on all 346, fit on train +
all 346, scored once. Every variant gets a paired bootstrap against the original head (val 346 and audit 58).

    python runs/others/head_refit_study.py               # everything
    python runs/others/head_refit_study.py --only mlp    # just the MLP heads
"""
import argparse
import itertools
import time
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
ap.add_argument("--only", default="all", choices=("all", "linear", "mlp"), help="which head family to study")
a = ap.parse_args()
torch.set_num_threads(8)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # the features are tiny: a fit is milliseconds on a GPU

z = np.load(RUNS / a.run / "results" / f"features_s{a.step}.npz", allow_pickle=True)
ids, split = z["study_ids"].astype(str), z["split"].astype(str)
F = torch.tensor(z["feats"], dtype=torch.float32, device=DEV)
W0, B0 = torch.tensor(z["head_w"], device=DEV), torch.tensor(z["head_b"], device=DEV)
v4 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
tgt = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
Ytr = torch.tensor(v4.loc[ids, T].values, dtype=torch.float32, device=DEV)
tr, va, au = (split == "train"), (split == "val"), (split == "audit")
Ybin = np.zeros((len(ids), 12), int)
Ybin[va | au] = (tgt.loc[ids[va | au], T].values > 0.5).astype(int)
vi, ai = np.where(va)[0], np.where(au)[0]
print(f"{F.shape[0]} studies x {F.shape[1]} features: train {tr.sum()}, val {va.sum()}, audit {au.sum()} | device {DEV}")


def auc_cols(Y, P):
    return np.array([roc_auc_score(Y[:, j], P[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan for j in range(12)])


def orig(rows):
    return (F[rows] @ W0.T + B0).cpu().numpy()


# ---------------- models ----------------
def fit_linear(rows, steps, pull, seed=0):
    torch.manual_seed(seed)
    W, B = W0.clone().requires_grad_(True), B0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([W, B], lr=1e-3, weight_decay=0.0)
    X, Y = F[rows], Ytr[rows]
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ W.T + B, Y) + pull * ((W - W0) ** 2).sum()
        loss.backward()
        opt.step()
    W, B = W.detach(), B.detach()
    return lambda r: (F[r] @ W.T + B).cpu().numpy()


class MLP(torch.nn.Module):
    def __init__(self, hidden, drop, residual):
        super().__init__()
        self.residual = residual
        self.net = torch.nn.Sequential(torch.nn.LayerNorm(F.shape[1]), torch.nn.Linear(F.shape[1], hidden),
                                       torch.nn.GELU(), torch.nn.Dropout(drop), torch.nn.Linear(hidden, 12))
        if residual:                                  # starts EXACTLY at the trained head
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x)
        return out + (x @ W0.T + B0) if self.residual else out


def fit_mlp(rows, hidden, drop, wd, steps, residual, seed=0):
    torch.manual_seed(seed)
    net = MLP(hidden, drop, residual).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    X, Y = F[rows], Ytr[rows]
    net.train()
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(net(X), Y).backward()
        opt.step()
    net.eval()

    def pred(r):
        with torch.no_grad():
            return net(F[r]).cpu().numpy()
    return pred


LIN = [("linear", dict(steps=s, pull=p)) for s, p in itertools.product((50, 200, 800), (0.0, 1e-3, 1e-2))]
MLPS = ([("mlp_residual", dict(hidden=h, drop=0.3, wd=w, steps=s, residual=True))
         for h, w, s in itertools.product((256, 1024), (1e-3, 1e-2), (300, 1000))]
        + [("mlp_fresh", dict(hidden=512, drop=0.3, wd=w, steps=s, residual=False))
           for w, s in itertools.product((1e-3, 1e-2), (300, 1000))])


_CACHE = {}


def make(kind, kw, rows):
    """each (setting, training rows) is fitted ONCE: the shared-setting and per-label variants reuse every fit"""
    key = (kind, tuple(sorted(kw.items())), np.packbits(rows).tobytes())
    if key not in _CACHE:
        _CACHE[key] = fit_linear(rows, **kw) if kind == "linear" else fit_mlp(rows, **kw)
    return _CACHE[key]


# ---------------- nested CV ----------------
rng = np.random.default_rng(0)
outer = np.array_split(rng.permutation(vi), 5)


def cv_preds(configs, pool_val, folds_of):
    """{config index: predictions for every row of pool_val}, each fit on train + pool_val minus that fold"""
    out = {}
    for ci, (kind, kw) in enumerate(configs):
        P = np.zeros((len(ids), 12))
        for f in folds_of:
            rows = tr.copy()
            rows[np.setdiff1d(pool_val, f)] = True
            P[f] = make(kind, kw, rows)(f)
        out[ci] = P
    return out


def nested(configs, per_label):
    """nested-CV predictions for the 346 val + audit predictions; per_label picks each label's own config"""
    Pval = np.zeros((len(ids), 12))
    picks = []
    for k, f in enumerate(outer):
        pool = np.setdiff1d(vi, f)
        inner = np.array_split(np.random.default_rng(k + 1).permutation(pool), 4)
        cvp = cv_preds(configs, pool, inner)
        sc = np.array([auc_cols(Ybin[pool], cvp[c][pool]) for c in range(len(configs))])   # [config, label]
        choice = sc.argmax(0) if per_label else np.full(12, np.nanmean(sc, 1).argmax())
        picks.append(choice)
        rows = tr.copy(); rows[pool] = True
        fitted = {c: make(*configs[c], rows) for c in set(choice.tolist())}
        for j in range(12):
            Pval[f, j] = fitted[choice[j]](f)[:, j]
    # audit: choose on CV over all 346, fit on train + all 346
    cvp = cv_preds(configs, vi, outer)
    sc = np.array([auc_cols(Ybin[vi], cvp[c][vi]) for c in range(len(configs))])
    choice = sc.argmax(0) if per_label else np.full(12, np.nanmean(sc, 1).argmax())
    rows = tr.copy(); rows[vi] = True
    fitted = {c: make(*configs[c], rows) for c in set(choice.tolist())}
    Pau = np.zeros((len(ids), 12))
    for j in range(12):
        Pau[ai, j] = fitted[choice[j]](ai)[:, j]
    return Pval, Pau, picks, choice


def boot(P, Q, rows, n=2000):
    """paired bootstrap of macro AUROC(P) - macro AUROC(Q) over `rows`"""
    r = np.random.default_rng(0)
    d = []
    for _ in range(n):
        s = r.choice(rows, len(rows))
        d.append(np.nanmean(auc_cols(Ybin[s], P[s])) - np.nanmean(auc_cols(Ybin[s], Q[s])))
    d = np.array(d)
    return np.percentile(d, 5), np.percentile(d, 95), np.mean(d > 0)


A = np.zeros((len(ids), 12))
A[va | au] = orig(np.where(va | au)[0])
A_val, A_au = auc_cols(Ybin[vi], A[vi]), auc_cols(Ybin[ai], A[ai])
print(f"original head: val346 {np.nanmean(A_val):.4f}  audit58 {np.nanmean(A_au):.4f}")

# ---- 0. the first test's best settings, with their uncertainty ----
if a.only != "all":
    print("(section 0 skipped: --only %s)" % a.only)
print("\n0. IS THE FIRST TEST'S AUDIT GAIN REAL? (paired bootstrap, 2,000 resamples)")
for name, kw in ((("train only, 50 steps, no pull (B)", dict(steps=50, pull=0.0)),) if a.only == "all" else ()):
    p = fit_linear(tr, **kw)
    P = np.zeros((len(ids), 12)); P[va | au] = p(np.where(va | au)[0])
    for lab, rows in (("val346", vi), ("audit58", ai)):
        dd = np.nanmean(auc_cols(Ybin[rows], P[rows])) - np.nanmean(auc_cols(Ybin[rows], A[rows]))
        lo, hi, pr = boot(P, A, rows)
        print(f"   {name:36s} {lab:8s} {dd:+.4f}  90% [{lo:+.4f}, {hi:+.4f}]  better in {pr:.0%}")

# ---- 1 + 2 ----
results = {}
for label, configs, per_label in (("linear, one setting for all labels", LIN, False),
                                  ("linear, each label its own setting (12 heads)", LIN, True),
                                  ("MLP 1024->hidden->12, residual + fresh", MLPS, False),
                                  ("MLP, each label its own setting", MLPS, True)):
    if (a.only == "linear" and configs is MLPS) or (a.only == "mlp" and configs is LIN):
        continue
    t0 = time.time()
    Pval, Pau, picks, choice = nested(configs, per_label)
    pv, pa = auc_cols(Ybin[vi], Pval[vi]), auc_cols(Ybin[ai], Pau[ai])
    lo, hi, pr = boot(Pval, A, vi)
    alo, ahi, apr = boot(Pau, A, ai)
    results[label] = (pv, pa)
    print(f"\n{label}   ({time.time() - t0:.0f} s)")
    print(f"   val346 (nested CV) {np.nanmean(pv):.4f} ({np.nanmean(pv) - np.nanmean(A_val):+.4f})  90% [{lo:+.4f}, {hi:+.4f}]  better in {pr:.0%}")
    print(f"   audit58            {np.nanmean(pa):.4f} ({np.nanmean(pa) - np.nanmean(A_au):+.4f})  90% [{alo:+.4f}, {ahi:+.4f}]  better in {apr:.0%}")
    used = [f"{configs[c][0]} {configs[c][1]}" for c in sorted(set(choice.tolist()))]
    print(f"   settings chosen for the audit fit: {used[:4]}{' ...' if len(used) > 4 else ''}")

print("\nPER LABEL, val346 nested CV (original | linear one setting | linear per label | MLP | MLP per label):")
names = list(results)
print("   columns:", " | ".join(names))
for j, t in enumerate(T):
    row = " | ".join(f"{results[n][0][j]:.4f} ({results[n][0][j] - A_val[j]:+.4f})" for n in names)
    print(f"  {t:17s} {A_val[j]:.4f} | {row}")
print("\nPER LABEL, audit58 (same order):")
for j, t in enumerate(T):
    row = " | ".join(f"{results[n][1][j]:.4f} ({results[n][1][j] - A_au[j]:+.4f})" for n in names)
    print(f"  {t:17s} {A_au[j]:.4f} | {row}")
