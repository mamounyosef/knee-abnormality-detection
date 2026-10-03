"""audit_noise_check.py - when audit-58 moves during training, is it real or noise? (no "truth" needed)

User's point (2 Oct): val (704, LLM labels) may itself be wrong, so judging audit-58 against val is not
fair. This asks the question that needs no reference: does an audit move REPLICATE?

  1. SPLIT-HALF: the 58 split into two random halves of 29. If a check-to-check change of the model is
     real, both halves see it; if it is noise, they disagree. corr(delta half A, delta half B) per run,
     averaged over 30 random splits, then Spearman-Brown to the full 58: rel = 2r / (1 + r).
     The same for val's halves (352 + 352) as a yardstick.
  2. AUDIT vs VAL: corr(delta audit, delta val), check to check. Two different study sets and label
     sources: what they share is the model actually changing.
  3. NOISE SIZE: the typical check-to-check move of audit vs of val, and how much of audit's move is
     shared with its other half.
Everything twice: all checks, and only checks from epoch 6 on (where stopping is decided, after the
early climb that makes every curve rise together).

    python runs/others/audit_noise_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
D = RUNS.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
NSPLIT = 30
rng = np.random.default_rng(0)


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    d = p[:, None] - n[None, :]
    return (d > 0).mean() + 0.5 * (d == 0).mean()


def macro(Y, P, idx):
    return np.nanmean([auc(Y[idx, j], P[idx, j]) for j in range(12)])


def load(pdir, kind, st, ids=None):
    fs = sorted(pdir.glob(f"*_{kind}_s{st}.npz"), key=lambda f: f.stat().st_mtime)
    if not fs:
        return None
    try:
        z = np.load(fs[-1], allow_pickle=True)
        df = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]])
        tg = pd.DataFrame(z["targets"], index=df.index) if "targets" in z.files else None
    except Exception:
        return None
    if df.shape[1] != 12:
        return None
    if ids is not None:
        if not set(ids) <= set(df.index):
            return None
        return df.loc[ids].values, (tg.loc[ids].values if tg is not None else None)
    return df, tg


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return np.corrcoef(a[m], b[m])[0, 1]


aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
U = [str(s) for s in aud.index]
YA = (aud[["official_" + t for t in T]].values >= 0.5).astype(int)

rows = []
for vl in sorted(RUNS.glob("*/results/val_log.csv")):
    run = vl.parent.parent.name
    cfg = (vl.parent.parent / "config.yaml").read_text(encoding="utf-8", errors="replace")
    if "train_full" in cfg:
        continue
    v = pd.read_csv(vl)
    if not {"val_macro_auroc", "step", "epoch"} <= set(v.columns):
        continue
    v = v.dropna(subset=["val_macro_auroc"]).drop_duplicates("step", keep="last").sort_values("step")
    pdir = vl.parent / "predictions"
    steps, A, V, ep = [], [], [], []
    vids, YV = None, None
    for st, e in zip(v.step.astype(int), v.epoch):
        a = load(pdir, "audit", st, U)
        if a is None:
            continue
        if vids is None:
            first = load(pdir, "val", st)
            if first is None or first[1] is None:
                continue
            vids = list(first[0].index)
            YV = (first[1].values >= 0.5).astype(int)
        vv = load(pdir, "val", st, vids)
        if vv is None:
            continue
        steps.append(st); ep.append(float(e)); A.append(a[0]); V.append(vv[0])
    if len(steps) < 10:
        continue
    ep = np.array(ep)
    nA, nV = len(U), len(vids)
    full_a = np.array([macro(YA, P, np.arange(nA)) for P in A])
    full_v = np.array([macro(YV, P, np.arange(nV)) for P in V])
    res = {"run": run, "checks": len(steps)}
    for tag, sel in (("all", np.ones(len(steps), bool)), ("ep6+", ep >= 6)):
        idx = np.where(sel)[0]
        if len(idx) < 6:
            continue
        da, dv = np.diff(full_a[idx]), np.diff(full_v[idx])
        res[f"{tag}_n"] = len(idx)
        res[f"{tag}_a_v"] = corr(da, dv)
        res[f"{tag}_move_a"] = np.median(np.abs(da))
        res[f"{tag}_move_v"] = np.median(np.abs(dv))
        ra, rv = [], []
        for _ in range(NSPLIT):
            pa = rng.permutation(nA); h1, h2 = pa[: nA // 2], pa[nA // 2:]
            a1 = np.array([macro(YA, A[i], h1) for i in idx]); a2 = np.array([macro(YA, A[i], h2) for i in idx])
            ra.append(corr(np.diff(a1), np.diff(a2)))
            pv = rng.permutation(nV); g1, g2 = pv[: nV // 2], pv[nV // 2:]
            v1 = np.array([macro(YV, V[i], g1) for i in idx]); v2 = np.array([macro(YV, V[i], g2) for i in idx])
            rv.append(corr(np.diff(v1), np.diff(v2)))
        res[f"{tag}_half_a"] = np.nanmean(ra)
        res[f"{tag}_half_v"] = np.nanmean(rv)
    rows.append(res)
    print(f"  {run:60s} checks {len(steps):3d}  ep6+: audit halves agree {res.get('ep6+_half_a', np.nan):+.2f}, "
          f"val halves {res.get('ep6+_half_v', np.nan):+.2f}, audit~val {res.get('ep6+_a_v', np.nan):+.2f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_suffix(".csv"), index=False)
sb = lambda r: 2 * r / (1 + r) if r > -1 else np.nan   # noqa: E731
print(f"\n{len(df)} runs. Check-to-check changes; median over runs (and the share of runs where the value is > 0).")
for tag, lab in (("all", "ALL CHECKS (includes the early climb)"), ("ep6+", "FROM EPOCH 6 ON (where stopping is decided)")):
    d = df.dropna(subset=[f"{tag}_half_a"])
    ha, hv, av = d[f"{tag}_half_a"], d[f"{tag}_half_v"], d[f"{tag}_a_v"]
    print(f"\n{lab}: {len(d)} runs")
    print(f"  audit half vs audit half   r = {ha.median():+.2f}  (> 0 in {(ha > 0).mean():.0%})  -> full-58 reliability {sb(ha.median()):+.2f}")
    print(f"  val half vs val half       r = {hv.median():+.2f}  (> 0 in {(hv > 0).mean():.0%})  -> full-704 reliability {sb(hv.median()):+.2f}")
    print(f"  audit vs val               r = {av.median():+.2f}  (> 0 in {(av > 0).mean():.0%})")
    print(f"  typical move per check     audit {d[f'{tag}_move_a'].median():.4f}   val {d[f'{tag}_move_v'].median():.4f}")
