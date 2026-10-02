"""audit_stop_reliability.py - can the 58 audit studies alone choose WHEN to stop a full-data run?

Idea under test (user, 2 Oct): train the full-data relabels on everything EXCEPT the 58 audit studies,
and stop at the audit-58 peak instead of at a fixed epoch. Checked on every finished split-data run
that has both a 704-study validation curve and per-step audit predictions:

  regret = best val AUROC of the run  -  val AUROC at the step audit-58 would have chosen
           (val is 12x larger, so it is the reference for "where the run really peaked")
  for three audit rules: raw argmax, 3-point smoothed argmax, 5-point smoothed argmax
  and, for comparison, the FIXED rule full-data runs use today: the step at the end of the val-peak
  epoch of a run on DIFFERENT labels is not available per run, so the baseline shown is
  "stop at a fixed 85% of the steps the run trained" and "stop at the last validation".
STABILITY: the 58 studies resampled 200 times (paired across steps), giving the spread of the
audit-chosen step (in epochs) and of its regret. That is the noise a real full-data run would face.

    python runs/others/audit_stop_reliability.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
D = RUNS.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B = 200
rng = np.random.default_rng(0)

aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
U = [str(s) for s in aud.index]
Y = (aud[["official_" + t for t in T]].values >= 0.5).astype(int)
BOOT = rng.integers(0, len(U), (B, len(U)))


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    d = p[:, None] - n[None, :]
    return (d > 0).mean() + 0.5 * (d == 0).mean()


def macro(P, idx=None):
    y = Y if idx is None else Y[idx]
    s = P if idx is None else P[idx]
    return np.nanmean([auc(y[:, j], s[:, j]) for j in range(12)])


def smooth(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values if w > 1 else np.asarray(x)


rows = []
for vl in sorted(RUNS.glob("*/results/val_log.csv")):
    run = vl.parent.parent.name
    cfgtxt = (vl.parent.parent / "config.yaml").read_text(encoding="utf-8", errors="replace")
    if "train_full" in cfgtxt or "_full" in run and "fulldata" in run:
        continue                                   # full-data runs: their val is in-sample
    v = pd.read_csv(vl)
    if "val_macro_auroc" not in v or "step" not in v:
        continue
    v = v.dropna(subset=["val_macro_auroc"]).drop_duplicates("step", keep="last").sort_values("step")
    pdir = vl.parent / "predictions"
    steps, A = [], []
    for st in v.step.astype(int):
        fs = sorted(pdir.glob(f"*_audit_s{st}.npz"), key=lambda f: f.stat().st_mtime)
        if not fs:
            continue
        try:
            z = np.load(fs[-1], allow_pickle=True)
            df = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]])
        except Exception:                          # an empty / half-written npz: skip that step
            continue
        if not set(U) <= set(df.index) or df.shape[1] != 12:
            continue
        steps.append(st)
        A.append(df.loc[U].values)
    if len(steps) < 8:
        continue
    vv = v.set_index("step").loc[steps]
    val = vv.val_macro_auroc.values
    spe = float((vv.index / vv.epoch).median()) if "epoch" in vv else np.nan
    best = val.max()
    a_full = np.array([macro(P) for P in A])
    r = {"run": run, "n_steps": len(steps), "spe": spe, "val_best": best,
         "val_peak_step": steps[int(val.argmax())]}
    for w in (1, 3, 5):
        i = int(np.nanargmax(smooth(a_full, w)))
        r[f"regret_a{w}"] = best - val[i]
        r[f"dstep_a{w}"] = (steps[i] - r["val_peak_step"]) / spe
    r["regret_last"] = best - val[-1]
    i85 = int(np.argmin(np.abs(np.array(steps) - 0.85 * steps[-1])))
    r["regret_85pct"] = best - val[i85]
    # bootstrap the 58: where would a noisy audit have stopped?
    regs, dst = [], []
    for b in range(B):
        ab = np.array([macro(P, BOOT[b]) for P in A])
        i = int(np.nanargmax(smooth(ab, 3)))
        regs.append(best - val[i])
        dst.append((steps[i] - r["val_peak_step"]) / spe)
    r["boot_regret_med"], r["boot_regret_p90"] = np.median(regs), np.percentile(regs, 90)
    r["boot_dstep_iqr"] = np.percentile(dst, 75) - np.percentile(dst, 25)
    rows.append(r)
    print(f"  {run:62s} steps {len(steps):3d}  regret a3 {r['regret_a3']:.4f}  boot p90 {r['boot_regret_p90']:.4f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(Path(__file__).with_suffix(".csv"), index=False)
print(f"\n{len(df)} split-data runs. Regret = val AUROC lost by stopping where the rule says, vs the run's own val peak.")
print(f"{'rule':34s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'max':>7s}  {'within 0.002':>12s}")
for c, lab in (("regret_a1", "audit-58 raw peak"), ("regret_a3", "audit-58, 3-pt smoothed peak"),
               ("regret_a5", "audit-58, 5-pt smoothed peak"), ("regret_85pct", "fixed: 85% of the run's steps"),
               ("regret_last", "fixed: last validation"),
               ("boot_regret_med", "audit-58 3-pt, resampled (median)"),
               ("boot_regret_p90", "audit-58 3-pt, resampled (p90)")):
    x = df[c]
    print(f"{lab:34s} {x.mean():7.4f} {x.median():7.4f} {x.quantile(.9):7.4f} {x.max():7.4f}  {(x <= 0.002).mean():12.0%}")
print(f"\naudit 3-pt peak vs val peak, in epochs: median {df.dstep_a3.median():+.2f}, "
      f"mean |diff| {df.dstep_a3.abs().mean():.2f}; resampling spread (IQR) median {df.boot_dstep_iqr.median():.2f} epochs")
