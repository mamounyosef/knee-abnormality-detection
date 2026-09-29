"""compare_step_by_step.py - two runs of the same recipe, compared at every shared step.

Validation (val_log.csv): loss and macro AUROC at each shared validation step.
Training (train_log.csv): lr, loss and gradient norm at each shared optimizer step, plus
100-step window means (single steps are one batch each, so they are noisy by nature).

    python runs/others/compare_step_by_step.py <run_a> <run_b>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
a, b = sys.argv[1], sys.argv[2]


def load(run, name, key):
    t = pd.read_csv(RUNS / run / "results" / name)
    t = t[t.run_id == t.run_id.iloc[-1]]
    if "trigger" in t:
        t = t[t.trigger.astype(str).str.startswith("every")]
    return t.drop_duplicates(key, keep="last").set_index(key)


va, vb = load(a, "val_log.csv", "step"), load(b, "val_log.csv", "step")
common = va.index.intersection(vb.index)
print(f"VALIDATION at {len(common)} shared steps ({common.min()}..{common.max()})")
print(f"{'step':>6} {'loss A':>8} {'loss B':>8} {'diff':>8} {'AUROC A':>8} {'AUROC B':>8} {'diff':>8}")
for s in common:
    la, lb = va.loc[s, "val_loss"], vb.loc[s, "val_loss"]
    ra, rb = va.loc[s, "val_macro_auroc"], vb.loc[s, "val_macro_auroc"]
    print(f"{s:>6} {la:8.4f} {lb:8.4f} {lb - la:+8.4f} {ra:8.4f} {rb:8.4f} {rb - ra:+8.4f}")
dl = (vb.loc[common, "val_loss"] - va.loc[common, "val_loss"]).abs()
dr = (vb.loc[common, "val_macro_auroc"] - va.loc[common, "val_macro_auroc"]).abs()
print(f"val loss |diff|: mean {dl.mean():.4f}, max {dl.max():.4f}; "
      f"val AUROC |diff|: mean {dr.mean():.4f}, max {dr.max():.4f}")

ta, tb = load(a, "train_log.csv", "global_step"), load(b, "train_log.csv", "global_step")
c = ta.index.intersection(tb.index)
print(f"\nTRAINING at {len(c)} shared steps ({c.min()}..{c.max()})")
for col in ("lr", "loss", "gnorm"):
    x, y = ta.loc[c, col].astype(float), tb.loc[c, col].astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = (y[ok] - x[ok]).abs()
    rel = (d / x[ok].abs().clip(lower=1e-12))
    print(f"  {col:6s} per step: |diff| mean {d.mean():.3g}, max {d.max():.3g} "
          f"(relative mean {rel.mean():.2%}); exact matches {int((d == 0).sum())}/{int(ok.sum())}")
bins = (pd.Series(c, index=c) - 1) // 100
for col in ("loss", "gnorm"):
    x = ta.loc[c, col].astype(float).groupby(bins.values).mean()
    y = tb.loc[c, col].astype(float).groupby(bins.values).mean()
    d = (y - x)
    print(f"  {col:6s} 100-step window means: |diff| mean {d.abs().mean():.4f}, max {d.abs().max():.4f}, "
          f"mean signed {d.mean():+.4f}")
