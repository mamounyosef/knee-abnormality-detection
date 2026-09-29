"""compare_convergence.py - val macro AUROC over training for several runs, side by side.

Answers "did a label change move the best step?": per epoch, the best validation inside it,
then each run's overall peak and how close to the peak the curve already is (within 0.002).

    python runs/others/compare_convergence.py run_a run_b [...]
"""
import sys
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
runs = sys.argv[1:]
logs = {}
for r in runs:
    v = pd.read_csv(RUNS / r / "results/val_log.csv")
    v = v[v.run_id == v.run_id.iloc[-1]]                      # latest run_id only
    logs[r] = v

print("best val macro AUROC within each epoch (step of that best)")
eps = sorted(set().union(*[set(v.epoch) for v in logs.values()]))
print("epoch  " + "  ".join(f"{r[-28:]:>30}" for r in runs))
for e in eps:
    cells = []
    for r in runs:
        v = logs[r][logs[r].epoch == e]
        cells.append(f"{v.val_macro_auroc.max():.4f} (s{int(v.loc[v.val_macro_auroc.idxmax(), 'step'])})"
                     if len(v) else "-")
    print(f"{e:>5}  " + "  ".join(f"{c:>30}" for c in cells))
print()
for r, v in logs.items():
    b = v.loc[v.val_macro_auroc.idxmax()]
    steps_per_epoch = v.step.max() / v.epoch.max()
    near = v[v.val_macro_auroc >= b.val_macro_auroc - 0.002].step.min()
    print(f"{r}: peak {b.val_macro_auroc:.4f} at step {int(b.step)} (epoch {int(b.epoch)}, "
          f"{b.step / steps_per_epoch:.1f} epochs); within 0.002 of peak from step {int(near)}; "
          f"last step {int(v.step.max())}")
