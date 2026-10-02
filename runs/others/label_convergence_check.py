"""label_convergence_check.py - do the w50 labels peak at a different step than the old labels?

Full-data retrains cannot early-stop (nothing is held out), so they train a FIXED number of epochs
taken from a split-data run's peak. If the new labels peak later or earlier, that number must move.
Same recipe (convnext_base_25d_18slices), only the label file differs. Per run:
  - the step of the best val (each on its OWN labels, so the levels are not comparable, only the timing)
  - the step of the best audit-58 (panel labels, the one shared target) and a smoothed audit peak
  - audit-58 at the end of each epoch, to see the curve's shape

    python runs/others/label_convergence_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
TWINS = [("convnext_base_25d_18slices", "v2 (Qwen)"),
         ("convnext_base_25d_18slices_mglabels_raw", "v3"),
         ("convnext_base_25d_18slices_oof_seedfix", "v3 + 30% ConvNeXt OOF"),
         ("convnext_base_25d_18slices_ens_seedfix", "v3 + 30% ensemble"),
         ("convnext_base_25d_18slices_oof_convnext_mii_labels_w30", "w30"),
         ("convnext_base_25d_18slices_oof_convnext_mii_labels_w50", "w50 (chosen)"),
         ("convnext_base_25d_18slices_oof_convnext_mii_labels_w60", "w60")]

print(f"{'run':58s} {'labels':22s} {'steps/ep':>8s} {'best val':>16s} {'best audit':>16s} {'audit, 3-pt smooth':>20s} {'last':>6s}")
curves = {}
for run, lab in TWINS:
    v = pd.read_csv(RUNS / run / "results" / "val_log.csv").drop_duplicates("step", keep="last").sort_values("step")
    spe = (v.step / v.epoch).median() if "epoch" in v else float("nan")
    bv = v.loc[v.val_macro_auroc.idxmax()]
    a = v.dropna(subset=["audit_macro_auroc"])
    ba = a.loc[a.audit_macro_auroc.idxmax()]
    sm = a.audit_macro_auroc.rolling(3, center=True, min_periods=1).mean()
    bs = a.loc[sm.idxmax()]
    print(f"{run:58s} {lab:22s} {spe:8.0f} {int(bv.step):6d} (ep {int(bv.epoch):2d})  {int(ba.step):6d} ({ba.audit_macro_auroc:.4f})"
          f"  {int(bs.step):6d} ({sm.max():.4f})  {int(v.step.max()):6d}")
    curves[lab] = a.groupby("epoch").audit_macro_auroc.last()

print("\naudit-58 at the last validation of each epoch:")
c = pd.DataFrame(curves)
print(c.to_string(float_format=lambda x: f"{x:.4f}"))
