"""cam_finding_analysis.py - where does the evidence for each finding sit? Decides which zoom runs are worth training.

The 2.5D parent (convnext_base_25d_18slices_v5_aug_base, its own 4 views and its own 0.40 mm / 336 px view) scores the
v5 val + audit studies; for every finding, on the studies POSITIVE for it, its exact class-activation map (CAM, the
per-region contribution to that finding's logit, as cam_preview.py) is summarised three ways:
  1. VIEW share:   which of the 4 views carries the positive evidence (where a zoom should look)
  2. SLICE focus:  share of a view's positive evidence in its top 2 of 6 image groups (3 slices each);
                   0.33 = spread evenly, near 1 = a few slices only -> zooming through the slices could help
  3. REGION spread: centroid of the evidence in each view (fractions of the frame) and its spread across studies;
                   a small spread = one consistent place -> a fixed-region crop works there
Writes runs/others/crop_rois/cam_finding_analysis.csv as well.

    python runs/others/crop_rois/cam_finding_analysis.py [max_studies]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared_code as sc  # noqa: E402
from cam_preview import load_parent  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 404
model, cfg = load_parent()
dev = torch.device("cuda")
model.to(dev).eval()
T, slots, g = cfg["tasks"], cfg["input"]["slots"], int(cfg["input"]["slices_as_channels"])
df = sc._eval_df(cfg, cfg["paths"]["val_csv"]).head(N).reset_index(drop=True)
plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
dl = torch.utils.data.DataLoader(sc.KneeShardDataset(df, cfg, plan, train=False), batch_size=4, num_workers=0)
bb, ln, W = model.backbone, model.backbone.head.norm, model.head.weight.float()
CAMS = []                                                   # per study: (12, slots, g, h, w) positive evidence
with torch.no_grad():
    for bi, (x, _) in enumerate(dl):
        xin = sc.prepare_batch(x, cfg, dev)
        f = bb.forward_features(xin).float()
        pooled = f.mean((2, 3))
        mu = pooled.mean(1, keepdim=True)
        sig = torch.sqrt(pooled.var(1, unbiased=False, keepdim=True) + ln.eps)
        fn = (f - mu[:, :, None, None]) / sig[:, :, None, None] * ln.weight.float()[None, :, None, None] + ln.bias.float()[None, :, None, None]
        cam = torch.einsum("nfhw,kf->nkhw", fn, W)                    # (N, 12, h, w)
        h, w = cam.shape[-2:]
        cam = cam.view(x.shape[0], len(slots), g, 12, h, w).permute(0, 3, 1, 2, 4, 5)
        CAMS.append(torch.clamp(cam, min=0).cpu().numpy())
        if bi % 25 == 0:
            print(f"  batch {bi}/{len(dl)}", flush=True)
C = np.concatenate(CAMS)                                  # (studies, 12, slots, g, h, w)
Y = df[T].to_numpy() > 0.5
h, w = C.shape[-2:]
ys, xs = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
rows = []
print(f"\n{len(df)} studies; views {slots}")
for j, t in enumerate(T):
    pos = Y[:, j]
    if pos.sum() < 5:
        continue
    c = C[pos, j]                                         # (npos, slots, g, h, w)
    mass = c.sum((2, 3, 4))                               # (npos, slots)
    share = (mass / np.maximum(mass.sum(1, keepdims=True), 1e-9)).mean(0)
    line = f"{t:17s} n={pos.sum():3d} | view share " + " ".join(f"{s}:{v:.2f}" for s, v in zip(slots, share))
    rec = {"finding": t, "n_pos": int(pos.sum())}
    for k, s in enumerate(slots):
        gm = c[:, k].sum((2, 3))                          # (npos, g)
        top2 = np.sort(gm, 1)[:, -2:].sum(1) / np.maximum(gm.sum(1), 1e-9)
        hw = c[:, k].sum(1)                               # (npos, h, w)
        tot = np.maximum(hw.sum((1, 2)), 1e-9)
        cy, cx = (hw * ys).sum((1, 2)) / tot, (hw * xs).sum((1, 2)) / tot
        rec.update({f"{s}_share": share[k], f"{s}_top2": top2.mean(), f"{s}_cy": cy.mean(), f"{s}_cx": cx.mean(),
                    f"{s}_spread": float(np.sqrt(cy.var() + cx.var()))})
    best = max(slots, key=lambda s: rec[f"{s}_share"])
    line += (f" | top view {best}: slice focus {rec[f'{best}_top2']:.2f}, centre y {rec[f'{best}_cy']:.2f} "
             f"x {rec[f'{best}_cx']:.2f}, spread {rec[f'{best}_spread']:.3f}")
    print(line)
    rows.append(rec)
pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent / "cam_finding_analysis.csv", index=False)
print("\nslice focus: 0.33 = even over the 6 image groups; region spread: in frame fractions (0.05 = ~7 mm)")
