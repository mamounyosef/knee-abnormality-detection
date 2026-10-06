"""localizer_robustness.py - how often does the joint-zoom localizer fail? Over many val/audit studies (the 0.30 run's
own config, the lightest to load): crops pushed against the frame edge (clamped), slots with no CAM signal (frame
centre fallback), and how far each view's centre wanders between studies. Writes the worst cases as PNGs.

    python runs/others/crop_rois/localizer_robustness.py [n_studies]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 160
RUN = "convnext_base_25d_v5_joint_zoom_030"
spec = importlib.util.spec_from_file_location("jz", ROOT / "runs" / RUN / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.cfg
dev = torch.device("cuda")
model = mod.build_model(cfg).to(dev).eval()
plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
df = sc._eval_df(cfg, cfg["paths"]["val_csv"]).sample(N, random_state=1).reset_index(drop=True)
ds = sc.KneeShardDataset(df, cfg, plan, train=False)
loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=4)
H, s, slots = cfg["input"]["size"], model.crop, cfg["input"]["slots"]
C, missing = [], []
with torch.no_grad():
    for x, _ in loader:
        C.append(model.centres(sc.prepare_batch(x, cfg, dev)).cpu().view(x.shape[0], len(slots), 2))
        missing.append((x.flatten(2).amax(2) == 0).view(x.shape[0], len(slots)))   # blank slot (series absent)
C = torch.cat(C).numpy()
M = torch.cat(missing).numpy()
print(f"{N} studies, frame {H} px, crop {s} px")
for k, sl in enumerate(slots):
    c = C[~M[:, k], k]
    clamp = ((c - s / 2 < 0) | (c + s / 2 > H)).any(1)
    centre = (np.abs(c - H / 2) < 1e-3).all(1)
    print(f"  {sl:10s} present {len(c):3d} | clamped at edge {clamp.mean():5.1%} | frame-centre fallback {centre.mean():5.1%} "
          f"| centre y {c[:, 0].mean() / H:.2f}+-{c[:, 0].std() / H:.2f}, x {c[:, 1].mean() / H:.2f}+-{c[:, 1].std() / H:.2f} (frame fractions)")
# worst: the centre furthest from that view's median, per view
med = np.nanmedian(np.where(M[..., None], np.nan, C), 0)
dist = np.linalg.norm(C - med[None], axis=2)
dist[M] = -1
out = Path(__file__).resolve().parent / "robustness_worst"
out.mkdir(exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for k, sl in enumerate(slots):
    for r, i in enumerate(np.argsort(-dist[:, k])[:2]):
        x, _ = ds[int(i)]
        img = x[k, 9].numpy()
        cy, cx = C[i, k]
        y0, x0 = int(np.clip(round(cy - s / 2), 0, H - s)), int(np.clip(round(cx - s / 2), 0, H - s))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img, cmap="gray"); ax[0].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="lime", lw=2))
        ax[1].imshow(img[y0:y0 + s, x0:x0 + s], cmap="gray")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"{sl} worst #{r + 1}: {dist[i, k]:.0f} px from the median centre")
        fig.savefig(out / f"{sl}_worst{r + 1}.png", dpi=60)
        plt.close(fig)
print("worst cases ->", out)
