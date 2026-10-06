"""zoom_tta_test.py - does averaging the zoom model over SHIFTED crops help? (test-time augmentation, free at train time)

Loads each study ONCE (the 0.30 run's own loader), finds the joint once, then scores the trunk on the crop at the found
centre plus shifted copies (+-off px in y and x), and compares the joint-finding AUROC of: centre only vs the mean of
5 crops vs the mean of 9 crops. Surfaces: audit-58 (panel labels) and the clean v5 val (v3 labels).

    python runs/others/crop_rois/zoom_tta_test.py [checkpoint] [off_px]
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUN = "convnext_base_25d_v5_joint_zoom_030"
CK = sys.argv[1] if len(sys.argv) > 1 else "top_step1500.pt"
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 16                   # 16 px = 4.8 mm at 0.30 mm/px
JOINT = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
spec = importlib.util.spec_from_file_location("jz", ROOT / "runs" / RUN / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.cfg
dev = torch.device("cuda")
model = mod.JointZoomModel(cfg)
sd = torch.load(ROOT / "runs" / RUN / "results" / "checkpoints" / CK, map_location="cpu", weights_only=False)["model"]
sc._load_model_state(model, sd)
model.to(dev).eval()
plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
df = sc._eval_df(cfg, cfg["paths"]["val_csv"])
au = sc._eval_df(cfg, cfg["evaluation"]["audit"]["csv"])
dl = torch.utils.data.DataLoader(sc.KneeShardDataset(df, cfg, plan, train=False), batch_size=4, num_workers=0)   # Windows: workers need a main guard
shifts5 = [(0, 0), (OFF, 0), (-OFF, 0), (0, OFF), (0, -OFF)]
shifts9 = shifts5 + [(OFF, OFF), (OFF, -OFF), (-OFF, OFF), (-OFF, -OFF)]
H, s, g = cfg["input"]["size"], model.crop, model.g
out = {k: [] for k in range(9)}
with torch.no_grad(), torch.autocast("cuda"):
    for bi, (x, _) in enumerate(dl):
        xb = sc.prepare_batch(x, cfg, dev)
        c = model.centres(xb)
        xv = xb.view(-1, g, *xb.shape[1:])
        for k, (dy, dx) in enumerate(shifts9):
            o = torch.clamp(torch.round(c + torch.tensor([dy, dx], device=dev) - s / 2), 0, H - s).long().cpu().tolist()
            crops = torch.stack([xv[i, :, :, y0:y0 + s, x0:x0 + s] for i, (y0, x0) in enumerate(o)])
            out[k].append(mod.SliceModel.forward(model, crops.view(-1, *crops.shape[2:])).float().cpu().numpy())
        if bi % 20 == 0:
            print(f"  batch {bi}/{len(dl)}", flush=True)
L = {k: np.concatenate(v) for k, v in out.items()}
ids = list(df.StudyInstanceUID)
T = cfg["tasks"]
J = [T.index(t) for t in JOINT]
preds = {"centre": L[0], "mean of 5": np.mean([L[k] for k in range(5)], 0), "mean of 9": np.mean([L[k] for k in range(9)], 0)}
is_au = np.array([u in set(au.StudyInstanceUID) for u in ids])
for surf, m in (("audit-58", is_au), ("clean v5 val", ~is_au)):
    y = df.loc[m, T].to_numpy() > 0.5
    print(f"\n{surf} ({m.sum()} studies), joint-finding macro AUROC (per finding):")
    for name, p in preds.items():
        a = [roc_auc_score(y[:, j], p[m, j]) for j in J]
        print(f"  {name:10s} {np.mean(a):.4f}  " + " ".join(f"{t[:8]} {v:.3f}" for t, v in zip(JOINT, a)))
np.savez(ROOT / "runs" / RUN / "results" / f"tta_{CK[:-3]}_off{OFF}.npz", ids=np.array(ids), **{f"s{k}": L[k] for k in L})
