"""save_zoom_samples.py - BEFORE / AFTER images of the joint-zoom model, one PNG per study, for an eye check.

Runs the real JointZoomModel of convnext_base_25d_v5_joint_zoom (seeded = the parent as localizer) on real studies
and records, with forward pre-hooks on the two stems, the EXACT tensors the networks receive (eval mode: no
augmentation, no crop jitter; only the ImageNet normalisation is undone for display, values 0-1 shown as-is):
  BEFORE = the localizer's input = the normal 2.5D model's view: full 134 mm at 336 px (0.40 mm/px)
  AFTER  = the zoom trunk's input: the 77 mm crop at 384 px (0.20 mm/px)
Each network image = 3 consecutive slices in its 3 channels, all 3 shown side by side. Per study: the 4 views'
middle image (of 6). Studies: audit-58 (panel labels) with and without joint findings, plus training studies.

    python runs/others/crop_rois/save_zoom_samples.py [out_dir] [n_audit] [n_train]
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

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "joint_zoom_samples"
N_AU = int(sys.argv[2]) if len(sys.argv) > 2 else 10
N_TR = int(sys.argv[3]) if len(sys.argv) > 3 else 6
JOINT = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = importlib.util.spec_from_file_location("jz_train", ROOT / "runs" / "convnext_base_25d_v5_joint_zoom" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.cfg
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.build_model(cfg).to(dev).eval()
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    au = sc._eval_df(cfg, cfg["evaluation"]["audit"]["csv"])
    au = pd.concat([au[au[JOINT].max(axis=1) > 0.5].head(N_AU - N_AU // 3), au[au[JOINT].max(axis=1) < 0.5].head(N_AU // 3)])
    au["source"] = "audit (panel labels)"
    tr = pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg)).sample(N_TR, random_state=0)
    tr["source"] = "val (report labels)"
    df = pd.concat([au, tr]).reset_index(drop=True)
    ds = sc.KneeShardDataset(df, cfg, plan, train=False)
    OUT.mkdir(parents=True, exist_ok=True)
    slots, g, s, H = cfg["input"]["slots"], int(cfg["input"]["slices_as_channels"]), model.crop, cfg["input"]["size"]
    mean = torch.tensor(cfg["input"]["norm_mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["input"]["norm_std"], device=dev).view(1, 3, 1, 1)
    # EXACT network inputs: forward pre-hooks on the stems record the tensors the two backbones really receive
    got = {}
    model.loc_backbone.stem.register_forward_pre_hook(lambda m, inp: got.__setitem__("before", inp[0].detach()))
    model.backbone.stem.register_forward_pre_hook(lambda m, inp: got.__setitem__("after", inp[0].detach()))
    for i in range(len(df)):
        x, _ = ds[i]
        with torch.no_grad():
            model(sc.prepare_batch(x[None], cfg, dev))                  # the real forward, eval mode
        # undo only the ImageNet normalisation: the pixel values on screen are the network's input values, 0-1
        bef = (got["before"].float() * std + mean).clamp(0, 1).cpu().numpy()   # (24, 3, 336, 336)
        aft = (got["after"].float() * std + mean).clamp(0, 1).cpu().numpy()    # (24, 3, 384, 384)
        fig, ax = plt.subplots(2, 4, figsize=(30, 6))
        for k, sl in enumerate(slots):
            im = k * g + g // 2                                         # the slot's middle image (of 6)
            # each network image holds 3 consecutive slices as its 3 channels; shown: all 3 side by side
            ax[0, k].imshow(np.concatenate(list(bef[im]), axis=1), cmap="gray", vmin=0, vmax=1)
            ax[0, k].set_title(f"BEFORE {sl} img {g // 2 + 1}/6: 3 slices x {bef.shape[-1]} px (134 mm)", fontsize=9)
            ax[1, k].imshow(np.concatenate(list(aft[im]), axis=1), cmap="gray", vmin=0, vmax=1)
            ax[1, k].set_title(f"AFTER: 3 slices x {aft.shape[-1]} px (77 mm crop, 2x detail)", fontsize=9)
            for a_ in ax[:, k]:
                a_.axis("off")
        st = df.loc[i, "StudyInstanceUID"]
        labs = ", ".join(t for t in JOINT if df.loc[i, t] > 0.5) or "none of the joint findings"
        fig.suptitle(f"study ...{st[-10:]}  ({df.loc[i, 'source']}, knee side {ds.sides.get(st)})   "
                     f"joint findings: {labs}", fontsize=12)
        fig.tight_layout()
        fp = OUT / f"{i + 1:02d}_{st[-10:]}.png"
        fig.savefig(fp, dpi=80)
        plt.close(fig)
        print("saved", fp.name, "|", labs, flush=True)
    print("folder:", OUT)


if __name__ == "__main__":
    main()
