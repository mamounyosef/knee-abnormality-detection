"""cam_preview.py - where would a two-stage crop model look? Draws the localizer's class-activation map (CAM) and the
crop box on real studies, so the crops can be checked by eye BEFORE any training.

Localizer = convnext_base_25d_18slices_v5_aug_base (top_step5900), the 2.5D parent every specialist branches from.
Its head is linear over (mean over images of LayerNorm(global-avg-pool(feature map))), so each feature-map cell's
contribution to a finding's logit is EXACT, no gradients: cam_i(h,w) = W_j . ((f_i(h,w) - mu_i) / sigma_i * gamma + beta),
with mu_i, sigma_i the LayerNorm statistics of image i's pooled vector (shared_code-free maths, see cam_maps below).

The study is loaded ONCE at 0.20 mm/px, 672 px (the same 134 mm frame at twice the detail); the localizer sees it
area-downsampled to 336 px = exactly its training view; the crop is cut from the 672 px frame.

    python runs/others/crop_rois/cam_preview.py [group] [n_studies]      group: lateral | pf   (default lateral, 8)
Output: runs/others/crop_rois/preview_<group>/<study>.png
"""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

PARENT = "convnext_base_25d_18slices_v5_aug_base"
CKPT = ROOT / "runs" / PARENT / "results" / "checkpoints" / "top_step5900.pt"
GROUPS = {   # findings whose CAM places the crop, and the views they are read on (as the v5 specialists)
    "lateral": (["Lateral Meniscus", "Lateral OA"], ["cor_fluid", "cor_plain", "sag_fluid", "sag_plain"]),
    "pf": (["PF OA"], ["ax_fluid", "ax_plain", "sag_plain", "sag_fluid"]),
}
HI, LO, CROP = 672, 336, 336
MODE = sys.argv[3] if len(sys.argv) > 3 else "finding"   # finding = the group's evidence | abs_all = anatomy          # hi-res frame, localizer view, crop side (px at 0.20 mm = 67 mm)


def load_parent():
    spec = importlib.util.spec_from_file_location("parent_train", ROOT / "runs" / PARENT / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = sc.load_config(ROOT / "runs" / PARENT, verbose=False)
    model = mod.build_model(cfg)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    sc._load_model_state(model, sd)
    return model.eval(), cfg


@torch.no_grad()
def cam_maps(model, xin, rows):
    """xin (N, 3, 336, 336) normalised -> (N, h, w) summed CAM of the given head rows (logit units per cell)."""
    bb = model.backbone
    f = bb.forward_features(xin)                       # (N, F, h, w)
    pooled = f.mean((2, 3))                            # global average pool, (N, F)
    ln = bb.head.norm                                  # timm ConvNeXt: pool -> LayerNorm -> flatten
    mu = pooled.mean(1, keepdim=True)
    sig = torch.sqrt(pooled.var(1, unbiased=False, keepdim=True) + ln.eps)
    fn = (f - mu[:, :, None, None]) / sig[:, :, None, None] * ln.weight[None, :, None, None] + ln.bias[None, :, None, None]
    W = model.head.weight[rows].sum(0)                 # (F,)
    cam = torch.einsum("nfhw,f->nhw", fn, W)
    if MODE == "abs_all":
        # anatomy, not evidence: how strongly each cell moves ANY finding's logit, either way
        cam_all = torch.einsum("nfhw,kf->knhw", fn, model.head.weight).abs().sum(0)
    # sanity: the cells average back to the pooled logit contribution (bias aside)
    ref = (bb.forward_head(f, pre_logits=True) * W).sum(1)      # timm's own pool + norm path
    assert torch.allclose(cam.mean((1, 2)), ref, atol=1e-3 * max(1.0, ref.abs().max().item())), "CAM does not decompose"
    return cam_all if MODE == "abs_all" else cam


def crop_centre(cam_slot, grid):
    """cam (images, h, w) -> (cy, cx) in 672 px: centroid of the positive part above the slot's median."""
    c = cam_slot.sum(0)
    c = torch.clamp(c - c.median(), min=0)
    if c.sum() <= 0:
        return HI / 2, HI / 2
    h, w = c.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    cy, cx = float((c * ys).sum() / c.sum()), float((c * xs).sum() / c.sum())
    s = LO / grid                                      # px per cell at 336
    return (cy + 0.5) * s * HI / LO, (cx + 0.5) * s * HI / LO


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    group = sys.argv[1] if len(sys.argv) > 1 else "lateral"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    finds, slots = GROUPS[group]
    model, pcfg = load_parent()
    rows = [pcfg["tasks"].index(t) for t in finds]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    cfg = copy.deepcopy(pcfg)
    cfg["input"].update(slots=slots, size=HI, mm_per_px=0.20)
    lcfg = copy.deepcopy(cfg)
    lcfg["input"]["resize_to"] = LO
    au = sc._eval_df(cfg, cfg["evaluation"]["audit"]["csv"])
    pos = au[au[finds].max(1) > 0.5].head(n - n // 4)
    neg = au[au[finds].max(1) < 0.5].head(n // 4)
    df = pd.concat([pos, neg]).reset_index(drop=True)
    plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
    ds = sc.KneeShardDataset(df, cfg, plan, train=False)
    out = Path(__file__).resolve().parent / f"preview_{group}_{MODE}"
    out.mkdir(exist_ok=True)
    g = int(cfg["input"]["slices_as_channels"])
    for i in range(len(df)):
        x, y = ds[i]                                   # (slots, 18, 672, 672) uint8
        xin = sc.prepare_batch(x[None], lcfg, dev)     # (slots*6, 3, 336, 336)
        cam = cam_maps(model, xin, rows).float().cpu()
        grid = cam.shape[-1]
        cam = cam.view(len(slots), g, grid, grid)
        fig, ax = plt.subplots(2, len(slots), figsize=(4 * len(slots), 8))
        for k, sl in enumerate(slots):
            cy, cx = crop_centre(cam[k], grid)
            top = int(cam[k].flatten(1).max(1).values.argmax())      # image with the strongest evidence
            img = x[k, top * 3 + 1].numpy()
            ax[0, k].imshow(img, cmap="gray")
            hm = torch.nn.functional.interpolate(cam[k].sum(0)[None, None], size=(HI, HI), mode="bilinear")[0, 0]
            ax[0, k].imshow(hm, cmap="jet", alpha=0.35)
            y0, x0 = int(np.clip(cy - CROP / 2, 0, HI - CROP)), int(np.clip(cx - CROP / 2, 0, HI - CROP))
            ax[0, k].add_patch(plt.Rectangle((x0, y0), CROP, CROP, fill=False, ec="lime", lw=2))
            ax[0, k].set_title(f"{sl} img {top}", fontsize=9)
            ax[1, k].imshow(img[y0:y0 + CROP, x0:x0 + CROP], cmap="gray")
            ax[1, k].set_title("crop 67 mm @0.2", fontsize=9)
            for a in ax[:, k]:
                a.axis("off")
        lab = ", ".join(f"{t}={int(df.loc[i, t] > 0.5)}" for t in finds)
        st = df.loc[i, "StudyInstanceUID"]
        fig.suptitle(f"{st[-12:]}  side={ds.sides.get(st)}  {lab}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{i:02d}_{st[-12:]}.png", dpi=70)
        plt.close(fig)
        print(i, st[-12:], lab, flush=True)
    print("saved to", out)


if __name__ == "__main__":
    main()
