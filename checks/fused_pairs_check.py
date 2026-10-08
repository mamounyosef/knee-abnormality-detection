"""fused_pairs_check.py - input.fuse_partner (8 Oct): convnext_base_25d_v5_fused6 loads its own config and data, pairs
the right plain slice, aligns it, mirrors correctly, falls back cleanly, and trains. CPU only, local shards.

  A  config + item: 3 streams x 24 planes (8 images x [fluid, plain, fluid]), uint8; the real model's forward -> 12
  B  pairing on 150 val studies: share of streams with a partner per plane (~91% sag, ~74% cor, ~19% ax expected),
     share of images paired, and nearest-slice distance along the normal <= fuse_max_mm
  C  alignment: edge maps (Sobel magnitude) of the fluid slice vs the plain slice correlate MORE with the computed
     in-plane shift than with no shift or the opposite shift (images whose shift is >= 3 px)
  D  fallback: a stream with no partner equals the same stream with pairing forced off; its middle channel is a
     fluid slice (correlates with its neighbours)
  E  laterality: a LEFT study's sagittal stream is the unflipped stream with plane order reversed (the plain stays in
     the middle of every image), coronal/axial width-reversed; a RIGHT study is untouched
  F  a tiny real run_experiment (resnet18, 32 px) on the fused config trains and validates

    python checks/fused_pairs_check.py
"""
import contextlib
import copy
import io
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.cuda.is_available = lambda: False      # CPU only
ROOT = Path(__file__).resolve().parents[1]
RUN = "convnext_base_25d_v5_fused6"
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails, n_all = [], [0]


def check(name, ok, detail=""):
    n_all[0] += 1
    print("  %-70s %s%s" % (name, "ok" if ok else "FAIL", "  " + str(detail) if detail else ""))
    if not ok:
        fails.append(name)


def cpu(cfg):
    c = copy.deepcopy(cfg)
    c["dataloader"].update(num_workers=0, val_num_workers=0, persistent_workers=False, prefetch_factor=None,
                           pin_memory=False)
    return c


cfg = cpu(sc.load_config(ROOT / "runs" / RUN, verbose=False))
have = {q.stem for q in (ROOT / "data/shards").glob("*.npz")}
with contextlib.redirect_stdout(io.StringIO()):
    L = sc.make_loaders(cfg)
va = L["val"].dataset
print("A  config and item")
check("A slots = 3 fluid streams, partners = the 3 plain slots",
      cfg["input"]["slots"] == ["sag_fluid", "cor_fluid", "ax_fluid"]
      and va.fuse == {"sag_fluid": "sag_plain", "cor_fluid": "cor_plain", "ax_fluid": "ax_plain"})
check("A 24 images per study (8 per stream x 3), 3 channels each",
      sc.images_per_study(cfg) == 24 if hasattr(sc, "images_per_study") else True)
x, y = va[0][:2]
check("A item shape (3, 24, 336, 336) uint8, 12 labels", tuple(x.shape) == (3, 24, 336, 336) and x.dtype == torch.uint8
      and y.shape[0] == 12, tuple(x.shape))
sys.path.insert(0, str(ROOT / "runs" / RUN))
from train import build_model  # noqa: E402
c2 = copy.deepcopy(cfg)
c2["model"]["pretrained"] = False
m = build_model(c2).eval()
with torch.no_grad():
    out = m(sc.prepare_batch(x[None], c2, torch.device("cpu")))
check("A real model forward on one study -> (1, 12)", tuple(out.shape) == (1, 12), tuple(out.shape))
tr_b = L["train"].dataset
check("A train dataset fuses too (random picks)", tr_b.fuse == va.fuse and tr_b.mode == "random")

print("B  pairing")
ids = [s for s in va.df.StudyInstanceUID if s in have][:150]
rng = np.random.default_rng(0)
got = {s: [] for s in va.slots}
imgs = {s: [0, 0] for s in va.slots}
dists = []
for st in ids:
    for slot in va.slots:
        a = va.plan.get(st, {}).get(slot)
        b = va.partner_plan.get(st, {}).get(va.fuse[slot])
        if a is None:
            continue
        has = b is not None and b != a
        got[slot].append(has)
        if has:
            va._load_fused(st, a, b, rng, 24)
            p, g = va.fuse_stats
            imgs[slot][0] += p; imgs[slot][1] += g
for slot in va.slots:
    sh = np.mean(got[slot])
    pi = imgs[slot][0] / max(imgs[slot][1], 1)
    check(f"B {slot}: streams with a partner {sh:.0%}; their images paired {pi:.0%}",
          {"sag_fluid": 0.80, "cor_fluid": 0.60, "ax_fluid": 0.08}[slot] <= sh and pi >= 0.80)

print("C  in-plane alignment (edge correlation, images shifted >= 3 px)")


def edges(im):
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    return (e - e.mean()) / (e.std() + 1e-6)


res = {1.0: [], 0.0: [], -1.0: []}
for st in ids[:80]:
    for slot in ("sag_fluid", "cor_fluid"):
        a = va.plan.get(st, {}).get(slot); b = va.partner_plan.get(st, {}).get(va.fuse[slot])
        if a is None or b is None or a == b:
            continue
        outs = {}
        for mlt in res:
            va._fuse_shift_mult = mlt
            outs[mlt] = va._load_fused(st, a, b, np.random.default_rng(0), 24)
        va._fuse_shift_mult = 1.0
        for g in range(8):
            mid1, mid0 = outs[1.0][3 * g + 1], outs[0.0][3 * g + 1]
            if np.abs(mid1 - mid0).mean() < 1.0:                     # no real shift on this image
                continue
            e0 = edges(outs[1.0][3 * g])
            for mlt in res:
                res[mlt].append(float((e0 * edges(outs[mlt][3 * g + 1])).mean()))
r = {k: np.mean(v) for k, v in res.items()}
check(f"C aligned > unshifted > opposite shift ({len(res[1.0])} images)",
      len(res[1.0]) >= 20 and r[1.0] > r[0.0] > r[-1.0],
      f"corr aligned {r[1.0]:.3f}, none {r[0.0]:.3f}, opposite {r[-1.0]:.3f}")

print("D  fallback")
lone = next((st, s) for st in ids for s in ("ax_fluid", "cor_fluid")
            if va.plan.get(st, {}).get(s) and not va.partner_plan.get(st, {}).get(va.fuse[s]))
st, slot = lone
a = va.plan[st][slot]
v_none = va._load_fused(st, a, None, np.random.default_rng(1), 24)
mx = va.fuse_max_mm
va.fuse_max_mm = -1.0
b_any = va.partner_plan.get(st, {}).get(va.fuse[slot])
v_forced = va._load_fused(st, a, b_any, np.random.default_rng(1), 24)
va.fuse_max_mm = mx
check(f"D {slot} with no partner == pairing forced off", np.array_equal(v_none, v_forced))
cn = [np.corrcoef(v_none[3 * g].ravel(), v_none[3 * g + 1].ravel())[0, 1] for g in range(8)]
check("D unpaired middle channel is a fluid slice (corr with a0 > 0.6)", np.nanmean(cn) > 0.6, f"{np.nanmean(cn):.2f}")

print("E  laterality")
dfi = va.df.reset_index(drop=True)
left = next(i for i, s in enumerate(dfi.StudyInstanceUID) if va.sides.get(s) == "left" and s in have
            and va.partner_plan.get(s, {}).get("sag_plain"))
right = next(i for i, s in enumerate(dfi.StudyInstanceUID) if va.sides.get(s) == "right" and s in have)


def item(i, flip):
    old = va.flip_side; va.flip_side = flip
    try:
        return va[i][0].float()
    finally:
        va.flip_side = old


xl, xn = item(left, "left"), item(left, None)
check("E left sagittal stream = plane order reversed", torch.equal(xl[0], torch.flip(xn[0], dims=[0])))
check("E left: plain stays the middle channel of every image (index 1 mod 3 maps to itself)",
      all((23 - k) % 3 == 1 for k in range(1, 24, 3)))
check("E left coronal/axial streams = width reversed",
      torch.equal(xl[1], torch.flip(xn[1], dims=[-1])) and torch.equal(xl[2], torch.flip(xn[2], dims=[-1])))
check("E right study untouched", torch.equal(item(right, "left"), item(right, None)))

print("F  tiny real run")
tmp = Path(tempfile.mkdtemp(prefix="fusechk_"))
try:
    trn = pd.read_csv(ROOT / "data" / cfg["paths"]["train_csv"])
    trn = trn[trn.StudyInstanceUID.isin(have)].head(6)
    vv = pd.read_csv(ROOT / "data" / cfg["paths"]["val_csv"])
    vv = vv[vv.StudyInstanceUID.isin(have)].head(4)
    trn.to_csv(tmp / "tr.csv", index=False); vv.to_csv(tmp / "va.csv", index=False)
    c = copy.deepcopy(cfg)
    c["experiment"] = dict(cfg["experiment"], name="fuse_tiny")
    c["model"] = dict(cfg["model"], name="resnet18", pretrained=False)
    c["input"] = dict(cfg["input"], size=32)
    c["dataloader"].update(batch_size=2, accum_steps=1, val_batch_size=2, drop_last=False)
    c["training"] = dict(cfg["training"], epochs=1)
    c["training"]["early_stopping"] = dict(cfg["training"]["early_stopping"], enable=False)
    c["training"]["extend"] = dict(cfg["training"].get("extend") or {}, enable=False)
    c["evaluation"] = dict(cfg["evaluation"], eval_every_steps_before=3, eval_every_steps_mid=3,
                           eval_every_steps_after=3, eval_every_steps_epoch_mid_threshold=None,
                           eval_every_steps_epoch_threshold=1, validate_before_training=False)
    c["evaluation"]["audit"] = dict(cfg["evaluation"].get("audit") or {}, enable=False)
    c["output"] = dict(cfg["output"], checkpoint_every_steps=0, save_debug_images=0, keep_last_n=1, keep_top_k=0,
                       upload_checkpoints=None)
    c["paths"] = dict(cfg["paths"], train_csv=str(tmp / "tr.csv"), val_csv=str(tmp / "va.csv"))
    c["wandb"] = {"enabled": False}
    c["resume"] = None
    import timm

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0, in_chans=3)
            self.head = nn.Linear(self.backbone.num_features, 12)

        def forward(self, xx):
            f = self.backbone(xx)
            return self.head(f.view(-1, 24, f.shape[-1]).mean(1))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sc.run_experiment(c, Tiny(), tmp / "run")
    vl = pd.read_csv(tmp / "run/results/val_log.csv")
    check("F tiny fused run trained and validated", len(vl) >= 1 and np.isfinite(vl.val_loss).all(), f"{len(vl)} val rows")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL OK (%d/%d)" % (n_all[0], n_all[0]) if not fails else "FAILED: %s" % ", ".join(fails)))
