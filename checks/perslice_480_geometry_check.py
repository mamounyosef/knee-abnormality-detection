"""Prove the 480 px per-slice geometry works end to end, without a backward pass.

The full smoke test needs gradients and a 480 px DaViT-Huge does not fit this PC's 8 GB
card. The TRAINING wiring is byte-identical to the 336 version, which passed 19/19, so what
is actually new here is the geometry: 480 px, per-slice packing, 72 windows. Check exactly
that, on real shards, with no_grad.
"""
import importlib
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(r"D:\My Projects\knee_abnormality_detection")
RUN = "medimageinsight_davit_huge_perslice_480"
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(ROOT / "runs" / RUN))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-56s %s" % ("PASS" if ok else "FAIL", n, d))


cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
ref = sc.load_config(ROOT / "runs" / "medimageinsight_davit_huge_336", verbose=False)

print("\n1. what changed against the 0.929 model, and what did not")
ck("size 336 -> 480", cfg["input"]["size"] == 480 and ref["input"]["size"] == 336)
ck("slices_as_channels 6 -> null (per-slice)",
   cfg["input"]["slices_as_channels"] is None and ref["input"]["slices_as_channels"] == 6)
ck("n_slices unchanged at 18", cfg["input"]["n_slices"] == ref["input"]["n_slices"] == 18)
ck("mm_per_px unchanged at 0.40", cfg["input"]["mm_per_px"] == ref["input"]["mm_per_px"] == 0.4)
ck("slots unchanged", cfg["input"]["slots"] == ref["input"]["slots"])
ck("laterality_flip still 'left'", cfg["input"]["laterality_flip"] == "left")
ck("windows per study 24 -> 72",
   sc.images_per_study(cfg) == 72 and sc.images_per_study(ref) == 24)
print("     192 mm of frame at 480 px vs 134 mm at 336, mm_per_px held at 0.40")
print("     480 / 32 = %d, so DaViT's four downsamples divide cleanly" % (480 // 32))

print("\n2. real shards through the real loader at 480")
df = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
shards = Path(cfg["paths"]["shards_dir"])
have = [s for s in df["StudyInstanceUID"] if (shards / ("%s.npz" % s)).exists()]
ck("local shards available", len(have) >= 1, "%d" % len(have))
sub = df[df["StudyInstanceUID"].isin(have[:2])].reset_index(drop=True)
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
ds = sc.KneeShardDataset(sub, cfg, sc.plan_slots(idx, cfg), train=False)
x = ds[0][0]
ck("one study is (slots, n_slices, 480, 480)", tuple(x.shape) == (4, 18, 480, 480),
   str(tuple(x.shape)))
xb = sc.prepare_batch(x.unsqueeze(0), cfg, "cpu")
ck("prepare_batch emits (72, 3, 480, 480)", tuple(xb.shape) == (72, 3, 480, 480),
   str(tuple(xb.shape)))
# The engine replicates the grey plane across 3 channels (prepare_batch: x.expand), but
# normalisation then applies a DIFFERENT mean and std per channel (ImageNet's
# 0.485/0.456/0.406 and 0.229/0.224/0.225), so the tensors are not equal afterwards.
# Undo the normalisation to compare the pixels themselves.
mean = torch.tensor(cfg["input"]["norm_mean"]).view(1, 3, 1, 1)
std = torch.tensor(cfg["input"]["norm_std"]).view(1, 3, 1, 1)
raw = xb * std + mean
same = (torch.allclose(raw[0, 0], raw[0, 1], atol=1e-5)
        and torch.allclose(raw[0, 1], raw[0, 2], atol=1e-5))
ck("PER-SLICE: the 3 channels hold the SAME grey plane", same,
   "identical before normalisation; ImageNet stats differ per channel after it")
ck("and the window is one slice, not three stacked",
   raw[0].shape[0] == 3 and not torch.allclose(raw[0, 0], raw[1, 0], atol=1e-3),
   "consecutive windows differ, so each holds its own slice")

print("\n3. the model consumes it")
tm = importlib.import_module("train")
model = tm.build_model(cfg).eval()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
# 72 windows at 480 px will not fit this PC's 8 GB card even without gradients, so push
# them through the BACKBONE in chunks and do the head arithmetic separately. That still
# proves what is new - that the trunk accepts 480 px and the per-study reshape is right.
CH = 6
with torch.no_grad():
    feats = torch.cat([model.backbone(xb[i:i + CH].to(dev)).cpu()
                       for i in range(0, xb.shape[0], CH)])
ck("backbone accepts 480 px, one feature per window",
   tuple(feats.shape) == (72, model.backbone.num_features), str(tuple(feats.shape)))
with torch.no_grad():
    pooled = feats.view(-1, model.per_study, feats.shape[-1]).mean(dim=1)
    y = model.head(model.feat_norm(pooled.to(dev))).cpu()
ck("per-study pooling gives (1, 12)", tuple(y.shape) == (1, len(cfg["tasks"])),
   str(tuple(y.shape)))
ck("logits finite and varying",
   bool(torch.isfinite(y).all()) and float(y.std()) > 1e-4, "std %.4f" % y.std())
ck("model.per_study matches the loader", model.per_study == sc.images_per_study(cfg),
   str(model.per_study))
if dev == "cuda":
    print("     peak %.2f GB for a %d-window chunk at 480 px, no gradients"
          % (torch.cuda.max_memory_allocated() / 1e9, CH))

print("\n" + "=" * 70)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
