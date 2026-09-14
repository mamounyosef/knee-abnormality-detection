"""
grouped_25d_check.py - prove the non-overlapping 2.5D layout is what we think it is.

Three claims in convnext_base_25d_group3 are worth failing loudly rather than discovering
in a val score:

  1. The channels hold CONSECUTIVE, NON-OVERLAPPING slices: (0,1,2), (3,4,5), ... (21,22,23).
     prepare_batch gets this from a reshape, and a reshape is exactly the operation that
     silently produces a STRIDED grouping - (0,8,16), (1,9,17), ... - if the axis order is
     the other way round. Both shapes are (B*32, 3, H, W), so nothing crashes either way and
     the run would just quietly train on interleaved slices.
  2. The stem is the PRETRAINED one, bit for bit. The whole argument for this run is that
     in_chans lands on 3 so timm does no filter adaptation at all. If it ever resolved to
     something else, this becomes the fourth slice-channel run instead of the first 2.5D one.
  3. Normalisation is PER-CHANNEL, not the scalar the widened-stem runs use, and it matches
     what convnext_small_25d already does.

CPU only, no data and no GPU needed.

    python checks/grouped_25d_check.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUN = "convnext_base_25d_group3"
fails = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
i = cfg["input"]
g = int(i["slices_as_channels"])
n_sl, n_slots = int(i["n_slices"]), len(i["slots"])
per_img = n_sl // g

print("\n%s: %d slots x %d slices, %d images per slot, %d channels each"
      % (RUN, n_slots, n_sl, g, per_img))

# --- 1. config arithmetic ---------------------------------------------------------
print("\nconfig")
check("slices_as_channels divides n_slices", n_sl % g == 0, "%d / %d" % (n_sl, g))
check("channels per image is exactly 3 (pretrained RGB stem)", per_img == 3, "got %d" % per_img)
check("images_per_study is 32", sc.images_per_study(cfg) == 32,
      "got %d" % sc.images_per_study(cfg))
check("that is 3x cheaper than 96", sc.images_per_study(cfg) * 3 == n_slots * n_sl)
check("grad_clip is at least 3.0", float(cfg["training"]["grad_clip"]) >= 3.0,
      str(cfg["training"]["grad_clip"]))

# --- 2. the grouping is contiguous and non-overlapping ----------------------------
# Encode each slice by its INDEX so the grouping is readable straight off the tensor: give
# slice d the constant value d, then whatever lands in a channel names the slice it came from.
print("\ngrouping (each slice carries its own index as its pixel value)")
h = w = 4
x = torch.arange(n_sl, dtype=torch.uint8).view(1, 1, n_sl, 1, 1).expand(2, n_slots, n_sl, h, w)
out = sc.prepare_batch(x.contiguous(), cfg, "cpu")

check("flattened shape is (B*32, 3, H, W)", tuple(out.shape) == (2 * 32, 3, h, w),
      str(tuple(out.shape)))

# undo the normalisation to recover the slice indices
mean = np.array(i["norm_mean"], dtype=np.float64).reshape(1, 3, 1, 1)
std = np.array(i["norm_std"], dtype=np.float64).reshape(1, 3, 1, 1)
rec = (out.numpy().astype(np.float64) * std + mean) * 255.0
groups = np.rint(rec[:, :, 0, 0]).astype(int)          # (B*32, 3), one row per image

first_slot = groups[:g]                                 # the 8 images of study 0, slot 0
print("    study 0, slot 0 ->", [tuple(r) for r in first_slot])
want = np.array([[k * per_img + c for c in range(per_img)] for k in range(g)])
check("groups are (0,1,2), (3,4,5), ... (21,22,23)", np.array_equal(first_slot, want))
check("every slice appears exactly once per slot",
      sorted(first_slot.ravel().tolist()) == list(range(n_sl)))
check("no slice appears in two images (non-overlapping)",
      len(set(first_slot.ravel().tolist())) == n_sl)
check("all %d images across the batch are grouped the same way" % len(groups),
      all(np.array_equal(groups[k * g:(k + 1) * g], want)
          for k in range(len(groups) // g)))

# --- 3. normalisation took the per-channel path -----------------------------------
print("\nnormalisation")
# Image 0 holds slices 0, 1 and 2, whose pixel values ARE 0, 1 and 2 under this encoding,
# so the expected output is (d/255 - mean[d]) / std[d] channel by channel. The scalar branch
# would instead divide all three by the same averaged statistic.
ch0 = out[0, :, 0, 0].numpy().astype(np.float64)
raw = np.arange(per_img, dtype=np.float64) / 255.0
expect_per_ch = (raw - np.array(i["norm_mean"])) / np.array(i["norm_std"])
expect_scalar = (raw - np.mean(i["norm_mean"])) / np.mean(i["norm_std"])
check("per-channel statistics were applied", np.allclose(ch0, expect_per_ch, atol=1e-4),
      "got %s" % np.round(ch0, 4).tolist())
check("NOT the scalar widened-stem branch",
      not np.allclose(ch0, expect_scalar, atol=1e-4),
      "scalar branch would give %s" % np.round(expect_scalar, 4).tolist())

# --- 4. the stem is the pretrained one, untouched ---------------------------------
print("\nstem (downloads ConvNeXt-Base weights once, then compares)")
try:
    import timm

    name = cfg["model"]["name"]
    ref = timm.create_model(name, pretrained=True, num_classes=0)
    got = timm.create_model(name, pretrained=True, num_classes=0, in_chans=per_img)
    rw = ref.stem[0].weight.detach()
    gw = got.stem[0].weight.detach()
    check("stem shape unchanged by in_chans=3", tuple(rw.shape) == tuple(gw.shape),
          "%s vs %s" % (tuple(rw.shape), tuple(gw.shape)))
    d = (rw - gw).abs().max().item()
    check("stem weights are bit-identical (no adaptation)", d == 0.0, "max diff %.3e" % d)
except Exception as exc:
    check("stem comparison", False, "%s: %s" % (type(exc).__name__, exc))

print("\n" + "=" * 72)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("all checks passed - grouping is contiguous, stem is pretrained, norm is per-channel")
