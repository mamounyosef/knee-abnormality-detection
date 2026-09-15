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

RUN = sys.argv[1] if len(sys.argv) > 1 else "convnext_base_25d_group3"
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
check("images_per_study is slots x groups", sc.images_per_study(cfg) == n_slots * g,
      "got %d, expected %d" % (sc.images_per_study(cfg), n_slots * g))
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

# resize_to, when set, changes the SPATIAL size inside prepare_batch (coatnet_rmlp2_384
# loads at 512 and hands the backbone 384), so compare against model_input_size, not the
# size the fake batch was built at.
exp_hw = sc.model_input_size(cfg) if i.get("resize_to") else h
check("flattened shape is (B*%d, 3, %d, %d)" % (n_slots * g, exp_hw, exp_hw),
      tuple(out.shape) == (2 * n_slots * g, 3, exp_hw, exp_hw),
      str(tuple(out.shape)))

# undo the normalisation to recover the slice indices
mean = np.array(i["norm_mean"], dtype=np.float64).reshape(1, 3, 1, 1)
std = np.array(i["norm_std"], dtype=np.float64).reshape(1, 3, 1, 1)
rec = (out.numpy().astype(np.float64) * std + mean) * 255.0
groups = np.rint(rec[:, :, 0, 0]).astype(int)          # (B*32, 3), one row per image

first_slot = groups[:g]                                 # the 8 images of study 0, slot 0
print("    study 0, slot 0 ->", [tuple(r) for r in first_slot])
want = np.array([[k * per_img + c for c in range(per_img)] for k in range(g)])
check("groups are contiguous: %s ... %s" % (tuple(want[0]), tuple(want[-1])),
      np.array_equal(first_slot, want))
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
# A checkpoint whose three norm values are EQUAL (0.5/0.5/0.5, the TensorFlow lineage that
# coatnet_rmlp2_384 uses) makes the two branches numerically identical, so this test cannot
# distinguish them and asserting it would be a false alarm.
if len(set(i["norm_mean"])) == 1 and len(set(i["norm_std"])) == 1:
    print("    %-52s n/a  (mean/std are uniform: %s)"
          % ("scalar-vs-per-channel", i["norm_mean"]))
else:
    check("NOT the scalar widened-stem branch",
          not np.allclose(ch0, expect_scalar, atol=1e-4),
          "scalar branch would give %s" % np.round(expect_scalar, 4).tolist())

# --- 4. the stem is the pretrained one, untouched ---------------------------------
print("\ninput layer (downloads the backbone weights once, then compares)")
try:
    import timm

    name = cfg["model"]["name"]
    ref = timm.create_model(name, pretrained=True, num_classes=0)
    got = timm.create_model(name, pretrained=True, num_classes=0, in_chans=per_img)

    def first_conv(m):
        """The layer that actually eats the input channels.

        ConvNeXt calls it stem[0], a ViT calls it patch_embed.proj, and both are a plain
        Conv2d(in_chans, dim, kernel=patch, stride=patch). Finding it by type rather than by
        name keeps this check valid for any backbone we try next.
        """
        import torch.nn as nn
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                return mod
        raise AttributeError("no Conv2d found - backbone takes its input some other way")

    rc, gc = first_conv(ref), first_conv(got)
    print("    input layer: %s kernel=%s out=%d" % (type(gc).__name__, gc.kernel_size,
                                                    gc.out_channels))
    k = gc.kernel_size[0] * gc.kernel_size[1] * per_img
    print("    budget per patch: %d -> %d  (%.2fx %s)"
          % (k, gc.out_channels, max(k, gc.out_channels) / min(k, gc.out_channels),
             "compression" if k > gc.out_channels else "expansion"))
    rw = rc.weight.detach()
    gw = gc.weight.detach()
    check("input-layer shape unchanged by in_chans=3", tuple(rw.shape) == tuple(gw.shape),
          "%s vs %s" % (tuple(rw.shape), tuple(gw.shape)))
    d = (rw - gw).abs().max().item()
    check("input-layer weights are bit-identical (no adaptation)", d == 0.0, "max diff %.3e" % d)
except Exception as exc:
    check("stem comparison", False, "%s: %s" % (type(exc).__name__, exc))

# --- 5. the RUN'S OWN model, built through its train.py -------------------------------
# Everything above builds timm directly, which is exactly how coatnet_rmlp2_384 slipped
# through: its CONFIG was converted to the 2.5D layout but its train.py still hardcoded
# per_study = slots * n_slices, so it built a 72-image model, was handed 24, and died on
# "shape [-1, 72, 1024] is invalid for input of size 98304" - after the GPU container had
# started. A real forward pass through the run's own build_model is the only thing that
# catches a config and a train.py that disagree.
print("\nthe run's own model (train.py -> forward pass)")
try:
    import importlib.util

    spec = importlib.util.spec_from_file_location("_t", ROOT / "runs" / RUN / "train.py")
    _t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_t)
    small = sc.load_config(ROOT / "runs" / RUN, verbose=False)
    # Use the run's REAL resolution. Shrinking to a convenient number breaks backbones whose
    # stride or window does not divide it: 112 gives CoAtNet's relative-position tables a
    # 3.5 grid and it fails with "size of tensor a (9) must match tensor b (16)". Batch 1
    # keeps a full-resolution CPU pass tractable.
    small["model"]["pretrained"] = False
    mdl = _t.build_model(small).eval()
    px = int(small["input"]["size"])
    want = sc.images_per_study(small)
    check("train.py's per_study matches images_per_study",
          getattr(mdl, "per_study", None) == want,
          "model says %s, engine says %d" % (getattr(mdl, "per_study", None), want))
    xb = torch.randint(0, 255, (1, n_slots, n_sl, px, px), dtype=torch.uint8)
    pbx = sc.prepare_batch(xb, small, "cpu")
    print("    %d images at %d px through %s ..." % (pbx.shape[0], pbx.shape[-1],
                                                     small["model"]["name"].split(".")[0]))
    with torch.no_grad():
        o = mdl(pbx)
    check("forward pass returns (B, n_tasks)",
          tuple(o.shape) == (1, sc.num_output_logits(small)), str(tuple(o.shape)))
    check("logits are finite", bool(torch.isfinite(o).all()))
except Exception as exc:
    check("the run's own model builds and runs", False,
          "%s: %s" % (type(exc).__name__, exc))

print("\n" + "=" * 72)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("all checks passed - grouping is contiguous, stem is pretrained, norm is per-channel")
