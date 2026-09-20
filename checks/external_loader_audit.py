"""external_loader_audit.py - hunt for the Qwen bug in every other external backbone.

THE BUG CLASS. Loading foreign weights needs judgement calls that produce a tensor of
exactly the right shape whichever way you decide: which interpolation mode resamples a
position table, whether antialiasing is on when downsampling it, what normalisation the
weights were trained under. Nothing raises. The model trains. The number is just worse than
it should be, and it looks like the backbone was a bad idea.

That is what happened with Qwen: its own vision model resamples the position table with
BILINEAR / align_corners=True, and the loader used bicubic / align_corners=False carried
over from the SAM loader. The worst position was off by 5.9 standard deviations.

This audit re-derives each answer from the AUTHORITY - timm's own resampling function, the
upstream repository, the published preprocessing config - rather than from what the loader
already does, and quantifies any gap.

Run:  python checks/external_loader_audit.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed_nhwc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import external_backbones as eb  # noqa: E402
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %s %-58s %s" % ("PASS" if ok else "FAIL", name, detail))


PRE = ROOT / "data" / "pretrained"

print("\n1. MedSAM / SAM absolute position embedding")
print("   AUTHORITY: timm.layers.resample_abs_pos_embed_nhwc, which is what timm's own")
print("   samvit checkpoint_filter_fn calls. Its defaults are bicubic AND antialias=True.")
ck_path = PRE / "medsam_vit_b.pth"
if ck_path.exists():
    raw = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    pe = None
    for k, v in raw.items():
        if k.endswith("pos_embed") and getattr(v, "ndim", 0) == 4:
            pe = v
            break
    if pe is None:
        print("      no 4-d pos_embed in the checkpoint")
    else:
        print("      checkpoint pos_embed %s" % (tuple(pe.shape),))
        for px in (336, 512):
            t = px // 16
            ours = eb._resample_pos_embed(pe, (t, t))
            theirs = resample_abs_pos_embed_nhwc(pe, [t, t])
            d = (ours - theirs).abs()
            ck("%dpx matches timm's own resampling" % px, float(d.max()) < 1e-5,
               "max diff %.4f = %.0f%% of one std (%.4f)"
               % (d.max(), 100 * d.max() / pe.std(), pe.std()))
else:
    print("      medsam_vit_b.pth not present locally, skipped")

print("\n2. SAM relative position tables")
print("   AUTHORITY: segment_anything/modeling/image_encoder.py get_rel_pos, which calls")
print('   F.interpolate(..., mode="linear") with align_corners left at its default.')
rp = torch.randn(127, 64)
ours = eb._resample_rel_pos(rp, 41)
theirs = F.interpolate(rp.reshape(1, rp.shape[0], -1).permute(0, 2, 1), size=41,
                       mode="linear").permute(0, 2, 1).reshape(41, -1)
ck("matches SAM's own get_rel_pos", torch.allclose(ours, theirs, atol=1e-6),
   "max diff %.2e" % (ours - theirs).abs().max().item())

print("\n3. Qwen3.5 position table (the bug that started this audit)")
qck = PRE / "qwen3_5_vision_vitb16.safetensors"
if qck.exists():
    from safetensors.torch import load_file
    qpe = load_file(str(qck))["model.visual.pos_embed.weight"]
    g, t = 48, 21
    x = qpe.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
    theirs = F.interpolate(x, size=(t, t), mode="bilinear", align_corners=True)
    theirs = theirs.permute(0, 2, 3, 1).reshape(t * t, -1)
    ours = eb._qwen_pos_embed(qpe, t * t).reshape(t * t, -1)
    ck("matches Qwen's bilinear / align_corners=True",
       torch.allclose(ours, theirs, atol=1e-6),
       "max diff %.2e" % (ours - theirs).abs().max().item())
else:
    print("      weights not present locally, skipped")

print("\n4. normalisation: what each run declares vs what its weights actually saw")
print("   %-34s %-26s %s" % ("run", "config mean", "what the source says"))
TRUTH = {
    # run                          (expected mean, why we believe it)
    "medsam_vit_b_336": ([0.485, 0.456, 0.406],
                         "MedSAM inherits Meta SAM's ImageNet pixel_mean/pixel_std"),
    "beitv2_base_336": (None, "timm pretrained_cfg decides; engine check enforces it"),
    "barlowtwins_resnet50_336": ([0.485, 0.456, 0.406],
                                 "Barlow Twins trains on ImageNet with torchvision stats"),
    "triad_plainconv3d_320": ([0.5], "ASSUMED - the card states no preprocessing"),
    "qwen3_5_vision_tower_336": ([0.5, 0.5, 0.5],
                                 "Qwen preprocessor_config.json, verbatim"),
}
for run, (want, why) in TRUTH.items():
    d = ROOT / "runs" / run
    if not d.exists():
        continue
    cfg = sc.load_config(d, verbose=False)
    got = list(cfg["input"].get("norm_mean") or [])
    print("   %-34s %-26s %s" % (run, str(got), why))
    if want is not None:
        ck("  %s normalisation matches its source" % run[:40],
           [round(v, 3) for v in got] == [round(v, 3) for v in want],
           "config %s vs expected %s" % (got, want))

print("\n5. Triad: the architecture was reimplemented, so its shape is an assumption")
print("   Its strides (1,2,2,2,2,2) and channels (32,64,128,256,320,320) were taken from")
print("   the published nnU-Net PlainConvEncoder. The tensor counter proves every weight")
print("   LANDED, not that it landed in the right place: a wrong stride still loads.")
tck = PRE / "triad_plainconvunet_mae.safetensors"
if tck.exists():
    from safetensors.torch import load_file as _lf
    t = _lf(str(tck))
    # the conv weight shapes themselves encode the channel plan, so check against them
    chans = []
    for i in range(len(eb.TRIAD_CHANNELS)):
        k = "encoder.stages.%d.convs.0.conv.weight" % i
        if k in t:
            chans.append(t[k].shape[0])
    ck("channel plan matches the checkpoint's own conv shapes",
       tuple(chans) == eb.TRIAD_CHANNELS if chans else False,
       "checkpoint %s vs code %s" % (tuple(chans), eb.TRIAD_CHANNELS))
    ck("Triad normalisation is FLAGGED as assumed, not silently trusted",
       "ASSUM" in Path(ROOT / "runs" / "triad_plainconv3d_320" / "config.yaml")
       .read_text(encoding="utf-8").upper())
else:
    print("      weights not present locally, skipped")

print("\n" + "=" * 74)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
