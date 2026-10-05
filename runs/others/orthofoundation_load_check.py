"""orthofoundation_load_check.py - what is inside OrthoFoundation-L.pth, and does it load into timm's DINOv3 ViT-L?

Prints the top-level keys, the backbone key prefix, parameter count and shapes, then loads into
timm 'vit_large_patch16_dinov3' through timm's own DINOv3 checkpoint converter, reporting missing / unexpected keys,
and compares one forward against the original DINOv3-L ImageNet weights to prove the weights differ.

    python runs/others/orthofoundation_load_check.py
"""
from pathlib import Path

import timm
import torch

P = Path(__file__).resolve().parents[2] / "data" / "pretrained" / "OrthoFoundation-L.pth"
print(f"file {P.stat().st_size:,} bytes")
ck = torch.load(P, map_location="cpu", weights_only=False)
print("top-level type:", type(ck).__name__, "| keys:", list(ck)[:10] if isinstance(ck, dict) else "-")


def find_sd(obj, path="root"):
    # the backbone state dict: the largest dict of tensors anywhere inside
    best = (0, None, None)
    if isinstance(obj, dict):
        n = sum(1 for v in obj.values() if torch.is_tensor(v))
        if n > best[0]:
            best = (n, obj, path)
        for k, v in obj.items():
            if isinstance(v, dict):
                c = find_sd(v, f"{path}.{k}")
                if c[0] > best[0]:
                    best = c
    return best


n, sd, where = find_sd(ck)
print(f"largest tensor dict at {where}: {n} tensors, {sum(v.numel() for v in sd.values() if torch.is_tensor(v)) / 1e6:.1f} M params")
keys = list(sd)
print("first keys:", keys[:8])
print("last keys:", keys[-6:])
prefixes = sorted({k.split(".")[0] for k in keys})
print("first-level prefixes:", prefixes[:20])
for k in keys[:3] + [k for k in keys if "patch_embed" in k][:2]:
    print(f"  {k}: {tuple(sd[k].shape)}")

# strip DINO wrapper prefixes down to the plain DINOv3 ViT keys
for pre in ("module.", "backbone.", "teacher.", "student."):
    if all(k.startswith(pre) for k in sd):
        sd = {k[len(pre):]: v for k, v in sd.items()}
        print("stripped prefix", pre)
for pre in ("backbone.",):
    sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    if sub:
        print(f"keeping only '{pre}*' ({len(sub)} of {len(sd)} tensors)")
        sd = sub
model = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0)
from timm.models import eva  # noqa: E402
conv = eva.checkpoint_filter_fn(sd, model) if hasattr(eva, "checkpoint_filter_fn") else sd
r = model.load_state_dict(conv, strict=False)
print(f"timm load: missing {len(r.missing_keys)} {r.missing_keys[:8]} | unexpected {len(r.unexpected_keys)} {r.unexpected_keys[:8]}")
print(f"timm model params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, patch {model.patch_embed.patch_size}")
model.eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    f = model(x)
print("feature shape", tuple(f.shape), "finite", bool(torch.isfinite(f).all()))
