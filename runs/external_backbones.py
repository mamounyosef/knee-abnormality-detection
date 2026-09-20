r"""external_backbones.py - load pretrained weights that do not ship inside timm.

Two of the backbones we want are published as raw checkpoints rather than timm entries:

  MedSAM ViT-B    Apache 2.0, Zenodo DOI 10.5281/zenodo.10689643. A medical fine-tune of
                  Meta SAM, so its encoder is architecturally identical to timm's
                  `samvit_base_patch16` and its weights can be mapped onto it.
  Barlow Twins    MIT, facebookresearch/barlowtwins. A self-supervised ResNet-50, which maps
                  onto timm's `resnet50`.

WHY THIS FILE EXISTS RATHER THAN A load_state_dict CALL. `strict=False` is silent: if the
key names do not line up, the backbone stays at RANDOM INIT and the run trains from scratch
while every log line looks normal. That failure is invisible in the loss curve - a randomly
initialised ConvNeXt still learns - and we would credit or blame the pretrained weights for
a result they had no part in. So every loader here COUNTS the tensors it actually filled and
raises if the count is short.

THE RESOLUTION PROBLEM, which is specific to SAM. SAM was pretrained at 1024x1024, so its
absolute position embedding is a 64x64 grid and its relative-position tables are 127 entries
(2*64-1). Running the encoder at 336 or 512 needs 21x21 / 32x32 grids and 41 / 63 entries.
Those nine tensors must be RESAMPLED, not dropped: dropping them leaves the position
information randomly initialised while the other 168 tensors load correctly, which is the
worst case - a model that is 95% pretrained and silently broken where it matters most.
"""

from pathlib import Path

import torch
import torch.nn.functional as F


def _unwrap(ckpt):
    """Checkpoints nest the real state dict under various keys."""
    for k in ("model", "state_dict", "module"):
        if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return ckpt


def _resample_pos_embed(v, want_hw):
    """SAM's absolute position embedding: (1, H, W, C) -> (1, h, w, C), bicubic."""
    if v.shape[1] == want_hw[0] and v.shape[2] == want_hw[1]:
        return v
    x = v.permute(0, 3, 1, 2).float()                    # (1, C, H, W)
    x = F.interpolate(x, size=want_hw, mode="bicubic", align_corners=False)
    return x.permute(0, 2, 3, 1).to(v.dtype)


def _resample_rel_pos(v, want_len):
    """SAM's relative-position table: (2L-1, C) -> (2l-1, C), linear.

    This is the same interpolation SAM applies internally when a window size differs from
    the table, so resizing the stored table is consistent with how the model uses it.
    """
    if v.shape[0] == want_len:
        return v
    x = v.permute(1, 0).unsqueeze(0).float()             # (1, C, 2L-1)
    x = F.interpolate(x, size=want_len, mode="linear", align_corners=False)
    return x.squeeze(0).permute(1, 0).to(v.dtype)


def load_medsam(model, ckpt_path, strict_count: bool = True):
    """Fill a timm `samvit_base_patch16` from a MedSAM (or Meta SAM) checkpoint.

    Returns the number of tensors filled. Raises if any of the model's parameters were left
    untouched, because a partially loaded encoder is worse than an obviously broken one.
    """
    raw = _unwrap(torch.load(str(ckpt_path), map_location="cpu", weights_only=False))
    # the segment_anything layout prefixes everything with image_encoder.
    sd = {k[len("image_encoder."):]: v for k, v in raw.items()
          if k.startswith("image_encoder.")}
    if not sd:
        sd = dict(raw)
    # timm's own converter handles the block/mlp naming differences
    try:
        from timm.models.vision_transformer_sam import checkpoint_filter_fn
        conv = checkpoint_filter_fn(dict(raw), model)
        if len(conv) >= len(sd):
            sd = conv
    except Exception:
        pass

    msd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k not in msd:
            continue
        want = msd[k]
        if v.shape != want.shape:
            if k == "pos_embed" and v.ndim == 4 and want.ndim == 4:
                v = _resample_pos_embed(v, (want.shape[1], want.shape[2]))
            elif "rel_pos" in k and v.ndim == 2 and want.ndim == 2:
                v = _resample_rel_pos(v, want.shape[0])
            else:
                continue
        if v.shape == want.shape:
            out[k] = v

    missing = [k for k in msd if k not in out]
    model.load_state_dict(out, strict=False)
    if strict_count and missing:
        raise RuntimeError(
            "MedSAM load filled only %d/%d tensors; unfilled: %s"
            % (len(out), len(msd), ", ".join(missing[:8])))
    return len(out)


def load_barlowtwins(model, ckpt_path, strict_count: bool = True):
    """Fill a timm `resnet50` (num_classes=0) from the Barlow Twins checkpoint.

    The released file is the backbone only, with no fc layer, so the classifier is expected
    to be absent; everything else must match.
    """
    raw = _unwrap(torch.load(str(ckpt_path), map_location="cpu", weights_only=False))
    sd = {}
    for k, v in raw.items():
        k = k.replace("module.", "").replace("backbone.", "")
        sd[k] = v
    msd = model.state_dict()
    out = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    # a num_classes=0 timm resnet has no fc weights, so nothing else may be missing
    missing = [k for k in msd if k not in out and not k.startswith("fc.")]
    model.load_state_dict(out, strict=False)
    if strict_count and missing:
        raise RuntimeError(
            "Barlow Twins load filled only %d/%d tensors; unfilled: %s"
            % (len(out), len(msd), ", ".join(missing[:8])))
    return len(out)


def resolve_weights(filename, cfg=None):
    """Find a checkpoint locally or on the Modal runs volume."""
    cands = []
    if cfg:
        cands.append(Path(cfg["modal"]["runs_mount"]) / "pretrained" / filename)
    cands += [Path("/runs/pretrained") / filename,
              Path(__file__).resolve().parent.parent / "data" / "pretrained" / filename]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("%s not found in %s" % (filename, [str(c) for c in cands]))
