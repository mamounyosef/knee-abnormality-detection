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


# =============================================================================
# Triad: a 3D nnU-Net PlainConvEncoder pretrained by masked autoencoding on
# Triad-131K (131,170 3D MRI volumes; brain, breast, prostate; T1/T2/FLAIR/DWI/fMRI/DCE).
#
# WEIGHTS ARE MIT (the Triad authors). The HuggingFace bundle we read is a JAX/Equinox PORT
# whose PORT CODE is Apache-2.0 / GPL-3.0 dual licensed - so the encoder below is written
# from the published nnU-Net architecture rather than copied from that port, and only the
# weight tensors are used. Upstream (wangshansong1/Triad) ships the same weights for PyTorch
# under MIT; the port is simply easier to download than a Google Drive link.
#
# THE LAYOUT DIFFERENCE THAT MATTERS: Equinox stores a conv bias as (C, 1, 1, 1) for
# broadcasting, PyTorch wants (C,). Loading without reshaping raises, which is the good case;
# what would be silent is getting the STRIDE pattern wrong, since a wrong stride still
# produces a valid tensor of the wrong spatial size.
# =============================================================================
TRIAD_CHANNELS = (32, 64, 128, 256, 320, 320)
TRIAD_STRIDES = (1, 2, 2, 2, 2, 2)          # five halvings -> every spatial dim /32


class _TriadConv(torch.nn.Module):
    """nnU-Net's conv-norm-activation unit: 3x3x3, InstanceNorm3d(affine), LeakyReLU."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv = torch.nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=True)
        self.norm = torch.nn.InstanceNorm3d(cout, affine=True, eps=1e-5)
        self.act = torch.nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class TriadEncoder(torch.nn.Module):
    """The pretrained encoder only. Returns the bottleneck, globally pooled.

    The published checkpoint is encoder-only - the self-supervised reconstruction decoder is
    stripped - so the transfer representation IS the skip pyramid. We take the deepest stage
    and global-average it, which is the usual way to turn a segmentation encoder into a
    classification feature.
    """

    num_features = TRIAD_CHANNELS[-1]

    def __init__(self, in_ch: int = 1):
        super().__init__()
        stages, cin = [], in_ch
        for cout, st in zip(TRIAD_CHANNELS, TRIAD_STRIDES):
            stages.append(torch.nn.Sequential(_TriadConv(cin, cout, st),
                                              _TriadConv(cout, cout, 1)))
            cin = cout
        self.stages = torch.nn.ModuleList(stages)

    def forward(self, x):                     # (B, 1, D, H, W)
        for s in self.stages:
            x = s(x)
        return x.mean(dim=(2, 3, 4))          # (B, 320)

    def set_grad_checkpointing(self, enable=True):
        self._grad_ckpt = enable


def load_triad(model, ckpt_path, strict_count: bool = True):
    """Fill a TriadEncoder from the published safetensors bundle.

    Equinox conv biases are (C, 1, 1, 1); PyTorch wants (C,). Everything else maps by name
    once `encoder.` is stripped and `convs.N` becomes the Sequential index.
    """
    from safetensors.torch import load_file
    raw = load_file(str(ckpt_path))
    out, msd = {}, model.state_dict()
    for k, v in raw.items():
        kk = k[len("encoder."):] if k.startswith("encoder.") else k
        kk = kk.replace(".convs.", ".")               # stages.0.convs.1.conv -> stages.0.1.conv
        if kk not in msd:
            continue
        want = msd[kk]
        if v.shape != want.shape and v.numel() == want.numel():
            v = v.reshape(want.shape)                 # the (C,1,1,1) -> (C,) bias case
        if v.shape == want.shape:
            out[kk] = v
    missing = [k for k in msd if k not in out and "num_batches_tracked" not in k]
    model.load_state_dict(out, strict=False)
    if strict_count and missing:
        raise RuntimeError("Triad load filled only %d/%d tensors; unfilled: %s"
                           % (len(out), len(msd), ", ".join(missing[:8])))
    return len(out)
