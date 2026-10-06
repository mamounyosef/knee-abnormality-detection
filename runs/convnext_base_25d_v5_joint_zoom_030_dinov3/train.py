"""
train.py — convnext_base_25d_v5_joint_zoom_030_dinov3

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_v5_joint_zoom_030_dinov3/train.py
Modal:  modal run runs/convnext_base_25d_v5_joint_zoom_030_dinov3/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_25d_v5_joint_zoom_030_dinov3"


def _pkg_root() -> Path:
    """Find the runs/ folder holding shared_code.py — /root/runs on Modal, ../ locally."""
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))          # so `import shared_code` works from either location
import shared_code as sc  # noqa: E402

cfg = sc.load_config(EXP_DIR, verbose=False)   # shared_config.yaml + this folder's config.yaml


class SliceModel(nn.Module):
    """2D backbone over every slice, pooled per study -> 12 logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        # num_classes=0 strips the ImageNet classifier, leaving a feature vector per image
        # slices_as_channels: one image per SLOT whose channels ARE its slices, so the
        # stem must accept n_slices inputs. timm adapts the pretrained RGB filters to the
        # wider input by tiling and rescaling them, so the weights are still used.
        i = cfg["input"]
        g = int(i.get("slices_as_channels") or 0)     # images per slot; 0 = off
        as_ch = g > 0
        in_ch = int(i["n_slices"]) // g if as_ch else int(i["channels"])
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True),
                                          num_classes=0, in_chans=in_ch,
                                          drop_rate=m.get("drop_rate", 0.0))
        if m.get("grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)    # local 8 GB card: recompute activations instead of storing
        feat = self.backbone.num_features
        # images per study: one per SLOT when slices are channels, otherwise one per slice
        self.per_study = len(i["slots"]) * (g if as_ch else i["n_slices"])
        self.pool = m.get("pool", "mean")            # how those images collapse into one vector
        if self.pool == "attn":
            # tiny network that scores each slice, so the model can weight informative ones
            self.pool_attn = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Linear(feat, sc.num_output_logits(cfg))   # one logit per label, 12
        # DINOv3's own final LayerNorm gain is ~2x ImageNet's, so a fresh norm before the head (as dinov3_convnext_base_v5)
        self.feat_norm = nn.LayerNorm(feat) if m.get("feat_norm") else nn.Identity()

    def forward(self, x):                      # x: (B*slots*D, C, H, W)
        f = self.backbone(x)                   # (B*slots*D, F)
        f = f.view(-1, self.per_study, f.shape[-1])   # regroup the flat images back per study
        if self.pool == "max":
            f = f.max(dim=1).values                   # strongest slice wins each feature
        elif self.pool == "attn":
            w = torch.softmax(self.pool_attn(f), dim=1)   # learned weights, summing to 1
            f = (f * w).sum(dim=1)
        else:
            f = f.mean(dim=1)                         # plain average over slices and slots
        return self.head(self.feat_norm(f))


PARENT_CKPT = "convnext_base_25d_18slices_v5_aug_base/results/checkpoints/top_step5900.pt"   # the LOCALIZER (frozen)
TRUNK_CKPT = "dinov3_convnext_base_v5/results/checkpoints/top_step5800.pt"   # trunk + feat_norm + head (LB member)


class JointZoomModel(SliceModel):
    """Two-stage in ONE module: a frozen copy of the 2.5D parent finds the knee joint, the trainable ConvNeXt reads it zoomed.

    The loader delivers each slot at 0.20 mm/px, 672 px (the parent's 134 mm frame at twice the detail). The localizer
    (self.loc_*) sees it area-downsampled to 336 px, exactly its training view, and its class-activation map is taken
    in ANATOMY mode: |contribution of each feature cell to every one of the 12 logits|, summed (exact decomposition of
    timm ConvNeXt's pool -> LayerNorm -> linear head, no gradients). The centroid of that map's above-median part, over
    the slot's images, is the joint centre (runs/others/crop_rois/cam_preview.py, checked by eye on normal and abnormal
    knees). The trunk then reads a crop x crop window (384 px = 77 mm at 0.20 mm/px) around it, the same window for
    every image of the slot. Training jitters the centre by up to jitter_px so the net does not overfit one placement.
    self.backbone / self.head stay the trainable trunk and head, so the engine's lr groups work unchanged; the
    localizer is frozen (requires_grad False, always eval) and saved inside the checkpoint, so inference is self-contained.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        z = cfg["model"]["zoom"]
        self.crop, self.loc_size, self.jitter = int(z["crop"]), int(z["loc_size"]), int(z.get("jitter_px", 0))
        i = cfg["input"]
        self.g = int(i["slices_as_channels"])
        self.n_slots = len(i["slots"])
        self.loc_backbone = timm.create_model(z["loc_model"], pretrained=False, num_classes=0, in_chans=3)
        self.loc_head = nn.Linear(self.loc_backbone.num_features, 12)
        for p in list(self.loc_backbone.parameters()) + list(self.loc_head.parameters()):
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.loc_backbone.eval()                  # frozen localizer: no dropout / stat drift, ever
        return self

    @torch.no_grad()
    def centres(self, x):
        """x (N, 3, H, H) normalised hi-res -> (N // g, 2) crop centres (cy, cx) in hi-res px, one per study-slot."""
        H = x.shape[-1]
        lo = torch.nn.functional.interpolate(x, size=(self.loc_size, self.loc_size), mode="area")
        f = self.loc_backbone.forward_features(lo).float()            # (N, F, h, w)
        pooled = f.mean((2, 3))
        ln = self.loc_backbone.head.norm
        mu = pooled.mean(1, keepdim=True)
        sig = torch.sqrt(pooled.var(1, unbiased=False, keepdim=True) + ln.eps)
        fn = ((f - mu[:, :, None, None]) / sig[:, :, None, None] * ln.weight.float()[None, :, None, None]
              + ln.bias.float()[None, :, None, None])
        cam = torch.einsum("nfhw,kf->nkhw", fn, self.loc_head.weight.float()).abs().sum(1)   # (N, h, w)
        h, w = cam.shape[-2:]
        cam = cam.view(-1, self.g, h, w).sum(1)                      # per study-slot
        cam = torch.clamp(cam - cam.flatten(1).median(1).values[:, None, None], min=0)
        tot = cam.flatten(1).sum(1).clamp_min(1e-6)
        ys = torch.arange(h, device=x.device, dtype=torch.float32)
        xs = torch.arange(w, device=x.device, dtype=torch.float32)
        cy = (cam.sum(2) * ys).sum(1) / tot
        cx = (cam.sum(1) * xs).sum(1) / tot
        flat = cam.flatten(1).sum(1) <= 0                            # no signal at all: frame centre
        cell = 32.0 * H / self.loc_size                              # ConvNeXt stride 32: 336 px -> 10 cells of 32 px
        c = torch.stack([(cy + 0.5) * cell, (cx + 0.5) * cell], 1)
        c[flat] = H / 2
        return c

    def forward(self, x):                      # x: (B*slots*g, 3, 672, 672)
        H, s = x.shape[-1], self.crop
        c = self.centres(x)
        if self.training and self.jitter:
            c = c + torch.empty_like(c).uniform_(-self.jitter, self.jitter)
        o = torch.clamp(torch.round(c - s / 2), 0, H - s).long().cpu().tolist()
        xv = x.view(-1, self.g, *x.shape[1:])
        crops = torch.stack([xv[k, :, :, y0:y0 + s, x0:x0 + s] for k, (y0, x0) in enumerate(o)])
        return super().forward(crops.view(-1, *crops.shape[2:]))


def _ckpt(rel, cfg):
    for c in (Path(cfg["modal"]["runs_mount"]) / rel, PKG_ROOT.parent / "runs" / rel):
        if c.exists():
            sd = dict(torch.load(c, map_location="cpu", weights_only=False)["model"])
            return c, {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    return None, None


def build_model(cfg: dict) -> nn.Module:
    """Trunk, feat_norm and head from the DINOv3 ConvNeXt member (it read single slices at 512 px; here it reads the
    2.5D zoom crops, 3 slices per image); the frozen localizer from the 2.5D parent, exactly as the 0.30 zoom twin."""
    model = JointZoomModel(cfg)
    lsrc, lsd = _ckpt(PARENT_CKPT, cfg)
    tsrc, tsd = _ckpt(TRUNK_CKPT, cfg)
    if lsrc is None or tsrc is None:
        # only training needs the seeds; at inference (Kaggle) the checkpoint carries trunk, head AND localizer
        print("  \U0001F50D seed checkpoint(s) missing - skipping the seed; fine for inference, WRONG for training", flush=True)
        return model
    assert tsd["head.weight"].shape[0] == 12 and lsd["head.weight"].shape[0] == 12 and len(cfg["tasks"]) == 12
    full = dict(tsd)                                              # backbone.*, feat_norm.*, head.*
    full.update({"loc_" + k: v for k, v in lsd.items() if k.split(".")[0] in ("backbone", "head")})
    r = model.load_state_dict(full, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(f"seed load mismatch - missing={r.missing_keys} unexpected={r.unexpected_keys}")
    print(f"  \U0001F50D trunk/head from {tsrc.parent.parent.parent.name}/{tsrc.name}, frozen localizer from "
          f"{lsrc.parent.parent.parent.name}/{lsrc.name}", flush=True)
    return model


def run_local():
    """Train on this PC's GPU — only useful for smoke tests, real runs go to Modal."""
    sc.set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
    sc.print_config(cfg)
    sc.run_experiment(cfg, build_model(cfg), EXP_DIR)


try:
    # everything below is skipped when modal isn't installed, so local runs still work
    import modal

    app = modal.App(cfg["experiment"]["name"])
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)  # input data
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)      # outputs
    _secrets = ([modal.Secret.from_name(cfg["modal"]["wandb_secret"])]
                if cfg.get("wandb", {}).get("enabled") else [])

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  secrets=_secrets, **sc.modal_resources(cfg))
    def train_remote(run_id: str = ""):
        """Runs inside the Modal container on the GPU named in config.yaml."""
        rcfg = sc.remote_cfg(cfg)                  # same config, container paths
        if run_id:
            rcfg["experiment"]["run_id"] = run_id  # allocated on the launching PC
        out_dir = Path(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"]
        sc.set_seed(rcfg["reproducibility"]["seed"], rcfg["reproducibility"]["deterministic"])
        sc.print_config(rcfg)
        model = build_model(rcfg)

        wb = cfg.get("wandb", {})
        run, log_fn = None, None
        if wb.get("enabled"):
            import wandb
            run = wandb.init(project=wb.get("project"), name=cfg["experiment"]["name"],
                             config=cfg, settings=wandb.Settings(x_disable_stats=True))
            log_fn = lambda d, s: run.log(d, step=s)
        try:
            sc.run_experiment(rcfg, model, out_dir, persist_fn=_runs.commit, log_fn=log_fn)
        finally:
            if run:
                run.finish()
            _runs.commit()                         # always save results, even if training crashed

    @app.local_entrypoint()                        # what `modal run train.py` calls
    def modal_main():
        sc.sync_on_interrupt(cfg)                  # ctrl+c still pulls the artifacts
        run_id = sc.launch_run_id(cfg, EXP_DIR)    # sequential, reserved here
        status = "crashed"
        try:
            train_remote.remote(run_id)
            status = "finished"
        finally:
            # the container can't reach this PC, so the pull happens here — and it
            # has to happen whether training finished or blew up
            sc.sync_back(cfg, status)

except ImportError:
    pass


if __name__ == "__main__":
    run_local()
