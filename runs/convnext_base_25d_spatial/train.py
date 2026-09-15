"""
train.py — convnext_base_25d_spatial

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_spatial/train.py
Modal:  modal run runs/convnext_base_25d_spatial/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_25d_spatial"


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


class SpatialLabelAttention(nn.Module):
    """12 label queries cross-attending over every SPATIAL POSITION of every image.

    WHAT CHANGES FROM convnext_base_25d_slotattn. That run attended over 24 tokens - one per
    image, already flattened by timm's global average pool. This attends over 24 x 100 = 2400
    tokens, because the pool is never applied. ConvNeXt-Base at 336 px ends on a 10x10 grid
    and each cell covers about 13.4 mm of knee, so a query can now answer "lateral meniscus,
    posterior third of the sagittal slice" - a LOCATION - instead of only "sagittal slot 2".

    That matters because the global average pool is the first of two places the baseline
    destroys information: after it, a fracture top-left and the same fracture bottom-right
    give an identical vector. slotattn re-weighted what survived; this recovers what was lost.

    It is affordable for one reason: cross-attention with few queries is LINEAR in the key
    count. 12 queries x 2400 keys is ~29k dot products, against 24 ConvNeXt forward passes.
    Self-attention over the same tokens would be 2400 squared, which is not affordable.

    MASKING, which the baseline gets wrong. A missing or undecodable series becomes a BLANK
    slot, and every run so far averages it in as though it were data. Both heads in the 0.941
    notebook mask it (`masked_fill(mask < 0.5, -10000.)`). A blank slot is constant after
    normalisation, so its per-image variance is exactly zero - that is how the mask is found
    here, with no change to the dataset.

    THE GATE STARTS OPEN, and that is deliberate. slotattn and tokenattn were gated residuals
    initialised to zero: both stayed near zero (0.012 and 0.025), both correlated 0.99 with
    their parent, and both came back neutral. The safety net was also the ceiling - with a
    mean that already works there is little pressure to open the gate. Here the branch starts
    CONTRIBUTING (gate_init 1.0) with the mean still present as a residual, so the question
    becomes "does spatial attention help?" rather than "will the model bother to try it?".
    The gate can still fall to zero if attention hurts, and it is logged every 200 steps.
    """

    def __init__(self, feat: int, n_tasks: int, n_slots: int, n_groups: int,
                 heads: int = 8, use_pos: bool = True, gate_init: float = 1.0,
                 proj: int = 0):
        super().__init__()
        self.n_slots, self.n_groups = n_slots, n_groups
        d = int(proj) if proj else feat
        # optional projection before attending, as their SlotHead does (dim -> 256) and
        # their _RadHead does (2048 -> 512). 0 keeps the full width.
        self.proj = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, d), nn.GELU()) if proj else None
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, d))
        self.pos_emb = nn.Parameter(torch.zeros(n_groups, d)) if use_pos else None
        self.query = nn.Parameter(torch.randn(n_tasks, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out_norm = nn.LayerNorm(d)
        self.w = nn.Parameter(torch.randn(n_tasks, d) * d ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_tasks))
        self.gate = nn.Parameter(torch.full((n_tasks,), float(gate_init)))

    def forward(self, fmap, blank):
        """fmap: (B*slots*groups, F, H, W) straight from forward_features, pool NOT applied.
        blank: (B, slots*groups) True where that image is a blank slot."""
        n_img, f, h, w = fmap.shape
        per = self.n_slots * self.n_groups
        b = n_img // per
        # (B*24, F, 10, 10) -> (B, 24, 100, F): one token per position per image
        t = fmap.flatten(2).transpose(1, 2).reshape(b, per, h * w, f)
        if self.proj is not None:
            t = self.proj(t)
        d = t.shape[-1]
        t = t.view(b, self.n_slots, self.n_groups, h * w, d)
        t = t + self.slot_emb[None, :, None, None, :]
        if self.pos_emb is not None:
            t = t + self.pos_emb[None, None, :, None, :]
        t = t.reshape(b, per * h * w, d)                 # (B, 2400, d)
        t = self.norm(t)
        # every position of a blank image is masked, so the softmax never spends weight on it
        kpm = blank.repeat_interleave(h * w, dim=1)      # (B, 2400)
        kpm = torch.where(kpm.all(dim=1, keepdim=True), torch.zeros_like(kpm), kpm)
        q = self.query[None].expand(b, -1, -1)
        a, _ = self.attn(q, t, t, key_padding_mask=kpm, need_weights=False)
        a = self.out_norm(a)
        return self.gate[None] * ((a * self.w[None]).sum(-1) + self.b[None])


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
        feat = self.backbone.num_features
        # images per study: one per SLOT when slices are channels, otherwise one per slice
        self.per_study = len(i["slots"]) * (g if as_ch else i["n_slices"])
        self.pool = m.get("pool", "mean")            # how those images collapse into one vector
        if self.pool == "attn":
            # tiny network that scores each slice, so the model can weight informative ones
            self.pool_attn = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Linear(feat, sc.num_output_logits(cfg))   # one logit per label, 12

        # the slot/label attention branch, config-gated so this file still runs the plain
        # model when it is off. is_fresh marks it as randomly initialised, so shared_code
        # gives it the HEAD learning rate rather than the pretrained trunk's.
        sa = m.get("spatial_attention") or {}
        self.spatial_attn = None
        if sa.get("enable"):
            self.spatial_attn = SpatialLabelAttention(
                feat, sc.num_output_logits(cfg), len(i["slots"]),
                g if as_ch else int(i["n_slices"]),
                heads=int(sa.get("heads", 8)), use_pos=bool(sa.get("use_pos", True)),
                gate_init=float(sa.get("gate_init", 1.0)), proj=int(sa.get("proj", 0)))
            self.spatial_attn.is_fresh = True

    def forward(self, x):                      # x: (B*slots*D, C, H, W)
        if self.spatial_attn is not None:
            # forward_features keeps the 10x10 grid; the pool below is applied by hand so the
            # mean-pool path is unchanged while the attention branch sees every position.
            fmap = self.backbone.forward_features(x)          # (B*24, F, 10, 10)
            f = fmap.mean(dim=(2, 3))                         # (B*24, F) - the usual pool
            # A blank slot is all-zero pixels, but normalisation is PER CHANNEL, so it
            # comes out as a different constant in each of the 3 channels. Variance over all
            # elements therefore is NOT zero - it picks up the between-channel spread. Take
            # the variance over the SPATIAL dims only, per channel, and require every channel
            # to be flat. (Caught by the smoke test: the naive version flagged 0 blanks.)
            blank = (x.var(dim=(2, 3)).amax(dim=1) < 1e-8).view(-1, self.per_study)
            f = f.view(-1, self.per_study, f.shape[-1])
            out = self.head(f.mean(dim=1))
            return out + self.spatial_attn(fmap, blank)
        f = self.backbone(x)                   # (B*slots*D, F)
        f = f.view(-1, self.per_study, f.shape[-1])   # regroup the flat images back per study
        tokens = f                                    # keep the per-image features for attention
        if self.pool == "max":
            f = f.max(dim=1).values                   # strongest slice wins each feature
        elif self.pool == "attn":
            w = torch.softmax(self.pool_attn(f), dim=1)   # learned weights, summing to 1
            f = (f * w).sum(dim=1)
        else:
            f = f.mean(dim=1)                         # plain average over slices and slots
        return self.head(f)


def build_model(cfg: dict) -> nn.Module:
    return SliceModel(cfg)


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
