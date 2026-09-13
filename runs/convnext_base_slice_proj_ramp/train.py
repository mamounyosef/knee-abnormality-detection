"""
train.py — convnext_base_slice_proj_ramp

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_slice_proj_ramp/train.py
Modal:  modal run runs/convnext_base_slice_proj_ramp/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_NAME = "convnext_base_slice_proj_ramp"


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

        # LEARNED slice projection: a 1x1 conv from the 24 slice-channels down to `proj`
        # before the backbone. This is the third point on the stem-bottleneck curve:
        #   slice_input_ch1   24 -> stem       4 passes/study,  stem 3.0x compression
        #   slice_input_ch2   2 x 12 -> stem   8 passes/study,  stem 1.5x
        #   this run          24 -> 12 -> stem 4 passes/study,  stem 1.5x, learned mixing
        # It gets ch2's stem budget at ch1's cost, paying for it with a learned layer.
        # A RAMP down the slice axis: 24 -> 16 -> GELU -> 8, then the stem.
        #
        # Ending at 8 is the point. ConvNeXt-Base's stem maps a 4x4 patch of in_chans to
        # 128 outputs, so 8 channels is 8*16 = 128 -> 128, exactly 1.00x: no compression
        # anywhere in the chain. 12 compresses 1.5x, 24 compresses 3.0x, and the ordinary
        # 3-channel layout EXPANDS 48 -> 128.
        #
        # The GELU is not decoration. Two 1x1 convs with nothing between them collapse to a
        # single 1x1 conv, so a linear 24 -> 16 -> 8 ramp would be algebraically identical
        # to a plain 24 -> 8 map. The nonlinearity is what makes the ramp a ramp.
        #
        # Both convs start as a SMOOTH RESAMPLE, not randomly: output j is the mean of the
        # input channels that map onto it. timm's in_chans adaptation works precisely
        # because the widened stem begins as the pretrained filter applied to the MEAN of
        # its inputs (measured: tiling the RGB filters and scaling by 3/24 reproduces the
        # 3-channel output to 1.3e-05), and a random projection in front would destroy that
        # on step 1.
        #
        # KNOWN COST, measured and accepted: the GELU shifts the stem's input from
        # (mean +0.22, std 0.74) to (mean +0.46, std 0.51). The pretrained stem was not
        # trained on that. If this run underperforms the other slice_* runs, that shift is
        # the first thing to suspect.
        stages = m.get("slice_proj")
        stages = [stages] if isinstance(stages, int) and stages else (stages or [])
        self.slice_proj = None
        if stages:
            layers, prev = [], in_ch
            for k, nxt in enumerate(stages):
                conv = nn.Conv2d(prev, nxt, kernel_size=1, bias=False)
                with torch.no_grad():
                    conv.weight.zero_()
                    edges = torch.linspace(0, prev, nxt + 1)
                    for o in range(nxt):
                        lo = int(edges[o].item())
                        hi = max(int(edges[o + 1].item()), lo + 1)
                        conv.weight[o, lo:hi, 0, 0] = 1.0 / (hi - lo)
                conv.is_fresh = True                 # -> optimizer.fresh_lr
                layers.append(conv)
                if k < len(stages) - 1:
                    layers.append(nn.GELU())
                prev = nxt
            self.slice_proj = nn.Sequential(*layers)
            in_ch = prev
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

    def forward(self, x):                      # x: (B*slots*D, C, H, W)
        if self.slice_proj is not None:
            x = self.slice_proj(x)             # (B*slots, 24, H, W) -> (B*slots, 8, H, W)
        f = self.backbone(x)                   # (B*slots*D, F)
        f = f.view(-1, self.per_study, f.shape[-1])   # regroup the flat images back per study
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
