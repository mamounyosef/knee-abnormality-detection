"""
train.py — convnext_base_25d_18slices

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_18slices/train.py
Modal:  modal run runs/convnext_base_25d_18slices/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_25d_mcl_top2_pooling"


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
        feat = self.backbone.num_features
        # images per study: one per SLOT when slices are channels, otherwise one per slice
        self.per_study = len(i["slots"]) * (g if as_ch else i["n_slices"])
        self.pool = m.get("pool", "mean")            # how those images collapse into one vector
        if self.pool == "attn":
            # tiny network that scores each slice, so the model can weight informative ones
            self.pool_attn = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Linear(feat, sc.num_output_logits(cfg))   # one logit per label, 12

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
        return self.head(f)



# --- top2 pooling, for MCL and nothing else ------------------------------------------
# The MCL is the Medial Collateral Ligament: a flat band on the INNER edge of the knee.
# Its whole length sits in very few of the 24 windows, so averaging all of them dilutes the
# evidence, while "average the two most suspicious" states it properly. Every other finding
# keeps the plain mean, exactly as the 0.9010 baseline has it.
TOP2_TASKS = ["MCL"]


class Top2Model(nn.Module):
    """The baseline model, with one column pooled differently."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        extra = dict(m.get("create_kwargs") or {})
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True),
                                          num_classes=0, drop_rate=m.get("drop_rate", 0.0),
                                          drop_path_rate=m.get("drop_path_rate", 0.0), **extra)
        if m.get("grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        feat = self.backbone.num_features
        self.per_study = sc.images_per_study(cfg)
        self.n_out = sc.num_output_logits(cfg)
        tasks = list(cfg["tasks"])
        self.top2_idx = [tasks.index(t) for t in TOP2_TASKS if t in tasks]
        if not self.top2_idx:
            raise ValueError("none of %s is in tasks" % TOP2_TASKS)
        self.mean_idx = [j for j in range(len(tasks)) if j not in self.top2_idx]
        # A top2 over 24 windows is systematically larger than their mean, so the two
        # pooled vectors reach the head on different scales. One norm each keeps the head's
        # effective learning rate the same for both.
        self.norm_mean = nn.LayerNorm(feat)
        self.norm_top2 = nn.LayerNorm(feat)
        self.head_mean = nn.Linear(feat, len(self.mean_idx))
        self.head_top2 = nn.Linear(feat, len(self.top2_idx))

    def forward(self, x):
        f = self.backbone(x)                               # (B*24, feat)
        f = f.view(-1, self.per_study, f.shape[-1])        # (B, 24, feat)
        out = f.new_zeros(f.shape[0], self.n_out)
        out[:, self.mean_idx] = self.head_mean(self.norm_mean(f.mean(dim=1))).to(out.dtype)
        v = f.topk(2, dim=1).values.mean(dim=1)            # (B, feat)
        out[:, self.top2_idx] = self.head_top2(self.norm_top2(v)).to(out.dtype)
        return out


def build_model(cfg: dict) -> nn.Module:
    return Top2Model(cfg)


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
