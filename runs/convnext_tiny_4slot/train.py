"""
train.py — convnext_tiny_4slot

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_tiny_4slot/train.py
Modal:  modal run runs/convnext_tiny_4slot/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_tiny_4slot"


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402

cfg = sc.load_config(EXP_DIR, verbose=False)


class SliceModel(nn.Module):
    """2D backbone over every slice, pooled per study -> 12 logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True),
                                          num_classes=0, drop_rate=m.get("drop_rate", 0.0))
        feat = self.backbone.num_features
        self.per_study = len(cfg["input"]["slots"]) * cfg["input"]["n_slices"]
        self.pool = m.get("pool", "mean")
        if self.pool == "attn":
            self.pool_attn = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Linear(feat, sc.num_output_logits(cfg))

    def forward(self, x):                      # x: (B*slots*D, C, H, W)
        f = self.backbone(x)                   # (B*slots*D, F)
        f = f.view(-1, self.per_study, f.shape[-1])
        if self.pool == "max":
            f = f.max(dim=1).values
        elif self.pool == "attn":
            w = torch.softmax(self.pool_attn(f), dim=1)
            f = (f * w).sum(dim=1)
        else:
            f = f.mean(dim=1)
        return self.head(f)


def build_model(cfg: dict) -> nn.Module:
    return SliceModel(cfg)


def run_local():
    sc.set_seed(cfg["reproducibility"]["seed"], cfg["reproducibility"]["deterministic"])
    sc.print_config(cfg)
    sc.run_experiment(cfg, build_model(cfg), EXP_DIR)


try:
    import modal

    app = modal.App(cfg["experiment"]["name"])
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)
    _secrets = ([modal.Secret.from_name(cfg["modal"]["wandb_secret"])]
                if cfg.get("wandb", {}).get("enabled") else [])

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  secrets=_secrets, **sc.modal_resources(cfg))
    def train_remote():
        rcfg = sc.remote_cfg(cfg)
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
            _runs.commit()

    @app.local_entrypoint()
    def modal_main():
        train_remote.remote()

except ImportError:
    pass


if __name__ == "__main__":
    run_local()
