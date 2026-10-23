"""
train.py — convnext_base_v5_96slices_per_finding_attn

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_v5_96slices_per_finding_attn/train.py
Modal:  modal run runs/convnext_base_v5_96slices_per_finding_attn/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_v5_96slices_per_finding_attn"


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


class PerFindingAttnModel(nn.Module):
    """2D backbone over every 3-slice window, per-finding attention over the window bag -> 12 logits.

    The public Raptor/0.949 head: LayerNorm -> Linear(F, 256) -> tanh -> dropout -> Linear(256, 12) scores every
    window once per finding, softmax over the windows, one pooled vector per finding, and a per-finding linear
    read-out W (12, F). Windows per study are 12 in training and 94 at eval (input.window_bag): self.k follows
    train() / eval()."""

    def __init__(self, cfg: dict):
        super().__init__()
        m, wb = cfg["model"], cfg["input"]["window_bag"]
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True), num_classes=0,
                                          in_chans=3, drop_rate=m.get("drop_rate", 0.0))
        F_ = self.backbone.num_features
        n = sc.num_output_logits(cfg)
        self.k_train, self.k_eval = int(wb["train_k"]), int(wb["eval_k"])
        self.k = self.k_train
        self.norm = nn.LayerNorm(F_)
        self.att = nn.Sequential(nn.Linear(F_, 256), nn.Tanh(), nn.Dropout(0.2), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def train(self, mode: bool = True):
        self.k = self.k_train if mode else self.k_eval
        return super().train(mode)

    def forward(self, x):                      # x: (B*k, 3, H, W)
        f = self.backbone(x)                   # (B*k, F)
        h = self.norm(f.view(-1, self.k, f.shape[-1]))        # (B, k, F)
        a = torch.softmax(self.att(h), dim=1)                 # (B, k, 12): weights over the windows, per finding
        pooled = torch.einsum("bkn,bkf->bnf", a, h)           # (B, 12, F)
        return (pooled * self.clsW).sum(-1) + self.clsb       # (B, 12)


def build_model(cfg: dict) -> nn.Module:
    return PerFindingAttnModel(cfg)


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
