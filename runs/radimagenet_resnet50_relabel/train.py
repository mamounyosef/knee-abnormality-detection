"""
train.py — radimagenet_resnet50_relabel

RadImageNet ResNet-50 in place of the ImageNet ConvNeXt trunk: supervised
pretraining on 1.35M CT/MRI/ultrasound images across 11 anatomies, knee MRI
among them. Thin entry point: load config, build the model, hand off to the
engine.

Local:  python runs/radimagenet_resnet50_relabel/train.py
Modal:  modal run runs/radimagenet_resnet50_relabel/train.py
"""

import re
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "radimagenet_resnet50_relabel"


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


def _weights_path(cfg: dict) -> Path:
    """Where the 435 MB checkpoint lives.

    It is deliberately NOT in the container image (see the `pretrained` entry in
    modal_image's ignore list) — it rides on the runs volume instead, uploaded once
    with `modal volume put`. Locally it sits in data/pretrained/.
    """
    name = cfg["model"]["weights"]
    for c in (Path(cfg["modal"]["runs_mount"]) / "_pretrained" / name,
              Path(cfg["paths"]["data_dir"]) / "pretrained" / name):
        if c.exists():
            return c
    raise FileNotFoundError(
        f"{name} not found. Locally: data/pretrained/. On Modal, upload it once with\n"
        f"  modal volume put {cfg['modal']['runs_volume']} "
        f"data/pretrained/{name} /_pretrained/{name}")


def _load_radimagenet(backbone: nn.Module, path: Path):
    """RadImageNet's ResNet-50 -> a timm resnet50.

    Two fixes before timm recognises the keys:
      1. the weights sit under ck["model"], beside the optimizer/scheduler state
      2. every key carries a "_orig_mod." prefix, left behind by torch.compile
    The 165-way classifier is dropped - it predicted RadImageNet's own pathology
    labels, and we build the trunk with num_classes=0 anyway.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    pref = "_orig_mod."
    sd = {(k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}      # the 165-way head
    r = backbone.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:            # loud, because a silent partial load
        raise RuntimeError(                            # would quietly train a half-random trunk
            f"RadImageNet load mismatch - missing={r.missing_keys} "
            f"unexpected={r.unexpected_keys}")
    print(f"  ✅ RadImageNet loaded from {path}  ({len(sd)} tensors)")


class SliceModel(nn.Module):
    """2D backbone over every slice, pooled per study -> 12 logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        # num_classes=0 strips the classifier, leaving a feature vector per image
        # ResNet is fully convolutional, so it takes our 512px input unchanged even
        # though RadImageNet trained it at 224 - no position embedding to resample.
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", False),
                                          num_classes=0,
                                          drop_rate=m.get("drop_rate", 0.0))
        if m.get("weights"):
            _load_radimagenet(self.backbone, _weights_path(cfg))
        if m.get("grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        feat = self.backbone.num_features
        self.per_study = len(cfg["input"]["slots"]) * cfg["input"]["n_slices"]  # images per study
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
