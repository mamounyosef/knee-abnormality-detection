"""
train.py — mricore_vitb

MRI-CORE ViT-B/16, self-supervised (DINO) on 6M MRI slices, in place of the
ImageNet ConvNeXt trunk. Thin entry point: load config, build the model, hand
off to the engine.

Local:  python runs/mricore_vitb/train.py
Modal:  modal run runs/mricore_vitb/train.py
"""

import re
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "mricore_vitb"


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
    # BOTH conventions. These runs were written against <runs_mount>/_pretrained/, while
    # external_backbones.resolve_weights (and therefore every workspace seeded since) uses
    # <runs_mount>/pretrained/ without the underscore. Looking in both costs a stat call
    # and means neither convention can strand a run on a volume that has the file.
    for c in (Path(cfg["modal"]["runs_mount"]) / "pretrained" / name,
              Path(cfg["modal"]["runs_mount"]) / "_pretrained" / name,
              Path("/runs/pretrained") / name,
              Path(cfg["paths"]["data_dir"]) / "pretrained" / name):
        if c.exists():
            return c
    raise FileNotFoundError(
        f"{name} not found. Locally: data/pretrained/. On Modal, upload it once with\n"
        f"  modal volume put {cfg['modal']['runs_volume']} "
        f"data/pretrained/{name} /pretrained/{name}")


def _load_mri_core(backbone: nn.Module, path: Path, grid: int):
    """MRI-CORE's DINO checkpoint -> a timm VisionTransformer.

    The file holds {"teacher": {...}} from the DINOv2 training codebase, so the keys
    need three fixes before timm recognises them:
      1. strip the "backbone." prefix (the rest of the teacher is the DINO head, which
         we throw away — it predicted cluster assignments, not findings)
      2. blocks.<chunk>.<i>.* -> blocks.<i>.*  (DINOv2 stores blocks in 4 chunks; the
         second index is already globally numbered 0..11, so it is the one to keep)
      3. resample pos_embed from the 14x14 grid it was trained at (224px) to ours
    """
    from timm.layers import resample_abs_pos_embed

    sd = torch.load(path, map_location="cpu", weights_only=False)["teacher"]
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue                                   # dino_head.* — discarded
        k = k[len("backbone."):]
        m = re.match(r"blocks\.\d+\.(\d+)\.(.*)", k)
        if m:
            k = f"blocks.{m.group(1)}.{m.group(2)}"
        out[k] = v
    out.pop("mask_token", None)                        # SSL-only, no counterpart in timm
    out["pos_embed"] = resample_abs_pos_embed(out["pos_embed"], new_size=[grid, grid],
                                              num_prefix_tokens=1)
    r = backbone.load_state_dict(out, strict=False)
    if r.missing_keys or r.unexpected_keys:            # loud, because a silent partial load
        raise RuntimeError(                            # would quietly train a half-random trunk
            f"MRI-CORE load mismatch — missing={r.missing_keys} "
            f"unexpected={r.unexpected_keys}")
    print(f"  ✅ MRI-CORE loaded from {path}  ({len(out)} tensors, pos_embed -> {grid}x{grid})")


class SliceModel(nn.Module):
    """2D backbone over every slice, pooled per study -> 12 logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        size = cfg["input"]["size"]
        # num_classes=0 strips the classifier, leaving a feature vector per image
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", False),
                                          num_classes=0, img_size=size,
                                          drop_rate=m.get("drop_rate", 0.0),
                                          drop_path_rate=m.get("drop_path_rate", 0.0))
        if m.get("weights"):
            _load_mri_core(self.backbone, _weights_path(cfg),
                           size // self.backbone.patch_embed.patch_size[0])
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
