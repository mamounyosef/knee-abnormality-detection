"""
train.py — orthofoundation_vitl_cross_attention_v5

OrthoFoundation-L (ytrsk/OrthoFoundation, MIT; DINOv3-L continued DINO pretraining on ~894k knee MRI slices +
~358k knee radiographs, so the DINOv3 licence applies too) in place of MRI-CORE ViT-B, inside the same
cross-slice-attention model. Each slice copied to 3 channels, as in its pretraining. Thin entry point.

Local:  python runs/orthofoundation_vitl_cross_attention_v5/train.py
Modal:  modal run runs/orthofoundation_vitl_cross_attention_v5/train.py
"""

import re
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "orthofoundation_vitl_cross_attention_v5"


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
    """Where the 1.2 GB checkpoint lives.

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


def _load_orthofoundation(backbone: nn.Module, path: Path):
    """OrthoFoundation-L.pth -> timm's DINOv3 ViT-L (Eva).

    The file is a flat state dict of the DINOv3 torch.hub ViT under a "backbone." prefix (the DINO head was not
    saved). timm's own eva.checkpoint_filter_fn converts DINOv3 hub keys; position is RoPE, so no table to resample
    for 336 px. Measured: 0 missing, 0 unexpected keys (runs/others/orthofoundation_load_check.py).
    """
    from timm.models import eva
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    r = backbone.load_state_dict(eva.checkpoint_filter_fn(sd, backbone), strict=False)
    if r.missing_keys or r.unexpected_keys:            # loud, because a silent partial load
        raise RuntimeError(                            # would quietly train a half-random trunk
            f"OrthoFoundation load mismatch — missing={r.missing_keys} unexpected={r.unexpected_keys}")
    print(f"  ✅ OrthoFoundation-L loaded from {path}  ({len(sd)} tensors)")


class CrossSliceBlock(nn.Module):
    """Attention across the S slices of a study, at one token position.

    B = studies per batch,  S = 96 slices per study (4 slots x 24),
    T = 1025 tokens per slice (32x32 patches + 1 CLS),  F = 768 (= dim = embed_dim).

    The 12 pretrained blocks attend over T WITHIN one slice; slices never see each other
    until the mean at the very end. This adds the missing axis: at each token position,
    attend over the S slices.

    UNGATED: out = x + h, a plain residual branch at full strength from step 1.

    There used to be a per-channel `gamma` scaling h, initialised near 0 so the block began
    as the identity. That was a trap. The gradient reaching the branch's OWN weights is
    scaled by gamma, so at 1e-3 the attention trained ~1000x too slowly, while gamma's own
    gradient was measured on a branch that was still random noise - the gate closed, which
    kept the branch untrained, which kept the gate closed. mricore_vitb_cross_attention_v5_ft
    rode that loop the whole way down: |gamma| 9.1e-4 -> 4.0e-4 over 600 steps, channels
    open 296 -> 52, val pinned at 0.8936 against a 0.8970 init. The gate is gone entirely.

    Attention is quadratic in sequence length and S=96 is far shorter than T=1025, so each
    of these costs ~1/11th of the spatial attention beside it.
    """

    def __init__(self, dim: int, num_heads: int, per_study: int, attn_drop: float = 0.0):
        super().__init__()
        self.S = per_study                            # S = 96
        self.norm = nn.LayerNorm(dim)                 # LayerNorm(F=768), normalizes over the last dim
        self.attn = nn.MultiheadAttention(dim, num_heads,      # F=768, 12 heads -> head_size is 64
                                          dropout=attn_drop, batch_first=True)
        self.is_fresh = True                          # -> optimizer.fresh_lr, not the trunk rate

    def forward(self, x, **_):                  # DINOv3 (Eva) passes rope= / attn_mask= to every block; unused here
        # x: (B*S, T, F) = (B*72, 446, 1024)
        BS, T, F = x.shape                            # BS = B*96, T = 1025, F = 768
        B = BS // self.S                              # studies in this batch
        h = self.norm(x)                              # (B*S, T, F) -> (B*S, T, F), over dim=2 (F)
        h = h.view(B, self.S, T, F)                   # (B*S, T, F) -> (B, S, T, F)
        h = h.permute(0, 2, 1, 3)                     # (B, S, T, F) -> (B, T, S, F)
        h = h.reshape(B * T, self.S, F)               # (B, T, S, F) -> (B*T, S, F), S is the sequence
        h, _ = self.attn(h, h, h, need_weights=False)  # (B*T, S, F) -> (B*T, S, F), attends over dim=1 (S)
        h = h.view(B, T, self.S, F)                   # (B*T, S, F) -> (B, T, S, F)
        h = h.permute(0, 2, 1, 3)                     # (B, T, S, F) -> (B, S, T, F)
        h = h.reshape(BS, T, F)                       # (B, S, T, F) -> (B*S, T, F)
        return x + h                                  # (B*S, T, F) + (B*S, T, F), plain residual


def _insert_cross_slice(backbone, per_study: int, after: list, attn_drop: float = 0.0):
    """Splice CrossSliceBlocks into the ViT's block list.

    timm keeps the 12 transformer blocks in an nn.Sequential where every element maps
    (B*S, T, F) -> (B*S, T, F), so a module with the same signature can simply be inserted.
    The pretrained blocks are not touched, only re-indexed.
    """
    dim = backbone.embed_dim
    heads = backbone.blocks[0].attn.num_heads
    out, added = [], 0
    for i, blk in enumerate(backbone.blocks, start=1):        # 1-based: "after block 3"
        out.append(blk)
        if i in after:
            out.append(CrossSliceBlock(dim, heads, per_study, attn_drop))
            added += 1
    backbone.blocks = nn.Sequential(*out)
    print(f"  \U0001F9E9 cross-slice attention: {added} block(s) after {after} "
          f"-> {len(out)} total, S={per_study}, dim={dim}, heads={heads}, ungated")
    return backbone


class SliceModel(nn.Module):
    """2D backbone over every slice, pooled per study -> 12 logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        size = cfg["input"]["size"]
        # num_classes=0 strips the classifier, leaving a feature vector per image
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", False),
                                          num_classes=0, img_size=size,
                                          **({"global_pool": m["global_pool"]} if m.get("global_pool") else {}),
                                          drop_rate=m.get("drop_rate", 0.0),
                                          drop_path_rate=m.get("drop_path_rate", 0.0))
        if m.get("weights"):
            _load_orthofoundation(self.backbone, _weights_path(cfg))
        if m.get("grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        self.per_study = len(cfg["input"]["slots"]) * cfg["input"]["n_slices"]   # S = 4*24 = 96
        xs = m.get("cross_slice") or {}
        if xs.get("enable"):
            # AFTER loading OrthoFoundation: the new blocks are ours, not in the checkpoint
            _insert_cross_slice(self.backbone, self.per_study,
                                list(xs.get("after", [3, 6, 9])),
                                float(xs.get("attn_drop", 0.0)))

        feat = self.backbone.num_features                 # F = 768
        self.feat_norm = nn.LayerNorm(feat) if m.get("feat_norm") else nn.Identity()
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
            f = f.mean(dim=1)                         # (B, S, F) -> (B, F), mean over dim=1
        return self.head(self.feat_norm(f))            # (B, F) -> (B, L)


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
