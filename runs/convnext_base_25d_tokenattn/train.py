"""
train.py — convnext_base_25d_tokenattn

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_tokenattn/train.py
Modal:  modal run runs/convnext_base_25d_tokenattn/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_25d_tokenattn"


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


class TokenSelfAttention(nn.Module):
    """Let the 24 image tokens REWRITE EACH OTHER before pooling.

    This is the other half of the public 0.941 notebook's V48 head, and the half
    convnext_base_25d_slotattn does NOT have. The distinction is exact:

        slotattn   12 label queries attend OVER the tokens. Each token reaches the output
                   unchanged and is merely WEIGHTED.  Q=queries, K=V=tokens.
        this run   the tokens attend over EACH OTHER. The sagittal token comes out
                   containing coronal and axial information.  Q=K=V=tokens.

    Shapes: (B, 24, 1024) in, (B, 24, 1024) out - same shape, different contents. Pooling
    afterwards is the ordinary mean, so the ONLY change against convnext_base_25d_18slices
    is that the tokens have talked before being averaged.

    Slot and position embeddings are added first, for the same reason as in slotattn: with a
    plain mean the model cannot tell a sagittal token from a coronal one, and self-attention
    is useless if the tokens carry no identity to attend on.

    GATED RESIDUAL, gate at zero: tokens_out = tokens + gate * attn(tokens). At step 0 this
    is the identity and the model is bit-identical to convnext_base_25d_18slices
    (val 0.9010 / LB 0.914).

    NOT a repeat of convnext_small_transformer_pool (0.8792). That REPLACED the mean with
    self-attention over 96 raw slices, on ConvNeXt-Small, on the old labels, with no slot or
    position embeddings and no residual. Four differences, and the one that matters most is
    that it could not fall back to the mean if attention failed to learn.
    """

    def __init__(self, feat: int, n_slots: int, n_groups: int, heads: int = 8,
                 use_pos: bool = True):
        super().__init__()
        self.n_slots, self.n_groups = n_slots, n_groups
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, feat))
        self.pos_emb = nn.Parameter(torch.zeros(n_groups, feat)) if use_pos else None
        self.norm = nn.LayerNorm(feat)
        self.attn = nn.MultiheadAttention(feat, heads, batch_first=True)
        # gate at zero -> identity at init. Safe here without a random-init trick because
        # MultiheadAttention's own out_proj is randomly initialised, so attn(tokens) is
        # non-zero and the gate therefore receives a non-zero gradient immediately.
        # scalar here, unlike slotattn's per-label gate: this branch rewrites the TOKENS,
        # which are shared by all 12 findings, so there is no per-label axis to gate on.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, f):                      # f: (B, per_study, feat)
        b, n, d = f.shape
        t = f
        if n == self.n_slots * self.n_groups:
            t = t.view(b, self.n_slots, self.n_groups, d)
            t = t + self.slot_emb[None, :, None, :]
            if self.pos_emb is not None:
                t = t + self.pos_emb[None, None, :, :]
            t = t.reshape(b, n, d)
        t = self.norm(t)
        a, _ = self.attn(t, t, t, need_weights=False)   # Q = K = V = the tokens
        return f + self.gate * a                       # (B, per_study, feat)


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
        sa = m.get("token_attention") or {}
        self.token_attn = None
        if sa.get("enable"):
            self.token_attn = TokenSelfAttention(
                feat, len(i["slots"]), g if as_ch else int(i["n_slices"]),
                heads=int(sa.get("heads", 8)), use_pos=bool(sa.get("use_pos", True)))
            self.token_attn.is_fresh = True

    def forward(self, x):                      # x: (B*slots*D, C, H, W)
        f = self.backbone(x)                   # (B*slots*D, F)
        f = f.view(-1, self.per_study, f.shape[-1])   # regroup the flat images back per study
        # the tokens rewrite each other BEFORE pooling, which is the whole point: pooling
        # then averages vectors that have already exchanged information
        if self.token_attn is not None:
            f = self.token_attn(f)
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
