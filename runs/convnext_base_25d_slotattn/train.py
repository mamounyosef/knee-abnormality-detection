"""
train.py — convnext_base_25d_slotattn

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_slotattn/train.py
Modal:  modal run runs/convnext_base_25d_slotattn/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_25d_slotattn"


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


class SlotLabelAttention(nn.Module):
    """Per-finding cross-attention over the per-image features, with slot identity.

    Mean pooling throws away two things the four slots genuinely differ on:

      WHICH SERIES a feature came from. Sagittal fluid-sensitive shows effusion and marrow
      oedema; coronal T1 shows anatomy. After f.mean(dim=1) the model cannot tell them apart,
      so it must learn one representation that works for all four at once.

      WHERE IN THE STACK it sat. Group 0 is the edge of the joint, group 3 the middle.

    And it forces all 12 findings to share ONE pooled vector, when we have measured focal and
    diffuse findings behaving in opposite directions under slice grouping (focal -0.0014,
    diffuse -0.0047 for 25d_group3 against 336px).

    So: add a learned slot embedding and a learned position embedding to each token, then let
    each of the 12 findings attend over the tokens with its OWN query. This is the public
    0.941 notebook's `SlotHead` (slot embeddings + one query per label + cross-attention) and
    the attention half of its V48 head (plane and position embeddings + MultiheadAttention).

    GATED RESIDUAL, gate initialised to ZERO. At step 0 this module contributes exactly
    nothing and the model is bit-identical to convnext_base_25d_18slices (val 0.9010 /
    LB 0.914). Attention has to earn its way in rather than replace a mean that already
    works. convnext_small_transformer_pool REPLACED the mean with self-attention and lost
    (0.8792), which is the failure mode this avoids.
    """

    def __init__(self, feat: int, n_tasks: int, n_slots: int, n_groups: int,
                 heads: int = 8, use_pos: bool = True):
        super().__init__()
        self.n_slots, self.n_groups = n_slots, n_groups
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, feat))
        self.pos_emb = nn.Parameter(torch.zeros(n_groups, feat)) if use_pos else None
        # one query per finding: Fracture may attend to different images than Synovitis
        self.query = nn.Parameter(torch.randn(n_tasks, feat) * 0.02)
        self.attn = nn.MultiheadAttention(feat, heads, batch_first=True)
        self.norm = nn.LayerNorm(feat)          # their kv_norm: LayerNorm on the keys/values
        self.out_norm = nn.LayerNorm(feat)      # their d_norm: LayerNorm on the ATTENDED
                                                # vector before the per-label readout
        # per-finding readout: each task reads its OWN attended vector.
        # RANDOM init, not zeros, and that is load-bearing. The branch output is
        #     gate * ((a . w).sum + b)
        # so with BOTH gate and w at zero, d/d(gate) = (a.w).sum + b = 0 and
        # d/d(w) = gate * a = 0: nothing in the branch ever receives a gradient and the whole
        # thing is permanently dead. The run would train for hours, contribute nothing, and
        # report exactly the parent's score - which reads as "attention does not help" rather
        # than "attention was never connected". Caught by the smoke test, not by training.
        self.w = nn.Parameter(torch.randn(n_tasks, feat) * feat ** -0.5)
        self.b = nn.Parameter(torch.zeros(n_tasks))
        # the gate. Zero means "contribute nothing" at init, so training starts from the
        # known-good mean-pool model and can only add from there. Safe now that w is random:
        # the gate's own gradient is the branch value, which is non-zero.
        # PER-LABEL gate, 12 values not one. Taken from their _GatedDelta, which uses
        # nn.Parameter(torch.zeros(n_labels)): each finding decides independently how much
        # attention to admit, so Fracture can lean on it while Synovitis ignores it. Free,
        # and strictly more expressive than a single scalar.
        self.gate = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, f):                      # f: (B, per_study, feat)
        b, n, d = f.shape
        t = f
        if n == self.n_slots * self.n_groups:  # add identity; skip if the layout disagrees
            t = t.view(b, self.n_slots, self.n_groups, d)
            t = t + self.slot_emb[None, :, None, :]
            if self.pos_emb is not None:
                t = t + self.pos_emb[None, None, :, :]
            t = t.reshape(b, n, d)
        t = self.norm(t)
        q = self.query[None].expand(b, -1, -1)         # (B, n_tasks, feat)
        a, _ = self.attn(q, t, t, need_weights=False)  # each finding pools its own way
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
        sa = m.get("slot_attention") or {}
        self.slot_attn = None
        if sa.get("enable"):
            self.slot_attn = SlotLabelAttention(
                feat, sc.num_output_logits(cfg), len(i["slots"]),
                g if as_ch else int(i["n_slices"]),
                heads=int(sa.get("heads", 8)), use_pos=bool(sa.get("use_pos", True)))
            self.slot_attn.is_fresh = True

    def forward(self, x):                      # x: (B*slots*D, C, H, W)
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
        out = self.head(f)
        if self.slot_attn is not None:
            out = out + self.slot_attn(tokens)        # gate starts at 0: a no-op at step 0
        return out


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
