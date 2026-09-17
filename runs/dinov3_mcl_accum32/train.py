"""
train.py — dinov3_convnext_base

ConvNeXt-Base distilled from the DINOv3 ViT-7B teacher on LVD-1689M (1.7B natural
images, self-supervised). Same architecture as convnext_base_relabel - only the
pretraining recipe differs, so this isolates weights from architecture.

Built with DINOv3.

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/dinov3_convnext_base/train.py
Modal:  modal run runs/dinov3_convnext_base/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "dinov3_mcl_accum32"


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
        # anything the architecture needs beyond the defaults: img_size / window_size for
        # the models pretrained at 224 or 256 that we run at 512
        extra = dict(m.get("create_kwargs") or {})
        # num_classes=0 strips the classifier, leaving a feature vector per image
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True),
                                          num_classes=0, drop_rate=m.get("drop_rate", 0.0),
                                          drop_path_rate=m.get("drop_path_rate", 0.0), **extra)
        if m.get("grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        feat = self.backbone.num_features
        # Normalize what the head receives. The backbone's own final LayerNorm ends in a
        # LEARNED gain that ships with the weights, and it differs wildly between
        # checkpoints - DINOv3's is ~2x ImageNet-22k's, which handed the head features
        # 4.8x larger, saturating the logits and inflating head gradients until they hit
        # grad_clip. This is a fresh norm (gamma=1, beta=0), so whatever scale the trunk
        # produces is divided out and the head's lr means the same thing for any trunk.
        self.feat_norm = nn.LayerNorm(feat) if m.get("feat_norm") else nn.Identity()
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
        return self.head(self.feat_norm(f))



# --- specialist seeding -------------------------------------------------------------
BRANCH_FROM = "dinov3_mcl_slice_ft/results/checkpoints/best.pt"
PARENT_TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
HEAD_INIT = "keep"           # "keep" = the checkpoint IS a 1-logit MCL model
                             # already, so load it whole. "slice" would index a
                             # 1-row head and "fresh" would discard the 0.8837.
REQUIRE_PARENT = False   # set True by the TRAINING entry points below


def _branch_init(model, cfg):
    """Seed from the parent checkpoint; slice or drop the head per HEAD_INIT.

    The parent head is Linear(1024 -> 12) and row i produces finding i with no cross-talk
    between rows, so copying row MCL into a (1, 1024) head reproduces the parent's MCL
    logits exactly. "fresh" keeps the backbone and leaves that row random instead.

    Nothing else in the checkpoint depends on which slots were fed: the backbone runs over
    one 2D image at a time and the pooling is a plain mean, so changing the slot set and the
    slice count moves no weight out of place.

    A missing parent is fatal when TRAINING (it would silently train from the pretrained
    weights and look normal) but fine at inference, where the branch's own weights
    immediately overwrite this.
    """
    src = None
    for c in (Path(cfg["modal"]["runs_mount"]) / BRANCH_FROM,
              PKG_ROOT.parent / "runs" / BRANCH_FROM):
        if c.exists():
            src = c
            break
    if src is None:
        if REQUIRE_PARENT:
            raise SystemExit(
                "branch seeding FAILED: %s is on neither the runs volume nor this PC.\n"
                "Training would silently start from the pretrained backbone instead of the\n"
                "parent, which is the whole point of this run. Copy the checkpoint to this\n"
                "workspace's runs volume first." % BRANCH_FROM)
        print("  no parent checkpoint (%s) - fine for inference, WRONG for training"
              % BRANCH_FROM, flush=True)
        return model

    sd = dict(torch.load(src, map_location="cpu", weights_only=False)["model"])
    keep = [PARENT_TASKS.index(t) for t in cfg["tasks"]]
    if HEAD_INIT == "keep":
        # continuing this specialist's OWN run: the head is already one row per task in
        # cfg["tasks"], so it transfers untouched. Slicing it would index row 1 of a 1-row
        # matrix and dropping it would throw away the score we are continuing from.
        want = (sc.num_output_logits(cfg), sd["head.weight"].shape[1])
        if tuple(sd["head.weight"].shape) != want:
            raise RuntimeError("head_init=keep wants a %s head, checkpoint has %s"
                               % (want, tuple(sd["head.weight"].shape)))
    elif HEAD_INIT == "slice":
        idx = torch.tensor(keep)
        sd["head.weight"] = sd["head.weight"][idx]
        sd["head.bias"] = sd["head.bias"][idx]
    else:
        # drop the head entirely and keep whatever the fresh model initialised
        sd.pop("head.weight", None)
        sd.pop("head.bias", None)
    r = model.load_state_dict(sd, strict=False)
    unexpected = list(r.unexpected_keys)
    missing = list(r.missing_keys)
    bad = [k for k in missing if "head" not in k]
    if unexpected or bad:
        raise RuntimeError("branch load mismatch - missing=%s unexpected=%s"
                           % (missing, unexpected))
    print("  seeded from %s (head_init=%s, rows %s -> %s)"
          % (src.name, HEAD_INIT, keep, ", ".join(cfg["tasks"])), flush=True)
    return model


def build_model(cfg: dict) -> nn.Module:
    return _branch_init(SliceModel(cfg), cfg)


def run_local():
    """Train on this PC's GPU — only useful for smoke tests, real runs go to Modal."""
    global REQUIRE_PARENT
    REQUIRE_PARENT = True
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
        global REQUIRE_PARENT
        REQUIRE_PARENT = True
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
