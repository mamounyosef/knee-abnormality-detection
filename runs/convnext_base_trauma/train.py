"""
train.py - convnext_base_trauma

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_trauma/train.py
Modal:  modal run runs/convnext_base_trauma/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_trauma"


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

BRANCH_FROM = "convnext_base_relabel/results/checkpoints/top_step2200.pt"
PARENT_TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

cfg = sc.load_config(EXP_DIR, verbose=False)   # shared_config.yaml + this folder's config.yaml


class SliceModel(nn.Module):
    """2D backbone over every slice, pooled per study -> 12 logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        # num_classes=0 strips the ImageNet classifier, leaving a feature vector per image
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True),
                                          num_classes=0, drop_rate=m.get("drop_rate", 0.0))
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


# =============================================================================
# branch: start from the shared trunk, keep only this group's head rows
# =============================================================================
def _branch_init(model, cfg):
    """Load the parent checkpoint and cut its 12-row head down to this group's rows.

    The parent trained all 12 findings together. Its head is a Linear(1024 -> 12): row i
    holds the weights that produce finding i. This group only predicts len(tasks) of them,
    so the head here is Linear(1024 -> len(tasks)) and we copy across exactly the rows for
    the findings we kept, in this config's task order. Everything else (the whole backbone)
    transfers unchanged.

    Without this the load would simply fail on a (12, 1024) vs (n, 1024) shape mismatch,
    and re-initialising the head instead would throw away 5 epochs of training.
    """
    src = None
    for c in (Path(cfg["modal"]["runs_mount"]) / BRANCH_FROM,
              PKG_ROOT.parent / "runs" / BRANCH_FROM):
        if c.exists():
            src = c
            break
    if src is None:
        raise FileNotFoundError(f"parent checkpoint not found: {BRANCH_FROM}")

    sd = torch.load(src, map_location="cpu", weights_only=False)["model"]
    keep = [PARENT_TASKS.index(t) for t in cfg["tasks"]]     # row numbers to carry over
    idx = torch.tensor(keep)
    sd = dict(sd)
    sd["head.weight"] = sd["head.weight"][idx]               # (12, 1024) -> (n, 1024)
    sd["head.bias"] = sd["head.bias"][idx]                   # (12,)      -> (n,)
    r = model.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(f"branch load mismatch - missing={r.missing_keys} "
                           f"unexpected={r.unexpected_keys}")
    print(f"  \U0001F33F branched from {src.name}: kept head rows {keep} "
          f"({', '.join(cfg['tasks'])})", flush=True)
    return model


def build_model(cfg: dict) -> nn.Module:
    return _branch_init(SliceModel(cfg), cfg)


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
