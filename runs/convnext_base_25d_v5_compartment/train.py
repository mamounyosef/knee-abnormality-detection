"""
train.py — convnext_base_25d_v5_compartment

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_v5_compartment/train.py
Modal:  modal run runs/convnext_base_25d_v5_compartment/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_25d_v5_compartment"


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

# BRANCH of the parent below: its whole trunk and its full 12-row head (all findings kept)
BRANCH_FROM = "convnext_base_25d_18slices_v5_aug_base/results/checkpoints/top_step5900.pt"   # val 0.9174, v4 labels, v5 split, flip
PARENT_TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


# COMPARTMENT-AWARE POOLING. After the left-knee flip every knee looks like a RIGHT knee, and in all 14,507 coronal and
# axial series the image's right edge is the patient's LEFT (checks/lateral_side_sign_check.py: ImageOrientationPatient
# h[0] > 0 in 100%). So on coronal and axial images the knee's LATERAL side is the image's LEFT half and the MEDIAL side
# its RIGHT half. Sagittal slices cut side to side (medial-lateral is the slice axis), so they have no in-plane halves
# and keep their whole image for every finding.
# Each image's 10x10 feature map is pooled three ways through the backbone's own head (pool + norm): whole, left half,
# right half. Per study: the whole-image average feeds the 7 non-compartment findings exactly as in the parent; the
# lateral average (coronal/axial left halves + sagittal whole) feeds the lateral findings, the medial average the medial
# ones (the MCL is the ligament on the inner edge).
SIDE = {"Lateral Meniscus": "lat", "Lateral OA": "lat",
        "Medial Meniscus": "med", "Medial OA": "med", "MCL": "med"}


class SliceModel(nn.Module):
    """2D backbone over every image; whole / lateral-half / medial-half averages per study -> 12 logits."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        i = cfg["input"]
        g = int(i.get("slices_as_channels") or 0)     # images per slot; 0 = off
        as_ch = g > 0
        in_ch = int(i["n_slices"]) // g if as_ch else int(i["channels"])
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True),
                                          num_classes=0, in_chans=in_ch,
                                          drop_rate=m.get("drop_rate", 0.0))
        feat = self.backbone.num_features
        self.per_slot = g if as_ch else int(i["n_slices"])
        self.per_study = len(i["slots"]) * self.per_slot
        # which of a study's images can be split into halves: coronal and axial slots only (slot-major order)
        halvable = [str(s).split("_")[0] in ("cor", "ax") for s in i["slots"] for _ in range(self.per_slot)]
        self.register_buffer("halvable", torch.tensor(halvable).view(1, -1, 1), persistent=False)
        side = [SIDE.get(t, "all") for t in cfg["tasks"]]
        self.register_buffer("use_lat", torch.tensor([s == "lat" for s in side]), persistent=False)
        self.register_buffer("use_med", torch.tensor([s == "med" for s in side]), persistent=False)
        self.head = nn.Linear(feat, sc.num_output_logits(cfg))   # one logit per finding, 12

    def forward(self, x):                      # x: (B*slots*D, C, H, W)
        fm = self.backbone.forward_features(x)                 # (N, F, h, w)
        w = fm.shape[-1]
        whole = self.backbone.forward_head(fm, pre_logits=True)                    # = backbone(x)
        lat = self.backbone.forward_head(fm[..., : w // 2], pre_logits=True)       # image left = lateral
        med = self.backbone.forward_head(fm[..., w - w // 2:], pre_logits=True)    # image right = medial
        F_ = whole.shape[-1]
        whole = whole.view(-1, self.per_study, F_)
        lat = torch.where(self.halvable, lat.view(-1, self.per_study, F_), whole)  # sagittal keeps the whole image
        med = torch.where(self.halvable, med.view(-1, self.per_study, F_), whole)
        out_all = self.head(whole.mean(dim=1))                                     # (B, 12)
        out_lat = self.head(lat.mean(dim=1))
        out_med = self.head(med.mean(dim=1))
        return torch.where(self.use_lat, out_lat, torch.where(self.use_med, out_med, out_all))


def _branch_init(model, cfg):
    """Load the parent checkpoint and cut its 12-row head down to this group's rows (same as convnext_base_fluid).

    Head row i of the parent produces finding i, so the head here is Linear(1024 -> len(tasks)) seeded with
    exactly the rows of the findings kept, in this config's task order. The whole backbone transfers unchanged.
    """
    src = None
    for c in (Path(cfg["modal"]["runs_mount"]) / BRANCH_FROM, PKG_ROOT.parent / "runs" / BRANCH_FROM):
        if c.exists():
            src = c
            break
    if src is None:
        # only training needs the seed; at inference (Kaggle) the branch's own weights overwrite this model
        print(f"  \U0001F33F no parent checkpoint ({BRANCH_FROM}) - skipping the branch seed; "
              f"fine for inference, WRONG if this is a fresh training run", flush=True)
        return model
    sd = dict(torch.load(src, map_location="cpu", weights_only=False)["model"])
    keep = [PARENT_TASKS.index(t) for t in cfg["tasks"]]     # row numbers to carry over
    idx = torch.tensor(keep)
    sd["head.weight"] = sd["head.weight"][idx]               # (12, 1024) -> (n, 1024)
    sd["head.bias"] = sd["head.bias"][idx]                   # (12,)      -> (n,)
    r = model.load_state_dict(sd, strict=False)
    if r.missing_keys or r.unexpected_keys:
        raise RuntimeError(f"branch load mismatch - missing={r.missing_keys} unexpected={r.unexpected_keys}")
    print(f"  \U0001F33F branched from {src.name}: kept head rows {keep} ({', '.join(cfg['tasks'])})", flush=True)
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
