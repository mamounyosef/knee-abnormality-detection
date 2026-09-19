"""
train.py — convnext_base_25d_18slices

Thin entry point: load config, build the model, hand off to the engine.

Local:  python runs/convnext_base_25d_18slices/train.py
Modal:  modal run runs/convnext_base_25d_18slices/train.py
"""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

EXP_NAME = "convnext_base_25d_perfinding_slots_and_pooling"


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



# --- per-finding pooling, over per-finding VIEWS -------------------------------------
# POOL_MAP: HOW each finding reduces its windows.
# MAX IS GONE. convnext_base_25d_perfinding_pool_features tested it from scratch and it
# lost 0 of 4 columns, with the worst single loss being Fracture -0.0175 - the very
# finding whose 'one suspicious window decides it' story motivated max in the first place.
# top2 stays, because MCL gained +0.0329 (0.8384 -> 0.8713) in that same run: the MCL is a
# flat band on the inner edge of the knee, present in very few coronal windows, so
# averaging all of them dilutes it while 'average the two most suspicious' does not.
POOL_MAP = {'top2': ['ACL', 'MCL'],
            'mean': ['Fracture', 'Contusion', 'Medial Meniscus', 'Lateral Meniscus',
                     'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's"]}

# SLOT_MAP: WHICH views each finding is read on. A knee is shot in three planes (sagittal =
# sliced side to side, coronal = sliced front to back, axial = sliced across the leg like
# a loaf) and two sequence types (fluid = fat-suppressed, so water and swelling glow white;
# plain = ordinary contrast, so anatomy is crisp).
#   ACL runs diagonally through the middle of the joint, so only sagittal sees its length.
#   MCL is a flat band on the inner edge; its whole length lies in one coronal plane.
#   The menisci are C-shaped cushions read along sagittal, with coronal to place them.
#   OA is a medial-versus-lateral comparison of joint space, which is what coronal shows.
#   PF OA is cartilage behind the kneecap, seen face-on only on axial.
#   Effusion, Synovitis, Baker's and Contusion are all fluid or swelling: invisible on plain.
#   A Fracture can be anywhere and needs plain for the sharp cortical line, so it keeps all six.
SLOT_MAP = {'ACL': ['sag_fluid', 'sag_plain'], 'MCL': ['cor_fluid', 'cor_plain'], 'Medial Meniscus': ['sag_fluid', 'sag_plain', 'cor_fluid'], 'Lateral Meniscus': ['sag_fluid', 'sag_plain', 'cor_fluid'], 'Medial OA': ['cor_fluid', 'cor_plain'], 'Lateral OA': ['cor_fluid', 'cor_plain'], 'PF OA': ['ax_fluid', 'ax_plain', 'sag_plain'], 'Effusion': ['ax_fluid', 'sag_fluid'], 'Synovitis': ['sag_fluid', 'cor_fluid', 'ax_fluid'], "Baker's": ['ax_fluid', 'sag_fluid'], 'Contusion': ['sag_fluid', 'cor_fluid', 'ax_fluid'], 'Fracture': ['sag_plain', 'sag_fluid', 'cor_plain', 'cor_fluid', 'ax_plain', 'ax_fluid']}


def _reduce(t, kind):
    """t: (..., N, C) -> (..., C), reducing the instance axis."""
    if kind == "mean":
        return t.mean(dim=-2)
    if kind == "max":
        return t.max(dim=-2).values
    if kind == "top2":
        return t.topk(min(2, t.shape[-2]), dim=-2).values.mean(dim=-2)
    raise ValueError(kind)


class MILModel(nn.Module):
    """2D backbone over all 36 windows; each finding pools its OWN views, its OWN way.

    Unlike the 4-slot parent, no two findings are guaranteed to share a pooled vector -
    each one reduces a different subset of windows - so each gets its own norm and its own
    single-output classifier rather than one Linear per pooling group.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        extra = dict(m.get("create_kwargs") or {})
        self.backbone = timm.create_model(m["name"], pretrained=m.get("pretrained", True),
                                          num_classes=0, drop_rate=m.get("drop_rate", 0.0),
                                          drop_path_rate=m.get("drop_path_rate", 0.0), **extra)
        if m.get("grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        feat = self.backbone.num_features
        self.per_study = sc.images_per_study(cfg)          # 36 windows: 6 slots x 6 groups
        self.n_out = sc.num_output_logits(cfg)
        tasks = list(cfg["tasks"])
        slots = list(cfg["input"]["slots"])
        # slots no longer contribute the same number of windows - coronal samples 24 slices
        # against everyone else's 18 - so each slot's start index is the running total of
        # the ones before it, not a fixed stride.
        n_win = [sc.slot_groups(cfg, s) for s in slots]
        start, acc = [], 0
        for n in n_win:
            start.append(acc)
            acc += n

        # windows are laid out SLOT-MAJOR: slot 0's windows, then slot 1's, and so on
        self.kinds, self.windows = [], []
        for t_name in tasks:
            k = next((k for k, v in POOL_MAP.items() if t_name in v), None)
            if k is None:
                raise ValueError("no pooling assigned for %s" % t_name)
            self.kinds.append(k)
            allowed = [s for s in SLOT_MAP.get(t_name, slots) if s in slots]
            if not allowed:
                raise ValueError("no usable view for %s; config has %s" % (t_name, slots))
            self.windows.append([start[si] + j
                                 for si, s in enumerate(slots) if s in allowed
                                 for j in range(n_win[si])])

        # one norm and one 1-output head PER FINDING. A max over 36 windows is
        # systematically larger than a mean over 12, so the pooled vectors arrive on
        # different scales and a shared norm would be miscalibrated for most of them.
        self.norms = nn.ModuleList([nn.LayerNorm(feat) for _ in tasks])
        self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in tasks])

    def forward(self, x):
        f = self.backbone(x)                               # (B*36, feat)
        f = f.view(-1, self.per_study, f.shape[-1])        # (B, 36, feat)
        cols = []
        for j, kind in enumerate(self.kinds):
            v = _reduce(f[:, self.windows[j], :], kind)    # (B, w_j, feat) -> (B, feat)
            cols.append(self.heads[j](self.norms[j](v)))   # (B, 1)
        return torch.cat(cols, dim=1)                      # (B, 12)

def build_model(cfg: dict) -> nn.Module:
    return MILModel(cfg)


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
