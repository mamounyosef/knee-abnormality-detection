r"""extract_features.py - cache the 96 per-slice feature vectors of a trained backbone.

WHY. Every model here pools the 96 slice features with a MEAN, which is order-blind: shuffle
the slices and the output is bit-identical. A sequence head (GRU, transformer) reads them in
anatomical order instead, so it can represent "bright on three CONSECUTIVE slices, dark
either side" - the thing a mean cannot express.

We tried that once, convnext_small_transformer_pool, and it lost (0.8792 against its
baseline's 0.8940). But it ran with lp_ft_epochs 0, so a randomly initialised head trained on
top of a pretrained trunk from step one, which is the documented way to destroy good features
(shared_config's lp_ft_epochs comment; we hit it with siglip2_vitb_512). So the question is
open: is sequence modelling useless here, or did we break it with a random head?

Caching the features settles that cheaply. The backbone runs ONCE over the corpus; after that
every head idea trains on a 4060 in minutes with the 88M trunk out of the loop entirely, and
the frozen-feature mean-pool baseline is one line. Only if a head beats it does an end-to-end
run make sense.

Output, per split, into the runs volume:
    feats_<split>.npy    float16 (n_studies, slots*slices, feat_dim)
    meta_<split>.npz     study_ids, targets, tasks, slots, n_slices

    modal run runs/others/extract_features.py
    modal run runs/others/extract_features.py --run-name dinov3_convnext_base --sets val
"""

import sys
from pathlib import Path

import numpy as np
import torch

RUN_NAME = "convnext_base_relabel"        # best single model: val 0.9050, LB 0.918


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def _load_backbone(cfg, exp_dir, device, checkpoint="best"):
    """The trained model, minus its head - we want the per-image features.

    CODE and WEIGHTS live in different places on Modal: train.py ships in the image at
    /root/runs/<run>/, while checkpoints sit on the runs VOLUME at /runs/<run>/results/.
    Conflating the two gives ModuleNotFoundError: no module named 'train'.
    """
    import importlib.util
    code = PKG_ROOT / RUN_NAME / "train.py"
    if not code.exists():
        raise SystemExit("no train.py at %s" % code)
    spec = importlib.util.spec_from_file_location("_extract_train_%s" % RUN_NAME, code)
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    model = tm.build_model(cfg)
    ck_dir = Path(exp_dir) / cfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"]
    path = sc._resolve_resume(checkpoint, ck_dir)
    if path is None:
        raise SystemExit("no checkpoint %r in %s" % (checkpoint, ck_dir))
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd)
    print("  loaded %s (%s)" % (path.name, RUN_NAME), flush=True)
    return model.to(device).eval()


@torch.no_grad()
def extract(cfg, exp_dir, out_dir, sets=("train", "val"), checkpoint="best"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_backbone(cfg, exp_dir, device, checkpoint)
    per_study = sc.images_per_study(cfg)
    loaders = sc.make_loaders(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in sets:
        if split not in loaders:
            print("  no loader for %r, skipping" % split, flush=True)
            continue
        ds = loaders[split].dataset
        ids = [str(s) for s in ds.df["StudyInstanceUID"]]
        n = len(ids)
        feats, targets, done = None, [], 0
        print("  %s: %d studies x %d images" % (split, n, per_study), flush=True)
        for bi, batch in enumerate(loaders[split]):
            x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) \
                else (batch["x"], batch["y"])
            imgs = sc.prepare_batch(x, cfg, device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                f = model.backbone(imgs)                      # (B*per_study, feat)
            f = f.float().view(-1, per_study, f.shape[-1]).cpu().numpy().astype(np.float16)
            if feats is None:
                feats = np.zeros((n, per_study, f.shape[-1]), np.float16)
            feats[done:done + len(f)] = f
            targets.append(np.asarray(y, np.float32))
            done += len(f)
            if bi % 20 == 0:
                print("    %s %d/%d" % (split, done, n), flush=True)
        np.save(out_dir / ("feats_%s.npy" % split), feats[:done])
        np.savez_compressed(out_dir / ("meta_%s.npz" % split),
                            study_ids=np.array(ids[:done], dtype=object),
                            targets=np.concatenate(targets)[:done],
                            tasks=np.array(cfg["tasks"], dtype=object),
                            slots=np.array(cfg["input"]["slots"], dtype=object),
                            n_slices=cfg["input"]["n_slices"], run=RUN_NAME)
        print("  wrote %s  %s  %.0f MB" % (split, feats[:done].shape,
                                           feats[:done].nbytes / 1e6), flush=True)


try:
    import modal

    _exp = PKG_ROOT / RUN_NAME
    cfg = sc.load_config(_exp, verbose=False)
    app = modal.App("extract-features")
    _image = sc.modal_image()
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)

    @app.function(image=_image,
                  volumes={cfg["modal"]["shards_mount"]: _shards,
                           cfg["modal"]["runs_mount"]: _runs},
                  **sc.modal_resources(cfg))
    def extract_remote(sets: list, checkpoint: str):
        rcfg = sc.remote_cfg(cfg)
        rcfg["evaluation"] = dict(rcfg["evaluation"])
        rcfg["evaluation"]["audit"] = dict(rcfg["evaluation"]["audit"], enable=False)
        exp_dir = Path(cfg["modal"]["runs_mount"]) / RUN_NAME
        out = Path(cfg["modal"]["runs_mount"]) / "others" / "features" / RUN_NAME
        try:
            extract(rcfg, exp_dir, out, tuple(sets), checkpoint)
        finally:
            _runs.commit()

    @app.local_entrypoint()
    def modal_main(sets: str = "train,val", checkpoint: str = "best"):
        extract_remote.remote([s for s in sets.split(",") if s], checkpoint)

except ImportError:
    pass


if __name__ == "__main__":
    _exp = PKG_ROOT / RUN_NAME
    _cfg = sc.load_config(_exp, verbose=False)
    extract(_cfg, _exp, _exp / "results" / "features")
