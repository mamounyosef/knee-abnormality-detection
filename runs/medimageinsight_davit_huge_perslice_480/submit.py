"""
submit.py — build everything Kaggle needs from a finished run.

Kaggle is a code competition: the scored inference runs inside a NOTEBOOK with no
internet, against a hidden test set. So this script cannot produce the real
submission.csv — it produces the pieces the notebook needs, and the notebook is
pushed with the Kaggle API (no browser clicks):

    weights/    model.pt + config.json + shared_code.py + model_def.py -> a Kaggle dataset
    notebook/   submit.ipynb + kernel-metadata.json                    -> a Kaggle notebook

RUN_LOCAL=False does the work on Modal (the checkpoints and the raw DICOM zip both
live there). Set it True to do it on this PC — useful when a workspace is out of
credit, and it needs the checkpoint already synced locally.

    python -m modal run runs/convnext_base_25d_18slices/submit.py       # RUN_LOCAL = False
    python runs/convnext_base_25d_18slices/submit.py                    # RUN_LOCAL = True
"""

import sys
from pathlib import Path

RUN_LOCAL = False              # True = do it here, False = do it on Modal
PUBLISH = True                 # after building: push the dataset + notebook and submit
WAIT_FOR_RUN = True            # poll until the notebook finishes, then submit
KAGGLE_USER = "mamounhussam"   # owner slug for the dataset and the notebook
CHECKPOINT = "best"            # best | last | top1.. | <step> | path
VERIFY_STUDIES = 3             # DICOM-vs-shard check; 0 skips it

EXP_NAME = "medimageinsight_davit_huge_perslice_480"


def _pkg_root() -> Path:
    for c in (Path("/root/runs"), Path(__file__).resolve().parent.parent):
        if (c / "shared_code.py").exists():
            return c
    return Path(__file__).resolve().parent.parent


PKG_ROOT = _pkg_root()
EXP_DIR = PKG_ROOT / EXP_NAME
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(EXP_DIR))
import shared_code as sc  # noqa: E402


def build(cfg, exp_dir, out_dir, ckpt_dir, data_zip=None):
    """Verify the DICOM path still matches the shards, then write the Kaggle assets."""
    import pandas as pd

    if VERIFY_STUDIES and data_zip:
        import shutil, tempfile, zipfile
        print(f"\n🔬 verifying the DICOM path against the stored shards "
              f"({VERIFY_STUDIES} studies)")
        series = pd.read_csv(sc.resolve("train_series.csv", cfg))
        with zipfile.ZipFile(data_zip) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
            studies = []
            for n in names:                       # <split>/<study>/<series>/<file>.dcm
                parts = Path(n).parts
                if len(parts) >= 3 and parts[-3] not in studies:
                    studies.append(parts[-3])
                if len(studies) >= VERIFY_STUDIES:
                    break
            tmp = Path(tempfile.mkdtemp())
            for n in names:
                if any(f"/{s}/" in f"/{n}" for s in studies):
                    zf.extract(n, tmp)
            ok = sc.verify_dicom_path(cfg, tmp, series, limit=VERIFY_STUDIES * 4)
            shutil.rmtree(tmp, ignore_errors=True)
        if not ok:
            raise SystemExit("❌ DICOM path does not reproduce the shards — fix before submitting")

    ck = sc._resolve_resume(CHECKPOINT, Path(ckpt_dir))
    if ck is None:
        raise SystemExit(f"no checkpoint {CHECKPOINT!r} in {ckpt_dir}")
    print(f"\n📦 packaging {ck}")
    return sc.build_kaggle_assets(cfg, exp_dir, out_dir, ck, KAGGLE_USER)


def _publish(out):
    """Kaggle steps run from this PC either way — the container has no credentials."""
    if not PUBLISH:
        print("PUBLISH is False — assets built, nothing uploaded")
        return
    slug = f"knee-{EXP_NAME.replace('_', '-')}"
    cfg = sc.load_config(EXP_DIR, verbose=False)
    sc.kaggle_publish(out, KAGGLE_USER, slug, f"{slug}-submit",
                      message=sc.submission_message(cfg, EXP_DIR), wait=WAIT_FOR_RUN)


def run_local():
    cfg = sc.load_config(EXP_DIR, verbose=False)
    res = EXP_DIR / cfg["output"]["run_dir"]
    out = build(cfg, EXP_DIR, EXP_DIR / "kaggle", res / cfg["output"]["checkpoints_dir"])
    _publish(out)


try:
    import modal

    cfg = sc.load_config(EXP_DIR, verbose=False)
    app = modal.App(f"{EXP_NAME}-submit")
    _image = sc.modal_image()
    _runs = modal.Volume.from_name(cfg["modal"]["runs_volume"], create_if_missing=True)
    _shards = modal.Volume.from_name(cfg["modal"]["shards_volume"], create_if_missing=True)
    _data = modal.Volume.from_name("knee-data", create_if_missing=True)

    @app.function(image=_image, timeout=3600, memory=16384,
                  volumes={cfg["modal"]["runs_mount"]: _runs,
                           cfg["modal"]["shards_mount"]: _shards, "/data": _data})
    def build_remote():
        """Returns the assets as bytes — the container cannot write to this PC."""
        import io, zipfile as zf2
        rcfg = sc.remote_cfg(cfg)
        out_dir = Path("/tmp/kaggle")
        res = Path(cfg["modal"]["runs_mount"]) / cfg["experiment"]["name"] / rcfg["output"]["run_dir"]
        archive = "/data/rsna-knee-abnormality-detection.zip"
        build(rcfg, Path("/root/runs") / EXP_NAME, out_dir,
              res / rcfg["output"]["checkpoints_dir"],
              data_zip=archive if Path(archive).exists() else None)
        buf = io.BytesIO()
        with zf2.ZipFile(buf, "w", zf2.ZIP_DEFLATED) as z:
            for f in out_dir.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(out_dir))
        return buf.getvalue()

    @app.local_entrypoint()
    def modal_main():
        blob = build_remote.remote()
        import io, zipfile as zf2
        out = EXP_DIR / "kaggle"
        out.mkdir(parents=True, exist_ok=True)
        with zf2.ZipFile(io.BytesIO(blob)) as z:
            z.extractall(out)
        print(f"\n⬇️  pulled the Kaggle assets to {out}")
        _publish(out)      # credentials live on this PC, not in the container

except ImportError:
    pass


if __name__ == "__main__":
    if RUN_LOCAL:
        run_local()
    else:
        print("RUN_LOCAL is False — launch with:  python -m modal run "
              f"runs/{EXP_NAME}/submit.py")
