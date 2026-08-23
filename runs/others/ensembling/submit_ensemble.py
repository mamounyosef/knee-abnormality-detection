"""
submit_ensemble.py — ship an ENSEMBLE of finished runs to Kaggle as one submission.

The per-run submit.py packages one checkpoint. This packages several and averages
their logits, which is where the gain actually is: on val, four members score 0.8983
against 0.8948 for the best single model.

Everything rides in ONE dataset (m<i>_model.pt / m<i>_config.json / m<i>_def.py, flat
because kaggle datasets skip subdirectories) and one notebook that rebuilds each
member under its own module name.

Runs entirely on THIS PC: it needs each member's best.pt synced locally, because a
single Modal workspace does not hold every run. Pull any that are missing with

    $env:MODAL_CONFIG_PATH = "$HOME\\.modal_session<N>.toml"
    python -m modal volume get knee-runs <run>/results/checkpoints/best.pt \\
        runs/<run>/results/checkpoints/best.pt

    python runs/others/ensembling/submit_ensemble.py
"""

import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
# MEMBER 0 OWNS THE LOADER: every member shares one DICOM decode pass, so they must
# agree on slots / n_slices / size / mm_per_px / windowing. They may differ in
# normalisation (MRI-CORE wants [0,1]) - that is applied per member, on GPU.
# posweight is deliberately NOT here. Added to these three it moves val by 0.0000:
# it is another ConvNeXt-Small on the same data as 4slot, so it makes the same
# mistakes. mricore_vitb earns its slot by being the only different architecture.
MEMBERS = [
    ("convnext_small_4slot", "best"),
    ("convnext_base_4slot", "best"),
    ("mricore_vitb", "best"),
]
PUBLISH      = True            # push the dataset + notebook and submit
WAIT_FOR_RUN = True            # poll until the notebook finishes, then submit
KAGGLE_USER  = "mamounhussam"
SLUG         = "knee-ensemble-3"      # dataset slug; notebook is <SLUG>-submit
NOTE         = "3-model ensemble (small_4slot + base_4slot + mricore_vitb) | val 0.8983 | audit58 0.8923"
# ===========================================================================

PKG_ROOT = Path(__file__).resolve().parent.parent.parent      # <repo>/runs
sys.path.insert(0, str(PKG_ROOT))
import shared_code as sc  # noqa: E402


def main():
    members, missing = [], []
    for name, ckpt in MEMBERS:
        exp_dir = PKG_ROOT / name
        cfg = sc.load_config(exp_dir, verbose=False)
        ck_dir = exp_dir / cfg["output"]["run_dir"] / cfg["output"]["checkpoints_dir"]
        path = sc._resolve_resume(ckpt, ck_dir)
        if path is None:
            missing.append((name, ck_dir))
            continue
        members.append((name, cfg, exp_dir, path))
    if missing:
        # loud and specific: a silently dropped member changes what gets submitted
        lines = "\n".join(f"    {n}: nothing at {d}" for n, d in missing)
        raise SystemExit(f"missing checkpoints for {len(missing)} member(s):\n{lines}\n"
                         f"  pull them with `modal volume get` - see this file's header")

    sc._banner(f"ENSEMBLE SUBMISSION ({len(members)} members)", "\U0001F9E9")
    out = sc.build_kaggle_ensemble_assets(
        members, Path(__file__).resolve().parent / "kaggle", KAGGLE_USER,
        SLUG, f"{SLUG}-submit", title=f"knee ensemble {len(members)}")

    if not PUBLISH:
        print("PUBLISH is False - assets built, nothing uploaded")
        return
    sc.kaggle_publish(out, KAGGLE_USER, SLUG, f"{SLUG}-submit",
                      message=NOTE, wait=WAIT_FOR_RUN)


if __name__ == "__main__":
    main()
