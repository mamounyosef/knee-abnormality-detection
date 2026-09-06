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

    python -m modal volume get knee-runs <run>/results/checkpoints/best.pt \\
        runs/<run>/results/checkpoints/best.pt

    python runs/others/ensembling/submit_ensemble.py
"""

import json
import sys
from pathlib import Path

# ============================ CONFIG (edit here) ============================
# MEMBER 0 OWNS THE LOADER: every member shares one DICOM decode pass, so they must
# agree on slots / n_slices / size / mm_per_px / windowing. They may differ in
# normalisation (MRI-CORE wants [0,1]) - that is applied per member, on GPU.
# posweight is deliberately NOT here. Added to these three it moves val by 0.0000:
# it is another ConvNeXt-Small on the same data as 4slot, so it makes the same
# mistakes. mricore_vitb and radimagenet_resnet50 earn their slots by being the only
# non-ConvNeXt architectures - radimagenet scores lowest solo (0.8799) and still adds
# +0.0018 to the blend.
#
# convnext_small_25d is NOT here either, and not for a scoring reason: 2.5D is a
# LOADER setting (neighbouring slices in the RGB channels), so it cannot share member
# 0's decode pass. It would need a second pass over every DICOM to buy +0.0004 on val,
# which is inside the noise. predict_dicom_ensemble now rejects it explicitly.
# dinov3_convnext_base is the 5th member. It LOSES solo to convnext_base_relabel
# (0.9011 vs 0.9050) on a byte-identical config, so supervised ImageNet-22k beats DINOv3
# as a standalone model - but it is the largest ensemble gain since the relabel: flat
# 0.9110 -> 0.9127, per-class out-of-fold 0.9116 -> 0.9140.
#
# THIS submission changes two things against the 0.924 one:
#   1. mricore_vitb_cross_attention REPLACES mricore_vitb_relabel. Cross-attention ties
#      it solo (0.8963 vs 0.8970) but fuses across the 96 slices inside the trunk, which
#      nothing else in the pool does - ablating its three cross-slice blocks costs 0.0356
#      AUROC. Adding it on top of mricore_vitb_relabel buys almost nothing (+0.0013);
#      swapping gains +0.0022. They share the MRI-CORE trunk, so they are the most
#      correlated pair in the pool and only one earns a slot.
#   2. PER-CLASS WEIGHTS instead of a flat average, fitted 5-fold and ranked on the
#      held-out estimate. Weighted beat flat by ~+0.002 OOF on every combination tried.
MEMBERS = [
    ("convnext_small_relabel", "best"),
    ("convnext_base_relabel", "best"),
    ("mricore_vitb_cross_attention", "best"),
    ("radimagenet_resnet50_relabel", "best"),
    ("dinov3_convnext_base", "best"),
]
# Per-class blend weights fitted by ensemble.py's search. Point at the summary json of
# the fit whose members match MEMBERS exactly, or None for a flat average.
#
# Ranked on the OUT-OF-FOLD number, never the in-sample one. For this fit:
#   flat 0.9144  |  OOF 0.9166  |  in-sample 0.9198
# The 0.0032 between OOF and in-sample is the overfitting; the 0.0022 between OOF and
# flat is what the weights actually buy, and it is the only one worth acting on.
BLEND_WEIGHTS = ("ensemble_results/2026-09-06_23-01-33_w5-C-for-mricore-search"
                 "/weighted_val_summary.json")
PUBLISH      = True            # push the dataset + notebook and submit
WAIT_FOR_RUN = True            # poll until the notebook finishes, then submit
KAGGLE_USER  = "mamounhussam"
SLUG         = "knee-ensemble-5-crossattn-weighted"   # dataset slug; notebook is <SLUG>-submit
NOTE         = ("5-model weighted ensemble: convnext_small + convnext_base + "
                "mricore_vitb_cross_attention + radimagenet + dinov3 | per-class weights, "
                "val OOF 0.9166 (flat 0.9144) | cross_attention REPLACES mricore_vitb_relabel")
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

    blend = None
    if BLEND_WEIGHTS:
        wf = Path(__file__).resolve().parent / BLEND_WEIGHTS
        if not wf.exists():
            raise SystemExit(f"BLEND_WEIGHTS points at nothing: {wf}")
        fit = json.loads(wf.read_text(encoding="utf-8"))
        fit_members = fit.get("members") or []
        want = [n for n, _ in MEMBERS]
        # a fit whose members differ from MEMBERS would silently weight the wrong models
        if fit_members != want:
            raise SystemExit(
                f"the weight fit does not match MEMBERS.\n"
                f"  fit:     {fit_members}\n"
                f"  MEMBERS: {want}\n"
                f"  re-run the search for these members, or set BLEND_WEIGHTS = None")
        blend = fit["weights_per_class"]
        print(f"  \u2696\ufe0f  weights from {wf.name}: OOF "
              f"{fit.get('oof_mean_auroc', float('nan')):.4f}  "
              f"flat {fit.get('flat_mean_auroc', float('nan')):.4f}")

    sc._banner(f"ENSEMBLE SUBMISSION ({len(members)} members)", "\U0001F9E9")
    out = sc.build_kaggle_ensemble_assets(
        members, Path(__file__).resolve().parent / "kaggle", KAGGLE_USER,
        SLUG, f"{SLUG}-submit", title=f"knee ensemble {len(members)}",
        blend_weights=blend)

    if not PUBLISH:
        print("PUBLISH is False - assets built, nothing uploaded")
        return
    sc.kaggle_publish(out, KAGGLE_USER, SLUG, f"{SLUG}-submit",
                      message=NOTE, wait=WAIT_FOR_RUN)


if __name__ == "__main__":
    main()
