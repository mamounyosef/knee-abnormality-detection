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
#
# THIS submission adds ONE member to the 0.929 five, for val 0.9147 -> 0.9156:
# mricore_vitb_mae_cross_attention, the cross-attention config rebuilt on the
# MAE-pretrained trunk. It ADDS rather than replaces its non-MAE twin: swapping ties at
# 0.9147, adding gives 0.9156. Its logits correlate only 0.873 with that twin, where two
# SEEDS of one config sit at 0.952 - changing the pretraining objective decorrelates a
# model more than changing the seed does.
#
# The 9-member set (these six plus convnext_base_{trauma,wear,fluid} fused with the parent
# into one vote) scores 0.9160, but is NOT shipped: measured on Kaggle's T4 it prices at
# 6.9-8.3 h of the 9 h cap against 5.5-6.6 h for these six, and +0.0004 val is well inside
# the 0.0055 noise. A timeout forfeits the run, so the cap decides. See runtime_budget.py.
MEMBERS = [
    ("convnext_base_relabel", "best"),
    ("mricore_vitb_cross_attention", "best"),
    ("mricore_vitb_mae_cross_attention", "best"),
    ("dinov3_convnext_base", "best"),
    ("convnext_large", "best"),
    ("swinv2_base_in22k", "best"),
]
# RadImageNet is deliberately ABSENT: its licence is "personal, non-commercial (e.g.,
# academic) research purposes only... any commercial use, sale, or other monetization is
# strictly prohibited", and a cash prize is a commercial use. Costs 0.0021 OOF.
#
# convnext_large IS here now, with the RIGHT checkpoint. The local best.pt used to be
# step 1200 / 0.8780 - a stale copy from the workspace where the run started - while the
# run actually continued elsewhere and peaked at step 2600 / 0.9061. The sweep scores
# members from results/predictions/*.npz (0.9061) but the bundle ships the CHECKPOINT
# (0.8780), so every earlier submission carrying convnext_large shipped a model 0.028
# worse than the one measured, including the 6-member that scored 0.922. Fixed: best.pt
# is now step 2600 and its `score` matches val_log's peak. ALWAYS check that before a
# submission - it is silent otherwise.
BLEND_WEIGHTS = None           # FLAT. The per-class search is fitting noise now: its
                               # out-of-fold estimate is 0.9152 against flat's 0.9156.
                               # For the 9-member branch-group construct instead, set this
                               # to "branch_group_weights.json" (make_branch_group_weights.py
                               # writes it; check_partial_head_blend.py verifies it).
STRESS       = 0              # >0 = MEASUREMENT run: repeat the 3 public studies this
                              # many times to get a marginal per-study rate, and write
                              # no submission.csv. Use with SUBMIT = False.
PUBLISH      = True
WAIT_FOR_RUN = True
SUBMIT       = True           # REAL SUBMISSION: costs one of the day's 5.
                              # False = run the kernel and stop. Kernel runs are FREE;
                              # only the submit costs one of the day's 5. Use it to see
                              # the notebook's real output before spending one.
KAGGLE_USER  = "mamounhussam"
SLUG         = "knee-ensemble-6-mae-flat"
NOTE         = ("6-model FLAT ensemble, PRIZE-ELIGIBLE (no RadImageNet): the 0.929 five "
                "plus mricore_vitb_mae_cross_attention, the cross-attention config "
                "rebuilt on the MAE-pretrained MRI-CORE trunk. It ADDS rather than "
                "replaces its non-MAE twin (swapping ties at 0.9147, adding gives "
                "0.9156): its logits correlate 0.873 with that twin where two SEEDS "
                "of one config sit at 0.952. Chosen over the 9-member 0.9160 set on "
                "RUNTIME - measured on a T4, the nine price at 6.9-8.3 h of the 9 h "
                "cap while this fits in 5.5-6.6 h, for 0.0004 val, inside noise. "
                "| val 0.9156 vs 0.9147 for the five | Built with DINOv3")
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
        blend_weights=blend, stress=STRESS)

    # A name the notebook uses but never defines parses fine, uploads fine, and fails on
    # Kaggle minutes later with the traceback swallowed by IPython's own formatter. Catch
    # it here instead: one second against a full push cycle.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lint_notebook import lint
    bad = lint(out / "notebook" / "submit.ipynb")
    if bad:
        raise SystemExit("the generated notebook would crash on Kaggle:\n  "
                         + "\n  ".join(bad))
    print("  ✅ notebook lint: no undefined names")

    if not PUBLISH:
        print("PUBLISH is False - assets built, nothing uploaded")
        return
    sc.kaggle_publish(out, KAGGLE_USER, SLUG, f"{SLUG}-submit",
                      message=NOTE, wait=WAIT_FOR_RUN, submit=SUBMIT)
    if not SUBMIT:
        print("  SUBMIT is False - the kernel ran, nothing was submitted. Fetch its "
              "submission.csv and inference_log.txt before flipping it True.")


if __name__ == "__main__":
    main()
