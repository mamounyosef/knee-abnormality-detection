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
# MEMBERS ARE GROUPED BY LOADER GEOMETRY (slots / n_slices / size / mm_per_px / windowing)
# and the notebook decodes ONCE PER GROUP, via predict_dicom_ensemble_grouped. Members that
# agree still share a single decode pass exactly as before; members that disagree now cost
# one extra pass (0.188 s/study on a T4) instead of being refused outright.
# Normalisation, resize_to and slices_as_channels never split a group: all three are applied
# per member inside prepare_batch, on GPU, from an already-decoded tensor.
# posweight is deliberately NOT here. Added to these three it moves val by 0.0000:
# it is another ConvNeXt-Small on the same data as 4slot, so it makes the same
# mistakes. mricore_vitb and radimagenet_resnet50 earn their slots by being the only
# non-ConvNeXt architectures - radimagenet scores lowest solo (0.8799) and still adds
# +0.0018 to the blend.
#
# convnext_small_25d is NOT here either, and now for a purely SCORING reason: a second
# decode pass is available since predict_dicom_ensemble_grouped, so the old objection (it
# could not share member 0's pass) no longer applies - but it only bought +0.0004 on val,
# inside the noise, and it costs 96 passes per study to do it.
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
    # THE 0.932 SEVEN, with Synovitis handed entirely to convnext_base_fluid.
    # This is the CLEAN isolation test. Submission 56299088 shipped this same routing but
    # ALSO swapped convnext_base_full in for convnext_base_relabel, and scored 0.931. Those
    # are two changes: the routing moves ONE finding by a measured +0.0014 macro, the member
    # swap moves ALL TWELVE and was never validated, because convnext_base_full trained on
    # all 704 val AND all 58 audit studies and so cannot be scored on either. An unmeasured
    # twelve-column change swamps a measured one-column change, so 0.931 said nothing about
    # the routing.
    # convnext_base_relabel is back, which makes this EXACTLY the configuration measured at
    # val 0.9178 / audit 0.9160. The routing is now the only difference from the 0.932 seven.
    # Order MUST match three_route_weights.json exactly.
    ("convnext_base_relabel", "best"),
    ("mricore_vitb_cross_attention", "best"),
    ("mricore_vitb_mae_cross_attention", "best"),
    ("dinov3_convnext_base", "best"),
    ("convnext_base_25d_18slices", "best"),
    ("swinv2_base_in22k", "best"),
    ("convnext_base_25d_poolattn", "best"),
    ("convnext_base_fluid", "best"),
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
BLEND_WEIGHTS = "three_route_weights.json"            # Synovitis, Fracture, PF OA
                               # per-class matrix; verified identical to the
                               # nested form (max logit gap 4.8e-07).
_OLD = None                    # FLAT. The per-class search is fitting noise now: its
                               # out-of-fold estimate is 0.9152 against flat's 0.9156.
                               # For the 9-member branch-group construct instead, set this
                               # to "branch_group_weights.json" (make_branch_group_weights.py
                               # writes it; check_partial_head_blend.py verifies it).
STRESS       = 0                 # >0 = MEASUREMENT run: repeat the 3 public studies this
                              # many times to get a marginal per-study rate, and write
                              # no submission.csv. Use with SUBMIT = False.
PUBLISH      = True
WAIT_FOR_RUN = True
SUBMIT       = True              # REAL SUBMISSION: costs one of the day's 5.
                              # False = push and run the kernel, then stop. Kernel runs are
                              # FREE, so always do that first and read inference_log.txt:
                              # it names every member that scored and the decode-pass count.
                              # Set STRESS > 0 with SUBMIT = False to get marginal per-study
                              # rates before committing to the 9 h cap.
KAGGLE_USER  = "mamounhussam"
SLUG         = "knee-three-route"
# KEEP THIS IN STEP WITH MEMBERS. Submission 56234763 shipped the 18-slice member while
# this text still named the 24-slice one, because the swap changed MEMBERS and SLUG and
# missed NOTE. Kaggle descriptions cannot be edited after submitting, so that record is
# permanently wrong. Re-read this string whenever MEMBERS changes.
NOTE         = ("THREE FINDINGS ROUTED TO A SINGLE MEMBER EACH. Members, each 1/7 on the "
                "other nine findings: convnext_base_relabel, mricore_vitb_cross_attention, "
                "mricore_vitb_mae_cross_attention, dinov3_convnext_base, "
                "convnext_base_25d_18slices, swinv2_base_in22k, convnext_base_25d_poolattn. "
                "Routed, each taking its column outright at weight 1.0 while the others are "
                "switched off on it: SYNOVITIS to convnext_base_fluid (0.8767 against the "
                "seven's 0.8604), FRACTURE to mricore_vitb_mae_cross_attention (0.9403 "
                "against 0.9254), PF OA to dinov3_convnext_base (0.8858 against 0.8762). "
                "Val macro 0.9159 flat -> 0.9193 routed. "
                "Every routed target is a member's OWN best checkpoint, already in the "
                "bundle, so this costs ZERO extra forward passes over the 8-member "
                "Synovitis-only submission and the runtime is unchanged. "
                "Synovitis was chosen a-priori (the fluid specialist was BUILT for it, "
                "10/10 cross-fitted seeds, +0.0096 on the 58-study panel set). Fracture and "
                "PF OA were found by scanning 1,263 scored checkpoints across 58 runs, so "
                "they carry selection risk the Synovitis routing does not.")
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

    # --- PREFLIGHT: everything that has actually gone wrong with a submission ----------
    # Each of these cost a real kernel run or, worse, shipped something that was not what
    # the record says it was.
    problems = []

    built = sorted((out / "weights").glob("m*_model.pt"))
    if len(built) != len(MEMBERS):
        problems.append("built %d model files for %d MEMBERS: %s"
                        % (len(built), len(MEMBERS), [p.name for p in built]))

    # NOTE is the permanent public record and CANNOT be edited after submitting. Submission
    # 56234763 shipped the 18-slice member while its description named the 24-slice one,
    # because the swap changed MEMBERS and SLUG and missed NOTE.
    note_l = NOTE.lower()
    unnamed = [n for n, _ in MEMBERS if n.lower() not in note_l]
    if SUBMIT and len(unnamed) > len(MEMBERS) // 2:
        problems.append("NOTE names only %d of the %d members - is it stale? missing: %s"
                        % (len(MEMBERS) - len(unnamed), len(MEMBERS), unnamed[:4]))

    # a blend that ignores a member it was handed is almost always a wiring mistake
    if blend:
        for name, _ in MEMBERS:
            if all(abs(row.get(name, 0.0)) < 1e-12 for row in blend.values()):
                problems.append("%s carries ZERO weight on every finding - why ship it?"
                                % name)

    if problems:
        raise SystemExit("preflight failed:\n  - " + "\n  - ".join(problems))
    print("  ✅ preflight: %d weight files, NOTE matches MEMBERS, every member votes"
          % len(built))

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
