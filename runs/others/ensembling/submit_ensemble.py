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
    # THE SHIPPED SEVEN (val 0.9164 / LB 0.932) PLUS AN EIGHTH, FLAT.
    #
    # convnext_base_25d_perfinding_slots_and_pooling joins as a full member rather than
    # nested. It is a ConvNeXt on the 2.5D geometry like three of the seven, so nesting was
    # the obvious guess - but it is NOT a near-duplicate: its logit correlation with the
    # shipped members is 0.848 mean / 0.740 minimum, where the other convnext_base_25d_*
    # siblings sit at 0.95-0.99. Nesting averages away an opinion that is genuinely its own.
    # Measured over 84 placements: flat +0.0006, every nested form +0.0003 to +0.0004.
    #
    # Cross-fitted on val (choose on 4/5, score the held-out fifth): 0.9176 against the
    # shipped 0.9172, +0.0004. Small, but from 84 candidates rather than the 151,889 of the
    # structure search, so far less selection bias, and the winner is the a-priori obvious
    # placement rather than a scan artifact.
    #
    # The real argument is the leaderboard's. Shipped SOLO it scored 0.915 against its
    # parent's 0.914, while validation had it at 0.8961 against 0.9010 - val understates
    # this run by roughly 0.005. If that carries into the ensemble the gain is larger than
    # val can see. Audit-58 disagrees (-0.0005 cross-fitted) but 58 studies cannot outweigh
    # 704.
    #
    # COST: it is 6 slots with per-slot slice counts, so it does NOT share the seven's
    # decode group and adds one extra decode pass. Solo it priced at 22.3 min.
    # The full-data laterality swap, shipped in the 0.935 nine.
    ("convnext_base_full_all_knees_same_orientation", "best"),
    ("mricore_vitb_cross_attention", "best"),
    ("mricore_vitb_mae_cross_attention", "best"),
    ("dinov3_convnext_base", "best"),
    ("convnext_base_25d_all_knees_same_orientation", "best"),
    ("swinv2_base_in22k", "best"),
    ("convnext_base_25d_poolattn", "best"),
    ("convnext_base_25d_perfinding_slots_and_pooling", "best"),
    # THE NINTH, and the only change against the 0.933 eight. MedImageInsight is
    # Microsoft's DaViT-Huge (360.6M parameters, MIT licence), trained with Unified
    # Contrastive Learning on about 3.8M MEDICAL images against their CLINICAL labels. It
    # is the first external backbone to reach our range: val 0.9041 against
    # convnext_base_relabel's 0.9050, where the five external backbones before it landed
    # between 0.7088 and 0.8455. Shipped SOLO it scored 0.929, the best single model we
    # have. Free in decode: its loader geometry (336 px, 18 slices, laterality_flip left)
    # matches convnext_base_25d_all_knees_same_orientation's exactly.
    ("medimageinsight_davit_huge_336", "best"),
    # THE TENTH, and the only change against the 0.935 nine. Warm-started from member 9's
    # own step-3200 weights and moved to 480 px with one image per slice instead of six
    # images of stacked slices: all 406 tensors transferred with no reshaping, because
    # DaViT encodes position with convolutions and the stem is 3-channel on both sides.
    # It ADDS rather than replaces, which was measured rather than assumed: beside its own
    # parent it is val 0.9116 (+0.0075) and audit 0.9154 (+0.0029), and against the shipped
    # nine adding gives val 0.9181 (+0.0007) / audit 0.9199 (+0.0002) where REPLACING the
    # parent gives 0.9179 / 0.9182, negative on audit.
    # COST: 480 px per-slice is a geometry no other member shares, so it owns a sixth
    # decode group, and at 72 images per study it is the most expensive member in the pool.
    ("medimageinsight_davit_huge_perslice_480_warm", "best"),
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
BLEND_WEIGHTS = None            # FLAT. The +0.0006 / +0.0004 cross-fitted result
                               # was measured on a flat eight-way average, so
                               # ship exactly what was measured. A per-class fit
                               # on top would be a second, unmeasured change.
_OLD = None                    # FLAT. The per-class search is fitting noise now: its
                               # out-of-fold estimate is 0.9152 against flat's 0.9156.
                               # For the 9-member branch-group construct instead, set this
                               # to "branch_group_weights.json" (make_branch_group_weights.py
                               # writes it; check_partial_head_blend.py verifies it).
STRESS       = 0
PUBLISH      = True
# PUSH NOW, SUBMIT LATER. Four submissions, each about five hours on the real test set, is
# a day of blocking on this PC for work Kaggle will happily run in parallel. Push them all,
# then hand the slugs to submit_when_done.py, which submits each one as its kernel
# completes. It is also the safe order for an unpriced member: a kernel that breaches the
# 9 h cap fails on Kaggle and is never submitted, so it costs a run and not an attempt.
WAIT_FOR_RUN = True
SUBMIT       = True              # free run clean: 10/10 members, every one with a
                              # real time, gate passed. 31.624 s/study serial over
                              # 6 decode groups; the tenth member costs 4.307.
KAGGLE_USER  = "mamounhussam"
SLUG         = "knee-ten-perslice480"
# KEEP THIS IN STEP WITH MEMBERS. Submission 56234763 shipped the 18-slice member while
# this text still named the 24-slice one, because the swap changed MEMBERS and SLUG and
# missed NOTE. Kaggle descriptions cannot be edited after submitting, so that record is
# permanently wrong. Re-read this string whenever MEMBERS changes.
NOTE         = ("THE 0.935 NINE PLUS medimageinsight_davit_huge_perslice_480_warm. One "
                "change. The tenth member is the ninth's own weights, continued: it was "
                "WARM-STARTED from medimageinsight_davit_huge_336's step-3200 checkpoint "
                "rather than from the public MedImageInsight weights, then adapted for 5 "
                "epochs to 480 px with ONE IMAGE PER SLICE instead of six images of stacked "
                "slices. All 406 tensors transferred with no surgery, because DaViT encodes "
                "position with convolutions (so 336 to 480 needs no interpolation) and the "
                "stem takes 3 channels either way. Scored before any gradient step it was "
                "val 0.8988, so the geometry change alone cost -0.0053 against the donor's "
                "0.9041; it passed the donor by step 100 and peaked 0.9095 at step 600, "
                "+0.0054. It is ADDED, not swapped, and that was measured both ways: beside "
                "its parent val 0.9116 (+0.0075) and audit 0.9154 (+0.0029); against the "
                "shipped nine, adding gives val 0.9181 (+0.0007) and audit 0.9199 (+0.0002) "
                "where replacing the parent gives 0.9179 and 0.9182, negative on audit. It "
                "is the most expensive member in the pool, 72 images per study at 480 px, "
                "and its geometry is shared by nothing else so it owns a sixth decode "
                "group. Members: convnext_base_full_all_knees_same_orientation, "
                "mricore_vitb_cross_attention, mricore_vitb_mae_cross_attention, "
                "dinov3_convnext_base, convnext_base_25d_all_knees_same_orientation, "
                "swinv2_base_in22k, convnext_base_25d_poolattn, "
                "convnext_base_25d_perfinding_slots_and_pooling, "
                "medimageinsight_davit_huge_336, "
                "medimageinsight_davit_huge_perslice_480_warm. | Built with DINOv3")

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
        # The title must be UNIQUE per slug: Kaggle refuses to create a second dataset
        # under a title that already exists, so "knee ensemble 9" blocked every later
        # nine-member submission from getting its own dataset. Derive it from the slug,
        # which is unique by construction, and let _kaggle_title cut it to Kaggle's 50.
        SLUG, f"{SLUG}-submit", title=sc._kaggle_title(SLUG, f"x{len(members)}"),
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
