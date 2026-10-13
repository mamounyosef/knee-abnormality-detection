# RAPTOR4 + RAPTOR FOLDS (13 Oct, user: submit): raptor4 (LB 0.942) with the 5 raptor96 folds added as one vote at 0.10,
# the four raptor4 members scaled x0.9: convnext_base_v5_raptor96_local 0.378, medimageinsight_davit_huge_v5_raptor96_sam
# 0.162, dinov3_convnext_base_v5 0.18, swinv2_base_in22k_v5_warm 0.18, convnext_base_v5_raptor96_f0..f4 0.02 each
# (logit). Local: audit 0.9340 / clean-121 0.9210 (raptor4 0.9335 / 0.9209). Folds at their best-val steps
# 7000 / 6960 / 6250 / 6928 / 5250. Two geometry groups -> the full stress gate applies.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/raptor4_folds_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\raptor4.py")).read_text(encoding="utf-8"))
FOLD_STEPS = [7000, 6960, 6250, 6928, 5250]
MEMBERS = MEMBERS + [(f"convnext_base_v5_raptor96_f{k}", _R.format(f"convnext_base_v5_raptor96_f{k}", f"top_step{s}.pt"))
                     for k, s in enumerate(FOLD_STEPS)]
BLEND_WEIGHTS = "raptor4_folds_weights.json"
SLUG = "knee-raptor4-folds"
NOTEBOOK_SLUG = "knee-raptor4-folds-a"
OUT_DIR = Path("D:/temp1/raptor4_folds")
NOTE = ("RAPTOR4 + RAPTOR FOLDS: " + ", ".join(n for n, _ in MEMBERS) + " at 0.378 / 0.162 / 0.18 / 0.18 / 0.02 x5 "
        "(logit), no public blend. | Built with DINOv3")
