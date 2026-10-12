# RAPTOR4 (12 Oct, user: submit): manual build from the raptor pair. convnext_base_v5_raptor96_local @9000 x0.42,
# medimageinsight_davit_huge_v5_raptor96_sam @5300 x0.18, dinov3_convnext_base_v5 @5800 x0.20, swinv2_base_in22k_v5_warm
# @2000 x0.20 (logit, same weights on every finding), no public notebook. Local: audit 0.9335, clean-121 0.9209 (pair
# 0.9305 / 0.9175, LB 0.940). Two geometry groups -> the full stress gate applies.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/raptor4_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
_R = r"D:\My Projects\knee_abnormality_detection\runs\{}\results\checkpoints\{}"
MEMBERS = [("convnext_base_v5_raptor96_local", _R.format("convnext_base_v5_raptor96_local", "top_step9000.pt")),
           ("medimageinsight_davit_huge_v5_raptor96_sam", _R.format("medimageinsight_davit_huge_v5_raptor96_sam", "top_step5300.pt")),
           ("dinov3_convnext_base_v5", _R.format("dinov3_convnext_base_v5", "top_step5800.pt")),
           ("swinv2_base_in22k_v5_warm", _R.format("swinv2_base_in22k_v5_warm", "top_step2000.pt"))]
BLEND_WEIGHTS = "raptor4_weights.json"
PUBLIC_SIDE = None
SLUG = "knee-raptor4"
NOTEBOOK_SLUG = "knee-raptor4-a"
OUT_DIR = Path("D:/temp1/raptor4")   # C: is full
NOTE = ("RAPTOR4: " + ", ".join(n for n, _ in MEMBERS) + " at 0.42 / 0.18 / 0.20 / 0.20 (logit), no public blend. "
        "| Built with DINOv3")
