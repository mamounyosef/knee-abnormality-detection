# GREEDY5 (12 Oct, user: submit it): greedy_from_raptor.py's full-121 build. convnext_base_v5_raptor96_local @9000 x0.286,
# dinov3_convnext_base_v5 @5800 x0.272, convnext_base_25d_spatial @2800 x0.170, convnext_base_25d_v5_attributes @3700 x0.150,
# medimageinsight_davit_huge_v5_raptor96_sam @5300 x0.122 (logit, every finding), no public notebook. Local audit 0.9348,
# clean-121 0.9209; the cross-fit says its picks after dinov3 do not hold out, so this is a test of the search itself.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/greedy5_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
_R = r"D:\My Projects\knee_abnormality_detection\runs\{}\results\checkpoints\{}"
MEMBERS = [("convnext_base_v5_raptor96_local", _R.format("convnext_base_v5_raptor96_local", "top_step9000.pt")),
           ("dinov3_convnext_base_v5", _R.format("dinov3_convnext_base_v5", "top_step5800.pt")),
           ("convnext_base_25d_spatial", _R.format("convnext_base_25d_spatial", "top_step2800.pt")),
           ("convnext_base_25d_v5_attributes", _R.format("convnext_base_25d_v5_attributes", "top_step3700.pt")),
           ("medimageinsight_davit_huge_v5_raptor96_sam", _R.format("medimageinsight_davit_huge_v5_raptor96_sam", "top_step5300.pt"))]
BLEND_WEIGHTS = "greedy5_weights.json"
PUBLIC_SIDE = None
SLUG = "knee-greedy5"
NOTEBOOK_SLUG = "knee-greedy5-a"
OUT_DIR = Path("D:/temp1/greedy5")   # C: is full
NOTE = ("GREEDY5: " + ", ".join(n for n, _ in MEMBERS) + " at 0.286 / 0.272 / 0.170 / 0.150 / 0.122 (logit), no public "
        "blend. | Built with DINOv3")
