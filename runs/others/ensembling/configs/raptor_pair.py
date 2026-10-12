# RAPTOR PAIR (11 Oct, user: yes): convnext_base_v5_raptor96_local @9000 (LB 0.939) + medimageinsight_davit_huge_v5_raptor96_sam
# @5300 (LB 0.937), logit weights 0.7 / 0.3 on every finding, no public notebook. runs/others/pair_blend_test.py: val-346
# 0.9310 (+0.0018), audit 0.9305 (+0.0034), clean 0.9172 (+0.0010) vs the ConvNeXt alone. Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/raptor_pair_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
MEMBERS = [("convnext_base_v5_raptor96_local",
            r"D:\My Projects\knee_abnormality_detection\runs\convnext_base_v5_raptor96_local\results\checkpoints\top_step9000.pt"),
           ("medimageinsight_davit_huge_v5_raptor96_sam",
            r"D:\My Projects\knee_abnormality_detection\runs\medimageinsight_davit_huge_v5_raptor96_sam\results\checkpoints\top_step5300.pt")]
BLEND_WEIGHTS = "raptor_pair_weights.json"
PUBLIC_SIDE = None
SLUG = "knee-raptor-pair"
NOTEBOOK_SLUG = "knee-raptor-pair-a"
OUT_DIR = Path("D:/temp1/raptor_pair")   # C: is full
NOTE = ("RAPTOR PAIR: convnext_base_v5_raptor96_local @9000 x0.7 + medimageinsight_davit_huge_v5_raptor96_sam @5300 "
        "x0.3 (logit), no public blend; v4 labels, v5 split, flipped.")
