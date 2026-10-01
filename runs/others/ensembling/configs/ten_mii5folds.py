# THE 0.938 TEN (original, ref 56489550) with medimageinsight_davit_huge_336_fulldata replaced by
# the 5 MedImageInsight v3-label folds (alone LB 0.936, ref 56747511). 14 members, flat logit mean.
# Runtime from the two kernels' timing.json: ~32-35 s/study serial, ~4.8-5.3 h on Kaggle's two T4s.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/ten_mii5folds.py
_CK = r"D:\My Projects\knee_abnormality_detection\runs\medimageinsight_davit_huge_336_v3_f{}\results\checkpoints\top_step{}.pt"
MEMBERS = [
    ("convnext_base_full_all_knees_same_orientation", "best"),
    ("mricore_vitb_cross_attention", "best"),
    ("mricore_vitb_mae_cross_attention_fulldata_flip", "best"),
    ("dinov3_convnext_base", "best"),
    ("convnext_base_25d_all_knees_same_orientation_fulldata", "best"),
    ("swinv2_base_in22k", "best"),
    ("convnext_base_25d_poolattn_fulldata_flip", "best"),
    ("convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip", "best"),
    ("medimageinsight_davit_huge_perslice_480_warm", "best"),
] + [(f"medimageinsight_davit_huge_336_v3_f{k}", _CK.format(k, s))
     for k, s in enumerate([2700, 3500, 3400, 2700, 3200])]
BLEND_WEIGHTS = None
SUBMIT = True
WAIT_FOR_RUN = True
SLUG = "knee-ten-mii-5folds"
OUT_DIR = Path(__file__).resolve().parent / "kaggle_ten_mii5"   # D: ~11.5 GB; C has too little room
NOTE = ("THE 0.938 TEN WITH THE FULL-DATA MEDIMAGEINSIGHT REPLACED BY ITS 5 V3-LABEL FOLDS "
        "(alone LB 0.936). Members: " + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
