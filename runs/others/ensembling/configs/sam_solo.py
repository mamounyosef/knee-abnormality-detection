# SAM SOLO (12 Oct, user: submit to test SAM on the LB): convnext_base_25d_18slices_v5_sam at step 5900 = the same step and
# 12-epoch schedule as its no-SAM twin convnext_base_25d_18slices_v5_aug_base's best (solo LB 0.930). Locally at 5900:
# val 0.9185 vs 0.9174, audit 0.9276 vs 0.9209. Decides SAM for the 96slices_per_finding_attn folds. Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/sam_solo_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
MEMBERS = [("convnext_base_25d_18slices_v5_sam",
            r"D:\My Projects\knee_abnormality_detection\runs\convnext_base_25d_18slices_v5_sam\results\checkpoints\top_step5900.pt")]
BLEND_WEIGHTS = None
PUBLIC_SIDE = None
SLUG = "knee-sam-solo"
NOTEBOOK_SLUG = "knee-sam-solo-a"
OUT_DIR = Path("D:/temp1/sam_solo")   # C: is full
NOTE = ("SAM SOLO: convnext_base_25d_18slices_v5_sam step 5900 (SAM twin of convnext_base_25d_18slices_v5_aug_base, LB 0.930; "
        "same step and schedule).")
