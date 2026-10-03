# RELABEL BATCH 1 swapped into the 0.942 fourteen (ref 56769791). Four members replaced by their relabelled versions
# (v4 labels, v5 split, flipped, best-val step); everything else unchanged. Flat logit mean, 14 members.
#     dinov3_convnext_base                                  -> dinov3_convnext_base_v5           (top_step5800)
#     convnext_base_full_all_knees_same_orientation         -> convnext_base_v5                  (top_step3700)
#     convnext_base_25d_all_knees_same_orientation_fulldata -> convnext_base_25d_18slices_v5_aug_base (top_step5900)
#     convnext_base_25d_poolattn_fulldata_flip              -> convnext_base_25d_poolattn_v5     (top_step3700: val -0.0004, audit +0.0025 vs its best-val 4000; user)
# Audit-58 offline (split analogues): 0.9268 -> 0.9289 (+0.0021, 79%; 0.9284 with poolattn 4000). Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/relabel_b1_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\ten_mii5folds.py")).read_text(encoding="utf-8"))
_R = r"D:\My Projects\knee_abnormality_detection\runs\{}\results\checkpoints\{}"
_NEW = {"dinov3_convnext_base": ("dinov3_convnext_base_v5", "top_step5800.pt"),
        "convnext_base_full_all_knees_same_orientation": ("convnext_base_v5", "top_step3700.pt"),
        "convnext_base_25d_all_knees_same_orientation_fulldata": ("convnext_base_25d_18slices_v5_aug_base", "top_step5900.pt"),
        "convnext_base_25d_poolattn_fulldata_flip": ("convnext_base_25d_poolattn_v5", "top_step3700.pt")}
assert all(any(n == old for n, _ in MEMBERS) for old in _NEW), "a member to swap is not in the fourteen"
MEMBERS = [((_NEW[n][0], _R.format(*_NEW[n])) if n in _NEW else (n, c)) for n, c in MEMBERS]
assert len(MEMBERS) == 14
SLUG = "knee-relabel-b1"
OUT_DIR = Path("C:/temp1/relabel_b1")
NOTE = ("0.942 FOURTEEN WITH RELABEL BATCH 1 (v4 labels, v5 split, flipped) SWAPPED IN. Members: "
        + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
