# COMBINED: relabel batch 2 (our best measured, LB 0.942, 14 members, flat logit mean) + the prize-safe public pipeline
# (yamadan96/rsna-knee-d4-public0946 code, 0.943 with RadImageNet; Stage 3 RadImageNet + its calibrator removed, see
# runs/others/ensembling/public_side/make_public_pipeline.py). The public side runs first in its own kernel (capped),
# then our ensemble, then per finding: (1 - w) x rank(ours) + w x rank(public). Ours alone if the public side fails.
# Their models trained on all 4,407 studies and were selected on audit-58, so only the leaderboard can score this.
# Stress first (both sides at 128 studies; the public side on fake study IDs pointing at the public studies):
#     python runs/others/ensembling/public_side/make_public_pipeline.py
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_b2_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b2.py")).read_text(encoding="utf-8"))
PUBLIC_SIDE = {
    "notebook": r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\public_side\public_pipeline.ipynb",
    "weight": 0.5,
    "cap_h": 3.0,           # their own claim is under 30 min on 2 x T4; ours needs ~4.5 h of the 9
    "datasets": ["dreaddevelopment/raptor-knee-maxspan", "dreaddevelopment/raptor-knee-native384",
                 "dreaddevelopment/raptor-knee-native384dense", "mattiaangeli/knee-mri-fold-weights",
                 "mattiaangeli/opencv-python-headless-4120088-x86", "mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
                 "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2", "mattiaangeli/rsna-knee-coatnet-global96-top3",
                 "pilkwang/rsna-knee-llm-labels", "pilkwang/rsna-knee-weights"],
    "models": ["metaresearch/dinov2/PyTorch/small/1"],
}
SLUG = "knee-combined-b2"
OUT_DIR = Path("C:/temp1/relabel_b2")       # rebuilt in place (the wipe frees the old staging first; C is tight)
NOTE = ("COMBINED: relabel batch 2 (LB 0.942) rank-blended 50/50 with the prize-safe public pipeline (Stage 3 "
        "RadImageNet removed). Members: " + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
