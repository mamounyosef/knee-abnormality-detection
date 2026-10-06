# COMBINED FIT-10 (6 Oct): fit-9 (configs/combined_public_fit9.py) + convnext_base_25d_v5_joint_zoom_030 (top_step1500,
# the best val joint-findings step) voting ONLY on the 5 joint findings (ACL, Medial/Lateral Meniscus, Medial/Lateral
# OA): weight 1 there, 0 elsewhere (make_fit10_weights.py -> fit10_weights.json); every other member weight 1.
# Offline (runs/others/zoom_ensemble_test.py): z030 on the joint findings +0.0020 audit (97%) / +0.0009 clean-121 (94%)
# on the 0.943 core, +0.0003 / +0.0006 on all 14. Cost ~0.1 h on ~1300 studies (0.6x a 2.5D member + its localizer).
# Same order and public side as fit-9: ours first, public capped to what is left of 8.2 h.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_fit10_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_fit9.py")).read_text(encoding="utf-8"))
MEMBERS = MEMBERS + [("convnext_base_25d_v5_joint_zoom_030",
                      _R.format("convnext_base_25d_v5_joint_zoom_030", "top_step1500.pt"))]
assert len(MEMBERS) == 10
BLEND_WEIGHTS = "fit10_weights.json"
# pure fp16 inference (user, 6 Oct): fp16_accuracy_check.py on 4 members: audit AUROC equal to 4 decimals, max logit
# change 0.0011; T4 1.08-1.4x faster. The zoom's CAM is computed in float32 after the half-precision trunk.
INFER_ENV = {"KNEE_INFER_HALF": "1"}
SLUG = "knee-combined-fit10"
NOTEBOOK_SLUG = "knee-combined-fit10-a"
OUT_DIR = Path("C:/temp1/combined_fit9")          # rebuilt in place (fit-9's staging; its dataset is already uploaded)
NOTE = ("COMBINED FIT-10 (fp16): fit-9 + the 0.30 mm joint-zoom model voting only on ACL, menisci, medial/lateral OA, "
        "rank-blended 50/50 with the prize-safe public pipeline (trimmed), ours first. Members: "
        + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
