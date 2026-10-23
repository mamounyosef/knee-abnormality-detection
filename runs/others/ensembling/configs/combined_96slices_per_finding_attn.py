# THE 0.943 COMBINED FIT (ref 56861632: our 6 + trimmed public pipeline, ours first) WITH convnext_base_v5_96slices_per_finding_attn
# (step 9000) IN PLACE OF convnext_base_25d_18slices_v5_aug_base (11 Oct, user). Offline, our part (S6 analogue):
# audit 0.9312 -> 0.9324 (+0.0012, 75%), clean-121 0.9189 -> 0.9202 (+0.0013, 86%). PerFindingAttn-96 decodes in its own
# geometry group (96 slices, 384 px at 0.365 mm), ~+20 min; the public side keeps the time left as before. Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_96slices_per_finding_attn_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_fit.py")).read_text(encoding="utf-8"))
_OLD_M, _NEW_M = "convnext_base_25d_18slices_v5_aug_base", "convnext_base_v5_96slices_per_finding_attn"
assert any(n == _OLD_M for n, _ in MEMBERS)
MEMBERS = [((_NEW_M, r"D:\My Projects\knee_abnormality_detection\runs\convnext_base_v5_96slices_per_finding_attn\results\checkpoints\top_step9000.pt")
            if n == _OLD_M else (n, c)) for n, c in MEMBERS]
assert len(MEMBERS) == 6 and any(n == _NEW_M for n, _ in MEMBERS)
SLUG = "knee-combined-raptor"
NOTEBOOK_SLUG = "knee-combined-raptor-a"
OUT_DIR = Path("C:/temp1/combined_96slices_per_finding_attn")
NOTE = ("COMBINED (0.943 recipe, PerFindingAttn-96 swapped in for aug_base): 6 members rank-blended 50/50 with the prize-safe "
        "public pipeline (no RadImageNet; Global96 and Repair-v1 readers skipped). Members: "
        + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
