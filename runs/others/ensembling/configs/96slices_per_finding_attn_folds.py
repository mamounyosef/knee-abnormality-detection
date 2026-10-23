# PERFINDINGATTN-96 FOLDS ALONE (13 Oct, user: submit): convnext_base_v5_96slices_per_finding_attn_f0..f4, flat logit mean, each at its best-val
# step 7000 / 6960 / 6250 / 6928 / 5250. Local: audit-58 0.9257 (fold mean), clean-121 0.9131, OOF-4342 0.9193
# (single convnext_base_v5_96slices_per_finding_attn 0.9272 / 0.9163, LB 0.939). One geometry group.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/96slices_per_finding_attn_folds_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\ens4.py")).read_text(encoding="utf-8"))
FOLD_STEPS = [7000, 6960, 6250, 6928, 5250]
MEMBERS = [(f"convnext_base_v5_96slices_per_finding_attn_f{k}", _R.format(f"convnext_base_v5_96slices_per_finding_attn_f{k}", f"top_step{s}.pt"))
           for k, s in enumerate(FOLD_STEPS)]
BLEND_WEIGHTS = None
SLUG = "knee-raptor-folds"
NOTEBOOK_SLUG = "knee-raptor-folds-a"
OUT_DIR = Path("D:/temp1/96slices_per_finding_attn_folds")
NOTE = ("PERFINDINGATTN-96 5 FOLDS: " + ", ".join(n for n, _ in MEMBERS) + ", flat logit mean, best-val steps "
        "7000/6960/6250/6928/5250, no public blend.")
