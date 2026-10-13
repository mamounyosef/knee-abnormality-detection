# RAPTOR FOLDS ALONE (13 Oct, user: submit): convnext_base_v5_raptor96_f0..f4, flat logit mean, each at its best-val
# step 7000 / 6960 / 6250 / 6928 / 5250. Local: audit-58 0.9257 (fold mean), clean-121 0.9131, OOF-4342 0.9193
# (single raptor96_local 0.9272 / 0.9163, LB 0.939). One geometry group.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/raptor_folds_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\raptor4.py")).read_text(encoding="utf-8"))
FOLD_STEPS = [7000, 6960, 6250, 6928, 5250]
MEMBERS = [(f"convnext_base_v5_raptor96_f{k}", _R.format(f"convnext_base_v5_raptor96_f{k}", f"top_step{s}.pt"))
           for k, s in enumerate(FOLD_STEPS)]
BLEND_WEIGHTS = None
SLUG = "knee-raptor-folds"
NOTEBOOK_SLUG = "knee-raptor-folds-a"
OUT_DIR = Path("D:/temp1/raptor_folds")
NOTE = ("RAPTOR96 5 FOLDS: " + ", ".join(n for n, _ in MEMBERS) + ", flat logit mean, best-val steps "
        "7000/6960/6250/6928/5250, no public blend.")
