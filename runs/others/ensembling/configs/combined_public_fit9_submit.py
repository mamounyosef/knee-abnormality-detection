# DIRECT SUBMISSION of configs/combined_public_fit9.py without a stress run (same members as the stressed set A, same
# public notebook and order as the 0.943 fit build). Kaggle's 3-study commit run and its output checks still gate it.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_fit9_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_fit9.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
SKIP_STRESS_GATE = True
