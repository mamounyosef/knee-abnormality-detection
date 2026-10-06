# DIRECT SUBMISSION of configs/combined_public_fit10.py (user, 6 Oct: "add the zooming and submit it"). No stress run;
# Kaggle's 3-study commit run and its output checks still gate it (the zoom is a new kind of member: own geometry group).
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_fit10_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_fit10.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
SKIP_STRESS_GATE = True
