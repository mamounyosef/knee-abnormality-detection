# DIRECT SUBMISSION of configs/combined_public_fit.py WITHOUT a stress run - the user's explicit choice (5 Oct night:
# "submit it as a normal submission ... without tests"). Kaggle's own 3-study commit run still happens first and its
# output checks (our 6 members ran, public side ok, blend written) still gate the attempt.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_fit_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_fit.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
SKIP_STRESS_GATE = True
