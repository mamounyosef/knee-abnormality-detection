# SUBMIT raptor4 (12 Oct, user: go ahead). Refused by submit_ensemble unless the stress run of this exact bundle passed.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/raptor4_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\raptor4.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
