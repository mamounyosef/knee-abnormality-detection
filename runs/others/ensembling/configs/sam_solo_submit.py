# SUBMIT the SAM solo (12 Oct, user: yes submit). Needs a passing STRESS run of this exact bundle first.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/sam_solo_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\sam_solo.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
