# STRESS commit run of configs/ten_mii5folds.py: the public studies repeated to 128 (4 decode chunks of 32),
# so the offload / reload path between chunks actually runs. STRESS suppresses submission.csv and SUBMIT is
# off, so this can never spend a submission.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/ten_mii5folds_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\ten_mii5folds.py")).read_text(encoding="utf-8"))
STRESS = 128
SUBMIT = False
WAIT_FOR_RUN = True
