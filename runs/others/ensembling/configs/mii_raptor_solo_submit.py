# SUBMIT MII raptor96 SAM solo (11 Oct, user: submit it on its own). Stress (128 studies) passed verify_stress_output
# except "2 decode chunks", which does not apply (one geometry group = one pass, as the hidden run): 128/128 studies,
# 1/1 members, RAM peak 6.0 / 33.7 GB, 3.18 s/study gpu, projected 1.45 h on the hidden set. Engine ff4c6d97cfee.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/mii_raptor_solo_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\mii_raptor_solo.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
