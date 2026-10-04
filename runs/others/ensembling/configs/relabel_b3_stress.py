# STRESS commit run of configs/relabel_b3.py (128 public studies repeated, 4 decode chunks). Never submits.
#     python runs/others/ensembling/configs/make_relabel_b3_weights.py
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/relabel_b3_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b3.py")).read_text(encoding="utf-8"))
STRESS = 128
SUBMIT = False
WAIT_FOR_RUN = True
