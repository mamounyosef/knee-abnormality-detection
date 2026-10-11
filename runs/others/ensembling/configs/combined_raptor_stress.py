# STRESS commit run of configs/combined_raptor.py (both sides at 128 studies, sequential). Never submits.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_raptor_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_raptor.py")).read_text(encoding="utf-8"))
STRESS = 128
SUBMIT = False
WAIT_FOR_RUN = True
