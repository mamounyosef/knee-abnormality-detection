# STRESS commit run of configs/combined_public_b2full.py (both sides at 128 studies, overlapped). Never submits.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_b2full_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2full.py")).read_text(encoding="utf-8"))
STRESS = 128
SUBMIT = False
WAIT_FOR_RUN = True
