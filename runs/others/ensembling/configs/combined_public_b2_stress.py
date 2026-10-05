# STRESS commit run of configs/combined_public_b2.py: our side 128 public studies repeated (4 decode chunks), the public
# side 128 fake study IDs over the public studies. Never submits.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_b2_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2.py")).read_text(encoding="utf-8"))
STRESS = 128
SUBMIT = False
WAIT_FOR_RUN = True
