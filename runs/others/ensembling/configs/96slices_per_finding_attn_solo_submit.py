# SUBMIT PerFindingAttn-96 solo (11 Oct, user: submit both). Stress (128 studies) passed the engine's checks by hand
# (verify_stress_output; the "2 decode chunks" rule does not apply: one geometry group = one pass, as the hidden run):
# 128/128 studies, 1/1 members, RAM peak 4.7 / 33.7 GB, projected 0.78 h on the hidden set.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/96slices_per_finding_attn_solo_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\96slices_per_finding_attn_solo.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
