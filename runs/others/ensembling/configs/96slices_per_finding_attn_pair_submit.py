# SUBMIT the 96slices_per_finding_attn pair (11 Oct, user: yes). Stress (128 studies) passed verify_stress_output except "2 decode chunks",
# which does not apply (both members share one geometry = one pass, as the hidden run): 128/128 studies, 2/2 members,
# per-class blend weights used, RAM peak 6.3 / 33.7 GB, gpu 1.03 + 2.91 s/study, projected 1.49 h. Engine ff4c6d97cfee.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/96slices_per_finding_attn_pair_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\96slices_per_finding_attn_pair.py")).read_text(encoding="utf-8"))
STRESS = 0
SUBMIT = True
WAIT_FOR_RUN = True
