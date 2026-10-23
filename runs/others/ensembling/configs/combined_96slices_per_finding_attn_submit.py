# Submit configs/combined_96slices_per_finding_attn.py from its PASSED stress stamp, no rebuild (11 Oct, user: submit both).
# Stress: 128 studies, every member in every chunk; ours 2.06 h + public projected 5.53 h into 6.14 h of room -> blends.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_96slices_per_finding_attn_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_96slices_per_finding_attn.py")).read_text(encoding="utf-8"))
SUBMIT_FROM_STAMP = "dd15fa1fbff1"
SUBMIT = True
