# Submit configs/combined_public_b2cut.py from its PASSED stress stamp, no rebuild.
# Stress (5 Oct): 9/9 members, public side ok 128/128 in 35.5 min (parallel CoAt readers), blend dry run ok,
# projected 6.15 h on 946 (public side 4.37 h). PUBLIC_W_SUBMIT None = the built weight (0.5).
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_b2cut_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2cut.py")).read_text(encoding="utf-8"))
SUBMIT_FROM_STAMP = "54cd84fd057c"
SUBMIT = True
PUBLIC_W_SUBMIT = None
