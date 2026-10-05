# Submit configs/combined_public_setA.py from its PASSED stress stamp, no rebuild.
# Stress (5 Oct): 9/9 members, public side ok 128/128 in 36.1 min, blend dry run ok, projected 6.93 h on 946
# (ours 2.24 h + public 4.44 h, sequential).
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_setA_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_setA.py")).read_text(encoding="utf-8"))
SUBMIT_FROM_STAMP = "8db425358c6c"
SUBMIT = True
PUBLIC_W_SUBMIT = None
SUBMIT_NOTEBOOK_SLUG = "knee-seta-submit-b"   # set A kernel kept erroring at start (5 Oct); same notebook, fresh kernel
