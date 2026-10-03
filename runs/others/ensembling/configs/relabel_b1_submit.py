# Submit relabel batch 1 (the 0.942 fourteen with four relabelled members) from its PASSED stress stamp, no rebuild.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/relabel_b1_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
SUBMIT_FROM_STAMP = "6ccdad5d92bb"
SUBMIT = True
