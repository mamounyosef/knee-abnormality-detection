# Submit the ten + 5 MII folds from its PASSED stress stamp (stress run 4, 1 Oct), without rebuilding.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/ten_mii5folds_submit.py
exec(open(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\ten_mii5folds.py",
          encoding="utf-8").read())
SUBMIT_FROM_STAMP = "63014019c018"
SUBMIT = True
