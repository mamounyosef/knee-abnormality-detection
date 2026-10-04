# Submit relabel batch 3 (18 members) from its PASSED stress stamp, no rebuild.
# Stress #2 (after the partial-head-group engine fix): 128 studies, 18/18 members in every chunk, 40m14s (18.9 s/study wall,
# ~5.0 h on 946), RAM peak 16.1 / 33.7 GB. Stress #1 failed on "geometry group has no full-width member" (fixed).
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/relabel_b3_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b3.py")).read_text(encoding="utf-8"))
SUBMIT_FROM_STAMP = "f42133575a31"
SUBMIT = True
