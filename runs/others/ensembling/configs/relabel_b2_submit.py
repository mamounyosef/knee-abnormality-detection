# Submit relabel batch 2 (the batch-1 fourteen with the two MRI-CORE v5 members) from its PASSED stress stamp, no rebuild.
# Stress: 128 studies, 14/14 members in every chunk, 37m54s (17.8 s/study, ~4.7 h on 946), RAM peak 16.3 / 33.7 GB.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/relabel_b2_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b2.py")).read_text(encoding="utf-8"))
SUBMIT_FROM_STAMP = "b4ae8ed14ec0"
SUBMIT = True
