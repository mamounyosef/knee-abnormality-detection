# The 5 MedImageInsight v3-label folds as one flat ensemble, each at its best-val step.
# Audit-58 of the 5-fold mean: 0.9252 (split parent alone 0.9124, ConvNeXt folds 0.9090 -> LB 0.918).
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/mii_v3_folds.py
_CK = r"D:\My Projects\knee_abnormality_detection\runs\medimageinsight_davit_huge_336_v3_f{}\results\checkpoints\top_step{}.pt"
MEMBERS = [(f"medimageinsight_davit_huge_336_v3_f{k}", _CK.format(k, s))
           for k, s in enumerate([2700, 3500, 3400, 2700, 3200])]
BLEND_WEIGHTS = None
SUBMIT = True
WAIT_FOR_RUN = True
SLUG = "knee-mii-336-v3-5folds"
OUT_DIR = Path(r"C:\temp1\mii_folds_kaggle")
NOTE = ("MedImageInsight DaViT-Huge 336, 5-fold cross-validation on v3 labels, flat mean of the "
        "5 fold models at their best-val steps (2700/3500/3400/2700/3200). Members: "
        + ", ".join(f"medimageinsight_davit_huge_336_v3_f{k}" for k in range(5))
        + ". Audit-58 of the fold mean 0.9252.")
