# MII PERFINDINGATTN-96 SAM SOLO (11 Oct, user: submit it on its own): medimageinsight_davit_huge_v5_96slices_per_finding_attn_sam at its best-val
# step 5300 (val 0.9195, audit 0.9271, clean-121 0.9147). The PerFindingAttn-96 input recipe on MII DaViT-Huge with SAM; ties the
# PerFindingAttn-96 ConvNeXt (LB 0.939) on audit, ~4x its inference cost. Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/mii_96slices_per_finding_attn_solo_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
MEMBERS = [("medimageinsight_davit_huge_v5_96slices_per_finding_attn_sam",
            r"D:\My Projects\knee_abnormality_detection\runs\medimageinsight_davit_huge_v5_96slices_per_finding_attn_sam\results\checkpoints\top_step5300.pt")]
BLEND_WEIGHTS = None
PUBLIC_SIDE = None
SLUG = "knee-mii-raptor96-solo"
NOTEBOOK_SLUG = "knee-mii-raptor96-solo-a"
OUT_DIR = Path("D:/temp1/mii_96slices_per_finding_attn_solo")   # C: is full
NOTE = ("MII PERFINDINGATTN-96 SAM SOLO: medimageinsight_davit_huge_v5_96slices_per_finding_attn_sam step 5300 (PerFindingAttn-96 input recipe on MII "
        "DaViT-Huge + SAM, 96 slices / 94 windows, per-finding attention; v4 labels, v5 split, flipped).")
