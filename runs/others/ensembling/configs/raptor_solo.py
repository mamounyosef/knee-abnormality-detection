# RAPTOR96 SOLO (11 Oct, user: submit it alone): convnext_base_v5_raptor96_local at its best-val step 9000 (val 0.9213,
# audit 0.9272, clean-121 0.9163, the best clean single). The public 0.949 input recipe on ConvNeXt-B: 5 slots / 96 slices,
# 94 overlapping 3-slice windows per study at inference, per-finding attention. Answers: does this recipe reach ~0.94+
# alone on the LB (ours top out at 0.930 single)? Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/raptor_solo_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
MEMBERS = [("convnext_base_v5_raptor96_local",
            r"D:\My Projects\knee_abnormality_detection\runs\convnext_base_v5_raptor96_local\results\checkpoints\top_step9000.pt")]
BLEND_WEIGHTS = None
PUBLIC_SIDE = None
SLUG = "knee-raptor96-solo"
NOTEBOOK_SLUG = "knee-raptor96-solo-a"
OUT_DIR = Path("C:/temp1/raptor96_solo")
NOTE = ("RAPTOR96 SOLO: convnext_base_v5_raptor96_local step 9000 (public 0.949 input recipe on ConvNeXt-B, "
        "96 slices / 94 windows, per-finding attention; v4 labels, v5 split, flipped).")
