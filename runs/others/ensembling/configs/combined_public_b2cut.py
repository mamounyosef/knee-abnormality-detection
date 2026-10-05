# COMBINED, TIME-TRIMMED: configs/combined_public_b2.py failed its stress time gate (10.52 h projected: ours 18.9 s/study +
# public 5.30 h). Our side loses the costliest members (runs/others/ensemble_named_subsets.py, vs all 14 = audit 0.9305 /
# clean-121 0.9156):
#     drop swinv2_base_in22k (5.1 GPU s/study) + medimageinsight_davit_huge_perslice_480_warm (4.7)
#          + 3 of the 5 MII 336 folds (f2, f3, f4; 0.7 each)        -> 9 members, 9.8 GPU s/study
#          audit 0.9293 (-0.0012) | clean-121 0.9164 (+0.0007)
# Public side measured at ~4.8 h on 946 (fixed cost ~4 min); cap 5.5 h so a slow hidden run still leaves ours its time.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_b2cut_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2.py")).read_text(encoding="utf-8"))
_DROP = ["swinv2_base_in22k", "medimageinsight_davit_huge_perslice_480_warm",
         "medimageinsight_davit_huge_336_v3_f2", "medimageinsight_davit_huge_336_v3_f3", "medimageinsight_davit_huge_336_v3_f4"]
assert all(any(n == d for n, _ in MEMBERS) for d in _DROP), "a member to drop is not in batch 2"
MEMBERS = [(n, c) for n, c in MEMBERS if n not in _DROP]
assert len(MEMBERS) == 9
# parallel CoAt reader pairs (their own switch; default off above 48 studies): 37.6 vs 43.3 min at 128, raw outputs equal
# within 2e-4 (public_side/compare_runs.py) -> ~4.3 h on 946
PUBLIC_SIDE = dict(PUBLIC_SIDE, cap_h=5.5, env={"RSNA_PARALLEL_COAT_READERS": "1"})
SLUG = "knee-combined-b2cut"
NOTE = ("COMBINED: batch 2 trimmed to 9 members (no SwinV2, no MII-480, MII folds f0-f1) rank-blended 50/50 with the "
        "prize-safe public pipeline (Stage 3 RadImageNet removed). Members: " + ", ".join(n for n, _ in MEMBERS)
        + ". | Built with DINOv3")
