# COMBINED, SET A (user, 5 Oct): the 9 members chosen by value, not cost (runs/others/ensemble_member_value_vs_cost.py,
# ensemble_named_subsets.py): batch 2 minus MII-480, MII folds f2-f4 and MRI-CORE MAE; SwinV2 kept as its relabelled
# twin swinv2_base_in22k_v5_warm (top_step2000, solo LB 0.934), the version the offline scaffold scored.
#     offline vs all 14: audit 0.9311 (+0.0006, 62%) | clean-121 0.9176 (+0.0020, 95%), 13.1 GPU s/study
# Same proven runtime settings as configs/combined_public_b2cut.py (sequential, parallel CoAt readers, cap 5.5 h).
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_setA_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2.py")).read_text(encoding="utf-8"))
_DROP = ["medimageinsight_davit_huge_perslice_480_warm", "medimageinsight_davit_huge_336_v3_f2",
         "medimageinsight_davit_huge_336_v3_f3", "medimageinsight_davit_huge_336_v3_f4",
         "mricore_vitb_mae_cross_attention_v5"]
assert all(any(n == d for n, _ in MEMBERS) for d in _DROP), "a member to drop is not in batch 2"
MEMBERS = [(n, c) for n, c in MEMBERS if n not in _DROP]
assert any(n == "swinv2_base_in22k" for n, _ in MEMBERS)
MEMBERS = [(("swinv2_base_in22k_v5_warm", _R.format("swinv2_base_in22k_v5_warm", "top_step2000.pt"))
            if n == "swinv2_base_in22k" else (n, c)) for n, c in MEMBERS]
assert len(MEMBERS) == 9
PUBLIC_SIDE = dict(PUBLIC_SIDE, cap_h=5.5, env={"RSNA_PARALLEL_COAT_READERS": "1"})
SLUG = "knee-combined-seta"
OUT_DIR = Path("C:/temp1/combined_setA")
NOTE = ("COMBINED SET A: 9 members chosen by leave-one-out value (no MII-480, MII folds f0-f1, no MRI-CORE MAE, SwinV2 v5) "
        "rank-blended 50/50 with the prize-safe public pipeline (Stage 3 RadImageNet removed). Members: "
        + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
