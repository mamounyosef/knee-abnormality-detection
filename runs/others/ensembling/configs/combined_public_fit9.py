# COMBINED FIT-9 (6 Oct): the 0.943 build (configs/combined_public_fit.py, 6 members + trimmed public, ours first) with the
# ~2.5 h it left unused spent on 3 more of OUR members, picked by leave-one-out value (ensemble_member_value_vs_cost.log):
#   + swinv2_base_in22k_v5_warm (helps 86% / 79%), + mricore_vitb_cross_attention_v5 (61% / 89%),
#   + medimageinsight_davit_huge_336_v3_f4 (best of the spare MII folds on audit, 69%)
# Budget on ~1300 studies: ours ~13.1 GPU s/study -> ~3.2 h (set A stress 2.24 h x 1.42 real-data factor),
# trimmed public ~4.3 h (128-train-study timing minus the skipped Global96/Repair-v1 pair) -> public done ~7.7 h,
# under the 8.2 h public cutoff. If it is cut, the 9 members alone stand; the kernel ends by 8.4 h either way.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_fit9_submit.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2.py")).read_text(encoding="utf-8"))
_KEEP = ["convnext_base_v5", "dinov3_convnext_base_v5", "convnext_base_25d_18slices_v5_aug_base",
         "convnext_base_25d_poolattn_v5", "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip",
         "medimageinsight_davit_huge_336_v3_f0", "medimageinsight_davit_huge_336_v3_f4",
         "mricore_vitb_cross_attention_v5", "swinv2_base_in22k"]
assert all(any(n == k for n, _ in MEMBERS) for k in _KEEP), "a member to keep is not in batch 2"
MEMBERS = [(n, c) for n, c in MEMBERS if n in _KEEP]
_SWAP = {"convnext_base_25d_poolattn_v5": ("convnext_base_25d_poolattn_full_v5", "top_step6600.pt"),
         "swinv2_base_in22k": ("swinv2_base_in22k_v5_warm", "top_step2000.pt")}
MEMBERS = [((_SWAP[n][0], _R.format(*_SWAP[n])) if n in _SWAP else (n, c)) for n, c in MEMBERS]
assert len(MEMBERS) == 9
PUBLIC_SIDE = dict(PUBLIC_SIDE, cap_h=7.5, order="last",
                   env={"RSNA_PARALLEL_COAT_READERS": "1", "RSNA_SKIP_COAT_READERS": "global96,repairv1"})
SLUG = "knee-combined-fit9"
NOTEBOOK_SLUG = "knee-combined-fit9-a"
OUT_DIR = Path("C:/temp1/combined_fit9")
NOTE = ("COMBINED FIT-9: 9 members by leave-one-out value rank-blended 50/50 with the prize-safe public pipeline "
        "(no RadImageNet; Global96 and Repair-v1 readers skipped), ours first. Members: "
        + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
