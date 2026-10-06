# COMBINED, SIZED FOR THE REAL TEST SET (5 Oct night): the earlier combined builds projected their runtime on 946 hidden
# studies and on 3 cached public studies repeated; the data page says ~1300 studies, and the combined submissions ran
# 8 h+. This one is planned to finish in ~7 h with a fallback that still fits:
#   - public side TRIMMED: RSNA_SKIP_COAT_READERS=global96,repairv1 (their slowest reader pair, ~30% of its time);
#     kept DINO, A5, Raptor, residual-gated + D4. Parallel reader pairs. Runs AFTER ours, in the time left.
#   - our side: the 6 cheapest high-value members (runs/others/ensemble_named_subsets.py "S6", 5.4 GPU s/study):
#       offline vs all 14: audit 0.9312 (+0.0007, 58%) | clean-121 0.9171 (+0.0015, 81%)
#     with convnext_base_25d_poolattn_full_v5 (full-data twin, user) in place of convnext_base_25d_poolattn_v5
#   - sequential (no overlap: the overlapped runs ran out of host RAM), fp32/amp as every submitted ensemble
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_fit_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2.py")).read_text(encoding="utf-8"))
_KEEP = ["convnext_base_v5", "dinov3_convnext_base_v5", "convnext_base_25d_18slices_v5_aug_base",
         "convnext_base_25d_poolattn_v5", "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip",
         "medimageinsight_davit_huge_336_v3_f0"]
assert all(any(n == k for n, _ in MEMBERS) for k in _KEEP), "a member to keep is not in batch 2"
MEMBERS = [(n, c) for n, c in MEMBERS if n in _KEEP]
MEMBERS = [(("convnext_base_25d_poolattn_full_v5", _R.format("convnext_base_25d_poolattn_full_v5", "top_step6600.pt"))
            if n == "convnext_base_25d_poolattn_v5" else (n, c)) for n, c in MEMBERS]
assert len(MEMBERS) == 6
# ORDER "last": ours runs FIRST (~1.4 h on 1300 studies) and writes a valid submission; the public side then gets
# min(7.5 h, 8.2 h - elapsed; killed 10 min later at worst = 8.4 h). If it finishes, the blend runs; if it is cut, ours stands. It cannot pass the 9 h cap.
PUBLIC_SIDE = dict(PUBLIC_SIDE, cap_h=7.5, order="last",
                   env={"RSNA_PARALLEL_COAT_READERS": "1", "RSNA_SKIP_COAT_READERS": "global96,repairv1"})
SLUG = "knee-combined-fit"
NOTEBOOK_SLUG = "knee-combined-fit-a"
OUT_DIR = Path("C:/temp1/relabel_b2")          # rebuilt in place (the failed 14-member staging, nothing references it)
NOTE = ("COMBINED (sized for ~1300 studies): 6 cheapest high-value members rank-blended 50/50 with the prize-safe "
        "public pipeline (no RadImageNet; Global96 and Repair-v1 readers skipped). Members: "
        + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
