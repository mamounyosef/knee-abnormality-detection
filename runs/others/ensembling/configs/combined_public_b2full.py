# COMBINED, ALL 14: batch 2 (LB 0.942, all fourteen members) + the prize-safe public pipeline, made to fit by
#   - pure fp16 inference (KNEE_INFER_HALF=1, shared_code._HalfInfer): T4 1.08-1.4x per member (SwinV2 1.4x), audit-58
#     AUROC unchanged (runs/others/ensembling/fp16_accuracy_check.py)
#   - the public kernel OVERLAPPED with our ensemble (it starts first, in the background; the blend waits for it)
#   - parallel CoAt reader pairs on the public side
# Sequential 14 + public projected 10.52 h; this stress run measures whether overlap + fp16 bring it under 8 h.
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/combined_public_b2full_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\combined_public_b2.py")).read_text(encoding="utf-8"))
assert len(MEMBERS) == 14
# user, 5 Oct: convnext_base_25d_poolattn_full_v5 (full-data twin, 12 epochs, final step 6600) replaces its split twin
assert any(n == "convnext_base_25d_poolattn_v5" for n, _ in MEMBERS)
MEMBERS = [(("convnext_base_25d_poolattn_full_v5", _R.format("convnext_base_25d_poolattn_full_v5", "top_step6600.pt"))
            if n == "convnext_base_25d_poolattn_v5" else (n, c)) for n, c in MEMBERS]
PUBLIC_SIDE = dict(PUBLIC_SIDE, cap_h=7.5, overlap=True, env={"RSNA_PARALLEL_COAT_READERS": "1"})
INFER_ENV = {"KNEE_INFER_HALF": "1"}
SLUG = "knee-combined-b2full"
NOTE = ("COMBINED: batch 2, all 14 members (fp16 inference), rank-blended 50/50 with the prize-safe public pipeline "
        "(Stage 3 RadImageNet removed), public side overlapped. Members: " + ", ".join(n for n, _ in MEMBERS)
        + ". | Built with DINOv3")
NOTEBOOK_SLUG = "knee-b2full-overlap"   # the -submit kernel errors at start with empty logs after its earlier failure (5 Oct); fresh kernel
