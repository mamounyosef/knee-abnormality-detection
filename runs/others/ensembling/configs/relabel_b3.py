# RELABEL BATCH 3 (user, 4 Oct): the batch-2 fourteen (ref 56824782) with
#     swinv2_base_in22k                                            -> swinv2_base_in22k_v5_warm      (top_step2000; audit 0.9300 -> 0.9305)
#     convnext_base_25d_18slices_v5_aug_base                       -> convnext_base_25d_full_v5      (top_step6600: its full-data twin, 12 epochs)
#     convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip -> convnext_base_25d_perfinding_v5 (PERFINDING_CKPT below, when the local run ends)
# plus three specialists that vote ONLY on their own findings (runs/others/ensemble_candidates_v5.py, positive on both surfaces):
#     convnext_base_25d_v5_spec_trauma  (Fracture, Contusion)          weight 1   top_step1000
#     convnext_base_25d_v5_spec_pfoa    (PF OA)                        weight 2   top_step1000
#     convnext_base_25d_v5_spec_fluid   (Effusion, Synovitis, Baker's) weight 1   top_step1500
#   specialists alone: audit 0.9305 -> 0.9309 (+0.0005, 70%), clean 346 0.9326 -> 0.9329 (+0.0003, 85%).
# plus ONE old-label member for diversity (runs/others/old_label_members_ensemble.py: the relabelled members echo the
# MedImageInsight teacher): mricore_vitb_cross_attention (old Qwen labels, best.pt = step 5040) on all 12 findings,
#   alone: audit +0.0010 (72%), clean-121 +0.0007 (80%).
# 18 members. Per-finding weights: relabel_b3_weights.json (make_relabel_b3_weights.py writes it from MEMBERS).
# Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/relabel_b3_stress.py
PERFINDING_CKPT = "top_step3000.pt"   # best val 0.9132 @ step 3000 (early stop at 3600); swap: audit -0.0006, clean-121 +0.0008 (84%)
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b2.py")).read_text(encoding="utf-8"))
assert PERFINDING_CKPT, "set PERFINDING_CKPT (convnext_base_25d_perfinding_v5 best step) before building batch 3"
_NEW3 = {"swinv2_base_in22k": ("swinv2_base_in22k_v5_warm", "top_step2000.pt"),
         "convnext_base_25d_18slices_v5_aug_base": ("convnext_base_25d_full_v5", "top_step6600.pt"),
         "convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip": ("convnext_base_25d_perfinding_v5", PERFINDING_CKPT)}
assert all(any(n == old for n, _ in MEMBERS) for old in _NEW3), "a member to swap is not in the batch-2 fourteen"
MEMBERS = [((_NEW3[n][0], _R.format(*_NEW3[n])) if n in _NEW3 else (n, c)) for n, c in MEMBERS]
SPEC_W = {"convnext_base_25d_v5_spec_trauma": ("top_step1000.pt", 1),
          "convnext_base_25d_v5_spec_pfoa": ("top_step1000.pt", 2),
          "convnext_base_25d_v5_spec_fluid": ("top_step1500.pt", 1)}
MEMBERS += [(n, _R.format(n, f)) for n, (f, _) in SPEC_W.items()]
MEMBERS += [("mricore_vitb_cross_attention", _R.format("mricore_vitb_cross_attention", "best.pt"))]   # old labels, diversity
assert len(MEMBERS) == 18
BLEND_WEIGHTS = "relabel_b3_weights.json"
SLUG = "knee-relabel-b3"
OUT_DIR = Path("C:/temp1/relabel_b3")
NOTE = ("RELABEL BATCH 3: batch-2 fourteen with SwinV2 v5, full-data 2.5D v5, per-finding v5, plus trauma / PF OA / fluid "
        "specialists voting on their own findings, plus the old-label MRI-CORE cross-attention. Members: " + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
