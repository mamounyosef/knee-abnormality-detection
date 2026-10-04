# RELABEL BATCH 2 swapped into the batch-1 fourteen (LB 0.942, ref 56805570). Two MRI-CORE members replaced by their
# relabelled versions (v4 labels, v5 split, flipped, best-val step); everything else unchanged. Flat logit mean, 14 members.
#     mricore_vitb_cross_attention                   -> mricore_vitb_cross_attention_v5      (top_step6480, val 0.9160)
#     mricore_vitb_mae_cross_attention_fulldata_flip -> mricore_vitb_mae_cross_attention_v5  (top_step3300, val 0.9074)
# Audit-58 offline (split analogues, runs/others/relabel_batch2_swap_audit.py): 0.9289 -> 0.9300 (+0.0010, 70%). Stress first:
#     python runs/others/ensembling/submit_ensemble.py runs/others/ensembling/configs/relabel_b2_stress.py
exec((Path(r"D:\My Projects\knee_abnormality_detection\runs\others\ensembling\configs\relabel_b1.py")).read_text(encoding="utf-8"))
_NEW2 = {"mricore_vitb_cross_attention": ("mricore_vitb_cross_attention_v5", "top_step6480.pt"),
         "mricore_vitb_mae_cross_attention_fulldata_flip": ("mricore_vitb_mae_cross_attention_v5", "top_step3300.pt")}
assert all(any(n == old for n, _ in MEMBERS) for old in _NEW2), "a member to swap is not in the batch-1 fourteen"
MEMBERS = [((_NEW2[n][0], _R.format(*_NEW2[n])) if n in _NEW2 else (n, c)) for n, c in MEMBERS]
assert len(MEMBERS) == 14
SLUG = "knee-relabel-b2"
OUT_DIR = Path("C:/temp1/relabel_b2")
NOTE = ("BATCH-1 FOURTEEN (0.942) WITH RELABEL BATCH 2 (MRI-CORE, v4 labels, v5 split, flipped) SWAPPED IN. Members: "
        + ", ".join(n for n, _ in MEMBERS) + ". | Built with DINOv3")
