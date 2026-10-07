"""make_round2_teachers.py - create the 15 ROUND-2 teacher fold runs (7 Oct).

Same 5 folds as round 1 (data/27_fold{k}_*: 21_fold{k} studies, v4 labels; data_code/27_make_round2_folds.py).
Three families, each copied from its round-1 Modal recipe with ONLY the labels (and for DINOv3 the weights) changed:
  convnext_base_25d_18slices_v4_flip_f{k}   <- convnext_base_25d_18slices_v3_flip_f4 (Modal H100 recipe)
  dinov3_convnext_base_25d_18slices_v4_f{k} <- the same recipe, timm weights convnext_base.dinov3_lvd1689m
  medimageinsight_davit_huge_336_v4_f{k}    <- medimageinsight_davit_huge_336_v3_f{k} (Modal H200)

    python runs/others/make_round2_teachers.py [convnext] [dinov3] [mii]   # default: all three
"""
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
HEAD = ("# {name}\n#\n# ROUND-2 TEACHER, fold {k} (7 Oct, user: yes). {what} Labels v3 -> v4 (data/27_fold{k}_*: the 21_fold{k}\n"
        "# studies with the 22_*-w50 labels every current member trains on). Its out-of-fold predictions, averaged with\n"
        "# the other two families, become the round-2 teacher (audit-58 stays clean: the 58 are in no fold).\n#\n"
        "# ---- the donor's notes follow ----\n")


def make(name, donor, k, what, extra=()):
    src, dst = R / donor, R / name
    dst.mkdir(exist_ok=False)
    for f in ("train.py", "config.yaml"):
        s = (src / f).read_text(encoding="utf-8")
        s = s.replace(donor, name)
        s = re.sub(r"21_fold\d_(train|val)\.csv", lambda m: f"27_fold{k}_{m.group(1)}.csv", s)
        if f == "config.yaml":
            for a, b in extra:
                assert s.count(a) == 1, (name, a)
                s = s.replace(a, b)
            s = HEAD.format(name=name, k=k, what=what) + s
        (dst / f).write_text(s, encoding="utf-8")
    print("created", name)


# per fold: the round-1 fold's OWN recipe and seed (folds 0-1 run on this PC with the local recipe of v3_flip_f0,
# seed 42 like round 1's f0-f2; folds 2-4 the Modal H100 recipe of their own v3_flip fold, seeds 42 / 1042 / 2042)
CNX = {0: "convnext_base_25d_18slices_v3_flip_f0", 1: "convnext_base_25d_18slices_v3_flip_f0",
       2: "convnext_base_25d_18slices_v3_flip_f2", 3: "convnext_base_25d_18slices_v3_flip_f3",
       4: "convnext_base_25d_18slices_v3_flip_f4"}
DINO = {0: "convnext_base_25d_18slices_v3_flip_f2", 1: "convnext_base_25d_18slices_v3_flip_f2",   # Modal recipe, seed 42
        2: "convnext_base_25d_18slices_v3_flip_f2", 3: "convnext_base_25d_18slices_v3_flip_f3",
        4: "convnext_base_25d_18slices_v3_flip_f4"}
FAMS = set(sys.argv[1:]) or {"convnext", "dinov3", "mii"}
for k in range(5):
    if "convnext" in FAMS:
        make(f"convnext_base_25d_18slices_v4_flip_f{k}", CNX[k], k, f"Copy of {CNX[k]} ({'LOCAL' if k < 2 else 'Modal H100'} recipe).")
    if "dinov3" in FAMS:
        make(f"dinov3_convnext_base_25d_18slices_v4_f{k}", DINO[k], k,
             f"Copy of {DINO[k]} with the DINOv3 ConvNeXt-Base weights (as dinov3_convnext_base_v5, but 2.5D at 336).",
             extra=[('  name: "convnext_base.fb_in22k_ft_in1k"   # timm id: ConvNeXt-Base pretrained on ImageNet-22k',
                     '  name: "convnext_base.dinov3_lvd1689m"   # DINOv3 ConvNeXt-Base (ungated), the dinov3_convnext_base_v5 weights')])
    if "mii" in FAMS:
        make(f"medimageinsight_davit_huge_336_v4_f{k}", f"medimageinsight_davit_huge_336_v3_f{k}", k,
             f"Copy of medimageinsight_davit_huge_336_v3_f{k}.")
