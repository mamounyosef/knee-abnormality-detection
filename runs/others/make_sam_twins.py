"""make_sam_twins.py - SAM-optimiser twins (8 Oct): copy a run, change ONE thing, judge against the donor.

  medimageinsight_davit_huge_336_v3_f0_sam   donor medimageinsight_davit_huge_336_v3_f0 + training.sam rho 0.05
  convnext_base_25d_18slices_v5_sam_zscore   donor convnext_base_25d_18slices_v5_sam (WITHOUT its extension)
                                             + input.window "zscore" (foreground z-score)

    python runs/others/make_sam_twins.py
"""
import re
import shutil
from pathlib import Path

R = Path(__file__).resolve().parents[1]
SAM_BLOCK = ("  sam:\n    enable: true\n    rho: 0.05                       # as convnext_base_25d_18slices_v5_sam "
             "(audit +0.0068 vs its twin)\n    adaptive: false\n")

TWINS = {
    "medimageinsight_davit_huge_336_v3_f0_sam": (
        "medimageinsight_davit_huge_336_v3_f0",
        "SAM TWIN (8 Oct, user): the MII DaViT-Huge fold-0 member (in the 0.943 core; audit 0.9221) + the SAM optimiser\n"
        "# (rho 0.05), which gave its ConvNeXt twin audit +0.0068 / clean-121 +0.0029. ONE change. 2 passes per step:\n"
        "# ~2 h H200 (donor 58 min), ~$12.",
        "sam"),
    "convnext_base_25d_18slices_v5_sam_zscore": (
        "convnext_base_25d_18slices_v5_sam",
        "ZSCORE TWIN (8 Oct, user): convnext_base_25d_18slices_v5_sam (12 epochs: val 0.9185 / audit 0.9276) with\n"
        "# input.window \"zscore\" (foreground = pixels above the stack mean; (v - mean) / sd, [-3, 3] -> [0, 255]) instead of\n"
        "# the [1, 99] percentile window. On the untrained window test zscore cost nothing (val 0.9247 vs 0.9249, audit\n"
        "# 0.9229 vs 0.9209), so trained on it, it is a diversity candidate. SAM kept so the comparison is SAM vs SAM.\n"
        "# ~1.5 h H100, ~$8.",
        "zscore"),
}

for name, (donor, why, kind) in TWINS.items():
    d = R / name
    d.mkdir(exist_ok=True)
    for f in ("config.yaml", "train.py", "evaluate.py", "plot_results.py", "submit.py"):
        if (R / donor / f).exists():
            shutil.copy(R / donor / f, d / f)
    for f in ("train.py", "evaluate.py", "plot_results.py", "submit.py"):
        p = d / f
        if p.exists():
            s = p.read_text(encoding="utf-8").replace(donor, name)
            s = re.sub(r'KAGGLE_NAME = "[^"]*"', f'KAGGLE_NAME = "{name[:50]}"', s)
            p.write_text(s, encoding="utf-8")
    p = d / "config.yaml"
    s = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    s = f"# {name}\n#\n# {why}\n#\n# ---- the donor's notes follow ----\n" + s.split("\n", 1)[1]
    s = s.replace(f'name: "{donor}"', f'name: "{name}"')
    if kind == "sam":
        assert "\n  sam:" not in s
        s = re.sub(r"^training:\n", "training:\n" + SAM_BLOCK, s, count=1, flags=re.M)
    else:
        # drop the donor's extension (this twin trains its 12 epochs from scratch)
        s = re.sub(r"  extend:[^\n]*\n(    [^\n]*\n)+", "", s, count=1)
        assert "extend:" not in s.split("# ---- the donor's notes follow ----")[1] or "enable: true" not in s
        s = re.sub(r"^input:\n", 'input:\n  window: "zscore"                  # foreground z-score (8 Oct twin)\n', s,
                   count=1, flags=re.M)
    p.write_text(s, encoding="utf-8")
    print("made", name)
