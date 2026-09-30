"""make_mii_v3_folds.py - the 5 MedImageInsight fold runs, copies of medimageinsight_davit_huge_336
(solo LB 0.929) with ONLY the train/val CSVs changed to data/21_fold{k}_{train,val}.csv (v3 labels).

Every .py file gets its own EXP_NAME; the config gets a header and the two CSV paths; nothing
else is touched. Refuses to overwrite an existing fold folder.

    python data_code/make_mii_v3_folds.py
"""
import re
import shutil
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
DONOR = "medimageinsight_davit_huge_336"
for k in range(5):
    name = f"{DONOR}_v3_f{k}"
    d = RUNS / name
    assert not d.exists(), f"{d} exists, not overwriting"
    d.mkdir()
    for f in ("config.yaml", "train.py", "evaluate.py", "plot_results.py", "submit.py"):
        s = (RUNS / DONOR / f).read_text(encoding="utf-8")
        s = re.sub(rf"\b{DONOR}\b", name, s)
        (d / f).write_text(s, encoding="utf-8")
    p = d / "config.yaml"
    s = p.read_text(encoding="utf-8")
    header = (f"# {name}\n#\n# FOLD {k} of 5 for MedImageInsight OUT-OF-FOLD LABELS: {DONOR} (solo LB 0.929) with\n"
              f"# ONLY the data changed: trains on 21_fold{k}_train.csv and validates on 21_fold{k}_val.csv\n"
              f"# (the study split of 13_fold{k}, v3 labels = Qwen + MedGemma, data_code/21_make_v3_folds.py).\n"
              f"# Its val predictions at the best step are this fold's out-of-fold predictions; the 5 folds\n"
              f"# cover all 4,342 train+val studies once. Same H200 recipe, flip, schedule, early stopping.\n#\n"
              f"# ---- the donor's notes follow ----\n")
    s, n1 = re.subn(r'^  train_csv: "05_train_qwen24t-med\.csv".*$', f'  train_csv: "21_fold{k}_train.csv"', s, flags=re.M)
    s, n2 = re.subn(r'^  val_csv: +"05_val_qwen24t-med\.csv".*$', f'  val_csv:   "21_fold{k}_val.csv"', s, flags=re.M)
    assert n1 == 1 and n2 == 1, (name, n1, n2)
    p.write_text(header + s, encoding="utf-8")
    print("created", name)
