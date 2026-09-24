"""run_name_check.py - every run folder's .py files must name THAT folder, never a parent.

Each train.py / evaluate.py / submit.py / plot_results.py sets EXP_NAME and loads
runs/<EXP_NAME>/config.yaml. A copied run that keeps its parent's EXP_NAME silently trains the
PARENT (its labels, its results folder). It happened on 15 Sep (five folds) and again on
24 Sep (convnext_base_25d_18slices_mglabels ran as its parent). Run this after creating or
copying any run.

    python checks/run_name_check.py [run ...]      # default: every run folder
"""
import re
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"
names = sys.argv[1:] or sorted(d.name for d in RUNS.iterdir() if (d / "config.yaml").exists())
bad = 0
for n in names:
    for py in sorted((RUNS / n).glob("*.py")):
        m = re.search(r'^EXP_NAME\s*=\s*"([^"]+)"', py.read_text(encoding="utf-8"), re.M)
        if m and m.group(1) != n:
            print(f"  FAIL  runs/{n}/{py.name}: EXP_NAME = {m.group(1)!r}")
            bad += 1
    cfg = (RUNS / n / "config.yaml").read_text(encoding="utf-8")
    m = re.search(r'^experiment:\s*\n\s+name:\s*"([^"]+)"', cfg, re.M)
    if m and m.group(1) != n:
        print(f"  FAIL  runs/{n}/config.yaml: experiment.name = {m.group(1)!r}")
        bad += 1
print(f"{len(names)} run folders checked, {bad} wrong name(s)" + ("" if bad else "  ALL OK"))
sys.exit(1 if bad else 0)
