"""
sweep_large.py — drive ensemble.py over a set of member combinations.

ensemble.py scores ONE combination per invocation and writes its own timestamped
folder under ensemble_results/. This patches its CONFIG block, runs it, and restores
the file, once per combination and once per split. Every combination therefore leaves
a real ensemble_results/ entry - nothing here reimplements the scoring.

Written for the ConvNeXt-Large question: Large came in at val 0.9061 / audit 0.9069
against Base's 0.9050 / 0.9130, so it is not worth submitting alone. A model that
weak solo can still earn a slot if its errors differ from the others', which is the
only thing this sweep is asking.

    python runs/others/ensembling/sweep_large.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "ensemble.py"

S = "convnext_small_relabel"
B = "convnext_base_relabel"
M = "mricore_vitb_relabel"
R = "radimagenet_resnet50_relabel"
L = "convnext_large"
D = "dinov3_convnext_base"
X = "radiodino_vitb"

COMBOS = [
    ("4-baseline",            [S, B, M, R]),        # the 0.919 submission, for reference
    ("5-dinov3",              [S, B, M, R, D]),     # the new submission
    ("5-large",               [S, B, M, R, L]),
    ("6-dinov3-large",        [S, B, M, R, D, L]),
    ("7-all",                 [S, B, M, R, D, L, X]),
    # does dinov3 REPLACE a member rather than join them?
    ("4-dinov3-for-small",    [D, B, M, R]),
    ("4-dinov3-for-base",     [S, D, M, R]),
    ("4-dinov3-for-radim",    [S, B, M, D]),
    ("4-dinov3-for-mricore",  [S, B, D, R]),
    # is a weak member now dead weight, with dinov3 present?
    ("4-drop-radim",          [S, B, M, D]),
    ("4-drop-small",          [B, M, R, D]),
    ("4-drop-mricore",        [S, B, R, D]),
    ("3-core",                [B, M, D]),
    # dinov3 + large together, without the weakest
    ("5-no-radim",            [S, B, M, D, L]),
]
SPLITS = ["val", "audit"]


def patch(text, members, tag, split):
    text = re.sub(r"FULL_MODELS = \[.*?\]",
                  "FULL_MODELS = [\n" + "".join('    "%s",\n' % m for m in members) + "]",
                  text, count=1, flags=re.S)
    text = re.sub(r'^SET             = "[a-z]+"',
                  'SET             = "%s"' % split, text, count=1, flags=re.M)
    text = re.sub(r'^RUN_TAG         = "[^"]*"',
                  'RUN_TAG         = "%s"' % tag, text, count=1, flags=re.M)
    return text


def main():
    original = SCRIPT.read_text(encoding="utf-8")
    results = {}
    try:
        for tag, members in COMBOS:
            for split in SPLITS:
                SCRIPT.write_text(patch(original, members, tag, split), encoding="utf-8")
                # the script prints box-drawing characters; Windows would decode the
                # pipe as cp1252 and blow up, so pin utf-8 and never fail on a stray byte
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                p = subprocess.run([sys.executable, str(SCRIPT)], env=env, cwd=str(HERE),
                                   capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                if p.returncode != 0:
                    print("FAILED %s/%s\n%s" % (tag, split, p.stdout[-1500:] + p.stderr[-1500:]))
                    continue
                m = re.findall(r"ENSEMBLE mean AUROC\s+([0-9.]+)", p.stdout)
                results[(tag, split)] = float(m[-1]) if m else float("nan")
                print("  %-24s %-6s %.4f" % (tag, split, results[(tag, split)]))
    finally:
        SCRIPT.write_text(original, encoding="utf-8")   # always restore the config

    print("\n%-24s %8s %8s  members" % ("combination", "val", "audit"))
    print("-" * 62)
    for tag, members in COMBOS:
        print("%-24s %8.4f %8.4f  %d" % (tag, results.get((tag, "val"), float("nan")),
                                         results.get((tag, "audit"), float("nan")),
                                         len(members)))


if __name__ == "__main__":
    main()
