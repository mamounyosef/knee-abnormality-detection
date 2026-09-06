"""
sweep_cross_attention.py — does mricore_vitb_cross_attention earn an ensemble slot?

It ties mricore_vitb_relabel solo (0.8962 vs 0.8970), so on its own number there is no
reason to submit it. The reason to sweep it anyway is that it is the only member that
fuses information ACROSS the 96 slices inside the trunk; every other member encodes each
slice independently and averages at the end. Ablating its three cross-slice blocks costs
0.0356 AUROC, so that mechanism is genuinely load-bearing — which is exactly the profile
that made dinov3_convnext_base worth a large slot despite losing solo.

The two questions this asks, and they are different:
  1. does C ADD to the winning five?
  2. does C REPLACE mricore_vitb_relabel? They share the MRI-CORE trunk, so they are the
     most correlated pair in the pool and the least likely to both earn a slot.

Also re-runs the two current submissions unchanged, so every number here is comparable
against them rather than against a remembered figure.

    python runs/others/ensembling/sweep_cross_attention.py
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
C = "mricore_vitb_cross_attention"          # the new member

COMBOS = [
    # --- references, so the new numbers are comparable to what was actually submitted ---
    ("5-submitted-0.924",      [S, B, M, R, D]),
    ("6-submitted-0.922",      [S, B, M, R, D, L]),

    # --- 1. does C simply ADD? ---
    ("6-five-plus-C",          [S, B, M, R, D, C]),
    ("7-six-plus-C",           [S, B, M, R, D, L, C]),

    # --- 2. does C REPLACE the member it shares a trunk with? ---
    ("5-C-for-mricore",        [S, B, C, R, D]),
    ("6-C-for-mricore-plus-L", [S, B, C, R, D, L]),

    # --- 3. can BOTH MRI-CORE variants earn a slot, or do they cannibalise? ---
    ("4-both-mricore-only",    [M, C, B, D]),
    ("3-mricore-pair-plus-B",  [M, C, B]),
    ("2-mricore-pair",         [M, C]),

    # --- 4. C against each single partner: where does its decorrelation actually pay? ---
    ("2-C-and-B",              [C, B]),
    ("2-C-and-D",              [C, D]),
    ("2-C-and-R",              [C, R]),
    ("2-C-and-S",              [C, S]),

    # --- 5. drop the weak members now that C is present ---
    ("5-drop-small",           [B, M, R, D, C]),
    ("5-drop-radim",           [S, B, M, D, C]),
    ("4-core-plus-C",          [B, M, D, C]),
    ("4-B-D-C-R",              [B, D, C, R]),

    # --- 6. C alone, as the floor ---
    ("1-C-alone",              [C]),
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
                env = dict(os.environ, PYTHONIOENCODING="utf-8")
                p = subprocess.run([sys.executable, str(SCRIPT)], env=env, cwd=str(HERE),
                                   capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                if p.returncode != 0:
                    print("FAILED %s/%s\n%s" % (tag, split,
                                                p.stdout[-1200:] + p.stderr[-1200:]))
                    continue
                m = re.findall(r"ENSEMBLE mean AUROC\s+([0-9.]+)", p.stdout)
                results[(tag, split)] = float(m[-1]) if m else float("nan")
                print("  %-26s %-6s %.4f" % (tag, split, results[(tag, split)]))
    finally:
        SCRIPT.write_text(original, encoding="utf-8")   # always restore the config

    print("\n%-26s %8s %8s   n  members" % ("combination", "val", "audit"))
    print("-" * 78)
    rows = sorted(COMBOS, key=lambda kv: -(results.get((kv[0], "val")) or 0))
    for tag, members in rows:
        print("%-26s %8.4f %8.4f  %2d  %s"
              % (tag, results.get((tag, "val"), float("nan")),
                 results.get((tag, "audit"), float("nan")), len(members),
                 " ".join(m.split("_")[0][:6] for m in members)))


if __name__ == "__main__":
    main()
