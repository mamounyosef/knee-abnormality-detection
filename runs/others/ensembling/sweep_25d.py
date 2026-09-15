"""
sweep_25d.py - where do the two non-overlapping 2.5D members belong?

Two runs put three consecutive slices in the RGB channels instead of copying one slice
three times, so the pretrained stem is untouched and a study costs far fewer passes:

    convnext_base_25d_group3    24 slices, 32 passes/study, val 0.9013, audit 0.9005
    convnext_base_25d_18slices  18 slices, 24 passes/study, val 0.9010, audit 0.9054

Individually each already replaces convnext_large in the default six (0.9164 -> 0.9167 and
0.9166). The open questions this answers:

  1. Do they ADD to each other, or are they near-duplicates? They share a layout and differ
     only in slice count, so this is closer to two seeds than to two architectures.
  2. If they are duplicates, does fusing them into ONE vote beat giving them two?
  3. Is convnext_large now dead weight, or does it still earn a slot alongside them?
  4. Can the expensive swinv2_base_in22k (4.44 s/study on a T4, the priciest member by far)
     be dropped once these cheap members are in?

Same mechanism as sweep_large.py: patch ensemble.py's CONFIG, run it, restore. Every
combination leaves a real ensemble_results/ entry; nothing here reimplements scoring.

    python runs/others/ensembling/sweep_25d.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "ensemble.py"

# the parent and its three task-group branches average into ONE vote
G = ["convnext_base_relabel", "convnext_base_trauma",
     "convnext_base_wear", "convnext_base_fluid"]
MC = "mricore_vitb_cross_attention"
MF = "mricore_vitb_mae_cross_attention_full"
DV = "dinov3_convnext_base"
SW = "swinv2_base_in22k"
L = "convnext_large"
A24 = "convnext_base_25d_group3"       # 32 passes/study
A18 = "convnext_base_25d_18slices"     # 24 passes/study
P336 = "convnext_base_336px"
DV2 = "dinov2_vits14_25d"
SA = "convnext_base_25d_slotattn"   # label-query attention head
TA = "convnext_base_25d_tokenattn"  # token-to-token attention head
PA = "convnext_base_25d_poolattn"   # learned weighting replacing the mean
SP = "convnext_base_25d_spatial"    # 2400 spatial tokens + blank masking
CN = "coatnet_rmlp2_384"            # conv+attention hybrid, val 0.8974 / audit 0.9087

# SHIPPED is submission 56258238: the 9-model branch group at val 0.9167, measured at
# 15.202 s/study = 6.33 h of the 9 h cap. Every CoAtNet question below is asked against it,
# because "does it help" only means anything relative to what we would otherwise send.
SHIPPED = [G, MC, MF, DV, A24, PA]

COMBOS = [
    ("shipped-0.9167",        SHIPPED),                            # the baseline to beat
    # --- does CoAtNet ADD? ------------------------------------------------------------
    ("shipped+coatnet",       [G, MC, MF, DV, A24, PA, CN]),
    ("bg24-plus-coatnet",     [G, MC, MF, DV, A24, SW, CN]),       # onto the 0.9167 w/ swin
    # --- does it REPLACE anything? ----------------------------------------------------
    # CoAtNet's val is BELOW every member it might replace, so if it still wins a slot the
    # gain is decorrelation, not strength - the same pattern poolattn showed.
    ("coatnet-for-pool",      [G, MC, MF, DV, A24, CN]),
    ("coatnet-for-24sl",      [G, MC, MF, DV, CN, PA]),
    ("coatnet-for-dinov3",    [G, MC, MF, A24, PA, CN]),
    ("coatnet-for-mricore",   [G, MC, DV, A24, PA, CN]),
    ("coatnet-for-group",     [MC, MF, DV, A24, PA, CN]),          # drop the 4-model group
    # --- is it a near-duplicate of the other 2.5D members? ----------------------------
    ("shipped+coatnet+18sl",  [G, MC, MF, DV, A24, PA, CN, A18]),
    ("coatnet-fused-with-24", [G, MC, MF, DV, [A24, CN], PA]),     # ONE shared vote
    # --- everything, to see whether the blend is simply saturated ---------------------
    ("shipped+coatnet+spat",  [G, MC, MF, DV, A24, PA, CN, SP]),
    ("kitchen-sink",          [G, MC, MF, DV, A24, PA, CN, A18, SP, SW]),
]
SPLITS = ["val", "audit"]


def lit(m):
    """Render a member as a Python literal: a list stays a nested sub-ensemble."""
    if isinstance(m, list):
        return "[" + ", ".join('"%s"' % x for x in m) + "]"
    return '"%s"' % m


def patch(text, members, tag, split):
    # match to a "]" at the START of a line: a plain non-greedy [.*?] would stop at the
    # first closing bracket, which is a NESTED sub-ensemble's, and truncate the list
    body = "".join("    %s,\n" % lit(m) for m in members)
    text = re.sub(r"FULL_MODELS = \[.*?\n\]", "FULL_MODELS = [\n" + body + "]",
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
                                                (p.stdout + p.stderr)[-1200:]))
                    continue
                m = re.findall(r"ENSEMBLE mean AUROC\s+([0-9.]+)", p.stdout)
                results[(tag, split)] = float(m[-1]) if m else float("nan")
                print("  %-22s %-6s %.4f" % (tag, split, results[(tag, split)]))
    finally:
        SCRIPT.write_text(original, encoding="utf-8")   # always restore the config

    print("\n%-22s %8s %8s  %s" % ("combination", "val", "audit", "votes"))
    print("-" * 58)
    rows = [(t, results.get((t, "val"), float("nan")),
             results.get((t, "audit"), float("nan")), len(ms))
            for t, ms in COMBOS]
    for t, v, a, n in rows:
        print("%-22s %8.4f %8.4f  %d" % (t, v, a, n))
    best = max((r for r in rows if r[1] == r[1]), key=lambda r: r[1], default=None)
    if best:
        print("\nbest on val: %s (%.4f)" % (best[0], best[1]))


if __name__ == "__main__":
    main()
