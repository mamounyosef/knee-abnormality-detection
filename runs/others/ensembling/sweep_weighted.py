"""
sweep_weighted.py — is a WEIGHTED blend actually better than a flat one, honestly measured?

Everything in the cross-attention sweep was flat: equal weight, zero fitted parameters,
nothing to overfit. Fitting weights can only help on the data you fit them on, so the
question is never "does the weighted number look better" (it always does) but "does it
survive being measured on studies the fit never saw".

Two numbers per combination, and the distance between them IS the overfitting:

  weighted_mean_auroc   weights fitted on all 704 val studies, then scored on those same
                        704. In-sample. Always flattering, never trustworthy. This is the
                        number that made base4 + radiodino look like 0.9150 when its
                        honest flat score was 0.9080.

  oof_mean_auroc        5-fold: weights fitted on 4 folds, scored on the held-out fifth,
                        repeated. An estimate of what the weights do on studies they were
                        not tuned against. This is the one to rank on.

Grid step is NOT the main lever. Overfitting scales with the number of free parameters:
per-class weights over M members and 12 findings is 12*M parameters fitted on 704 studies,
and several findings carry well under 100 positives. GRID_STEP 0.1 is already coarse
enough that halving it changes little; the parameter COUNT is what matters, which is why
the flat baseline is carried in every row below as the thing to beat.

    python runs/others/ensembling/sweep_weighted.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "ensemble.py"
RESULTS = HERE / "ensemble_results"

S = "convnext_small_relabel"
B = "convnext_base_relabel"
M = "mricore_vitb_relabel"
R = "radimagenet_resnet50_relabel"
L = "convnext_large"
D = "dinov3_convnext_base"
C = "mricore_vitb_cross_attention"
W = "swinv2_base_in22k"
Z = "convnext_base_seed1337"

# Does swinv2 earn a slot? It is our best SINGLE model (val 0.9053 / audit 0.9101) and the
# only hierarchical transformer in the pool, so it is the best decorrelation candidate left.
# Kept deliberately SHORT: every extra combination is another chance for the 704-study val
# to pick a winner by luck, and OOF has already misranked one submission.
COMBOS = [
    # RadImageNet is OUT of all of these - non-commercial licence, prize money is commercial.
    ("ref-SBCD",        [S, B, C, D]),          # the standing 0.928
    ("W-SBCDW",         [S, B, C, D, W]),       # add swinv2
    ("W-BCDW",          [B, C, D, W]),          # add it, drop convnext_small
    ("W-SBCDLW",        [S, B, C, D, L, W]),    # everything prize-eligible
    ("W-BCDLW",         [B, C, D, L, W]),
    ("W-swap-S",        [W, B, C, D]),          # swinv2 INSTEAD of convnext_small
    ("W-hier-seeds",    [S, [B, Z], C, D, W]),  # + the nested seed pair that gave +0.0011
]


def patch(text, members, tag, weights, grid_step):
    text = re.sub(r"FULL_MODELS = \[.*?\]",
                  "FULL_MODELS = [\n" + "".join('    "%s",\n' % m for m in members) + "]",
                  text, count=1, flags=re.S)
    text = re.sub(r'^SET             = "[a-z]+"', 'SET             = "val"',
                  text, count=1, flags=re.M)
    text = re.sub(r'^WEIGHTS         = "[^"]*"', 'WEIGHTS         = "%s"' % weights,
                  text, count=1, flags=re.M)
    text = re.sub(r'^RUN_TAG         = "[^"]*"', 'RUN_TAG         = "%s"' % tag,
                  text, count=1, flags=re.M)
    # rank a search run on the HELD-OUT number, not the in-sample one
    text = re.sub(r'^BEST_METRIC     = "[^"]*"', 'BEST_METRIC     = "oof_mean_auroc"',
                  text, count=1, flags=re.M)
    text = re.sub(r'^GRID_STEP   = [0-9.]+', 'GRID_STEP   = %s' % grid_step,
                  text, count=1, flags=re.M)
    return text


def newest_summary(tag):
    """The summary this invocation just wrote — newest folder carrying the tag."""
    cands = [p for p in RESULTS.glob("*%s*" % tag) if p.is_dir()]
    if not cands:
        return None
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    for name in ("weighted_val_summary.json", "ensemble_val_summary.json"):
        f = newest / name
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def run(members, tag, weights, grid_step=0.1):
    original = SCRIPT.read_text(encoding="utf-8")
    try:
        SCRIPT.write_text(patch(original, members, tag, weights, grid_step),
                          encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.run([sys.executable, str(SCRIPT)], env=env, cwd=str(HERE),
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    finally:
        SCRIPT.write_text(original, encoding="utf-8")
    if p.returncode != 0:
        print("FAILED %s (%s)\n%s" % (tag, weights,
                                      p.stdout[-1200:] + p.stderr[-1200:]))
        return None
    return newest_summary(tag), p.stdout


def main():
    rows = []
    for tag, members in COMBOS:
        flat_out = run(members, tag + "-flat", "flat")
        srch_out = run(members, tag + "-search", "search")
        flat = fl = None
        if flat_out:
            js, out = flat_out
            m = re.findall(r"ENSEMBLE mean AUROC\s+([0-9.]+)", out)
            fl = float(m[-1]) if m else None
        wt = oof = None
        if srch_out:
            js, out = srch_out
            if js:
                wt, oof = js.get("weighted_mean_auroc"), js.get("oof_mean_auroc")
            if wt is None:
                m = re.findall(r"weighted mean AUROC\s+([0-9.]+)", out, re.I)
                wt = float(m[-1]) if m else None
            if oof is None:
                m = re.findall(r"OOF mean AUROC\s+([0-9.]+)", out, re.I)
                oof = float(m[-1]) if m else None
        rows.append((tag, len(members), fl, wt, oof))
        print("  %-22s flat %s   weighted(in-sample) %s   OOF %s"
              % (tag, _f(fl), _f(wt), _f(oof)))

    print("\n%-22s %3s %9s %11s %9s %9s" %
          ("combination", "n", "flat", "in-sample", "OOF", "OOF-flat"))
    print("-" * 74)
    for tag, n, fl, wt, oof in sorted(rows, key=lambda r: -(r[4] or 0)):
        gain = (oof - fl) if (oof is not None and fl is not None) else None
        print("%-22s %3d %9s %11s %9s %9s"
              % (tag, n, _f(fl), _f(wt), _f(oof), _f(gain, sign=True)))
    print("-" * 74)
    print("in-sample minus OOF is the overfitting. OOF minus flat is the only real gain:")
    print("if it is <= 0, the weights are not paying for themselves and flat wins.")


def _f(v, sign=False):
    if v is None:
        return "    -"
    return ("%+.4f" if sign else "%.4f") % v


if __name__ == "__main__":
    main()
