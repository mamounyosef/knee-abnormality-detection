"""
analyze_labels.py - how good are the LLM labels, and how much of the gap is fixable?

Everything here is measured against the 58 studies the Society of Skeletal Radiology
panel labelled from the IMAGES. The labeler only ever sees the REPORT, so some
disagreement is irreducible: a finding the radiologist never wrote down cannot be
recovered from text at any prompt quality. This script separates the two.

For every (study, finding) cell it records:
    state  YES / NO / SILENT   - what the report says, by the multilingual keyword
                                 rules in audit_58_report_vs_label.py
    y      0/1                 - the panel's label
    p      0..1                - the LLM's probability

    python data_code/label_v2/analyze_labels.py

Writes _cells.csv (the raw table) and fewshot_25.csv (the selected examples).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

TASKS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
         "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# report wording that qualifies HOW MUCH of a finding there is
SMALL = ["mild", "minimal", "small", "trace", "slight", "some", "scant", "discrete", "little"]
LARGE = ["moderate", "large", "marked", "significant", "severe", "massive", "gross",
         "tense", "extensive"]
SEV_KEYS = {"Effusion": ["effusion", "joint fluid", "hydrops"],
            "Contusion": ["contusion", "bone bruise", "marrow edema", "marrow oedema",
                          "bone edema"]}


def _classify():
    """The YES/NO/SILENT rules, imported without running that script's body."""
    src = (ROOT / "data_code" / "audit_58_report_vs_label.py").read_text(encoding="utf-8")
    ns = {}
    exec(compile(src.split("tr = pd.read_csv(")[0], "aud", "exec"), ns)
    return ns["classify"]


def build_cells() -> pd.DataFrame:
    classify = _classify()
    llm = pd.read_csv(ROOT / "data/03_report_labels.csv").set_index("StudyInstanceUID")
    aud = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
    rep = pd.read_csv(ROOT / "data/02_reports_en.csv").set_index("StudyInstanceUID")
    ids = [i for i in aud.index if i in llm.index and i in rep.index]
    rows = []
    for sid in ids:
        en = rep.loc[sid, "Report_en"]
        txt = en if isinstance(en, str) else rep.loc[sid, "Report"]
        for t in TASKS:
            rows.append(dict(sid=sid, finding=t, state=classify(txt, t),
                             y=float(aud.loc[sid, f"official_{t}"]), p=float(llm.loc[sid, t])))
    d = pd.DataFrame(rows)
    d["err"] = (d.p - d.y).abs()
    d["fixable"] = np.where(d.state != "SILENT", d.err, 0.0)   # SILENT is not the prompt's fault
    return d


def matrices(d):
    P = d.pivot(index="sid", columns="finding", values="p")[TASKS]
    return (P.values.copy(), d.pivot(index="sid", columns="finding", values="y")[TASKS].values.copy(),
            d.pivot(index="sid", columns="finding", values="state")[TASKS].values, list(P.index))


def severity(sid, keys, rep):
    """'L' / 'S' / 'N' - how the report qualifies the amount, in sentences that mention it."""
    txt = str(rep.loc[sid, "Report_en"]) if sid in rep.index else ""
    import re
    seen = set()
    for s in re.split(r"[.\n;]", txt.lower()):
        if any(k in s for k in keys):
            if any(w in s for w in LARGE):
                seen.add("L")
            elif any(w in s for w in SMALL):
                seen.add("S")
    return "L" if "L" in seen else ("S" if "S" in seen else "N")


def pick_fewshot(d, k=25, coverage_bonus=2.0):
    """The k studies that teach the most.

    Ranked by FIXABLE error - a cell the report answers and the labeler still got wrong.
    A study the labeler already agrees with teaches nothing, and a SILENT cell teaches
    something unlearnable. The bonus forces at least one demonstrated error per finding,
    which pure ranking misses (it drops PF OA).
    """
    demo = {s: set(g[g.fixable > 0.5].finding) for s, g in d.groupby("sid")}
    fix = {s: g.fixable.sum() for s, g in d.groupby("sid")}
    sel, covered = [], set()
    for _ in range(k):
        best, bs = None, -1
        for s in fix:
            if s in sel:
                continue
            score = fix[s] + coverage_bonus * len(demo[s] - covered)
            if score > bs:
                best, bs = s, score
        sel.append(best)
        covered |= demo[best]
    return sel, covered


def main():
    d = build_cells()
    d.to_csv(OUT / "_cells.csv", index=False)
    P, Y, S, sids = matrices(d)
    rep = pd.read_csv(ROOT / "data/02_reports_en.csv").set_index("StudyInstanceUID")

    sc._banner("LABEL QUALITY vs THE 58-STUDY PANEL", "\U0001F9EA")
    base = sc.compute_metrics(Y, P, TASKS)["macro_auroc"]
    print(f"  current LLM labels : {base:.4f} macro AUROC")

    print("\n--- where the error lives ---")
    g = d.groupby("state").agg(cells=("err", "size"), mean_err=("err", "mean"),
                               total=("err", "sum"))
    g["share_%"] = (g.total / g.total.sum() * 100).round(1)
    print(g.round(3).to_string())

    # what a PERFECT prompt could reach: right on every cell the report answers,
    # unchanged where the report is silent
    ans = S != "SILENT"
    pa = P.copy()
    pa[ans] = Y[ans]
    print(f"\n  ceiling for a perfect prompt : "
          f"{sc.compute_metrics(Y, pa, TASKS)['macro_auroc']:.4f}")
    pb = P.copy()
    pb[~ans] = Y[~ans]
    print(f"  unreachable from text alone  : "
          f"{sc.compute_metrics(Y, pb, TASKS)['macro_auroc']:.4f}")

    print("\n--- per finding: current -> prompt ceiling ---")
    for i, t in enumerate(TASKS):
        a = sc.compute_metrics(Y[:, [i]], P[:, [i]], [t])["macro_auroc"]
        b = sc.compute_metrics(Y[:, [i]], pa[:, [i]], [t])["macro_auroc"]
        print(f"  {t:18s} {a:.3f} -> {b:.3f}   ({b - a:+.3f})")

    print("\n--- severity wording vs the panel (report states the finding) ---")
    for finding, keys in SEV_KEYS.items():
        sub = d[(d.finding == finding) & (d.state == "YES")].copy()
        sub["sev"] = [severity(s, keys, rep) for s in sub.sid]
        t = sub.groupby("sev").agg(n=("y", "size"), panel_pos_rate=("y", "mean"),
                                   llm_mean_p=("p", "mean"))
        print(f"\n  {finding}:")
        print(t.round(3).to_string())

    print("\n--- calibration: does its own confidence already flag the errors? ---")
    d["conf"] = (d.p - 0.5).abs() * 2
    d["correct"] = ((d.p > 0.5).astype(float) == d.y).astype(float)
    b = pd.cut(d.conf, [0, 0.2, 0.5, 0.8, 0.95, 1.0])
    print(d.groupby(b, observed=True).agg(cells=("correct", "size"),
                                          accuracy=("correct", "mean")).round(3).to_string())

    sel, covered = pick_fewshot(d)
    pd.Series(sel, name="StudyInstanceUID").to_csv(OUT / "fewshot_25.csv", index=False)
    sub = d[d.sid.isin(sel)]
    print(f"\n--- few-shot selection: {len(sel)} studies ---")
    print(f"  fixable error covered : {sub.fixable.sum() / d.fixable.sum() * 100:.1f}%")
    print(f"  findings demonstrated : {len(covered)}/12")
    print(f"  -> {OUT / 'fewshot_25.csv'}")


if __name__ == "__main__":
    main()
