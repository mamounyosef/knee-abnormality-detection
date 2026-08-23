"""
build_fewshot.py - turn the chosen 29 studies into a compact few-shot block.

The block is NOT whole reports. Each example is only the cells where the report
actually answers the question and the labeler got it wrong, written as
"this wording -> the panel scored this". That is the teaching signal: a cell the
labeler already agrees with teaches nothing, and a SILENT cell teaches something
unlearnable from text.

Cost matters because the block is prepended to every one of ~4,259 calls:

    whole reports + all 12 labels   ~13.6k tokens
    evidence sentences + 12 labels   ~9.0k
    only the disagreements           ~1.6k   <- this

    python data_code/label_v2/build_fewshot.py

Writes fewshot_block.txt next to this file. 03_label_reports.py --fewshot reads it.
"""

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent

HEADER = """
CALIBRATION EXAMPLES - real reports from this dataset, with what the expert panel
actually scored. These are the cases where the report IS explicit and a careful reader
still gets it wrong, so read them as corrections to your instinct, not as a summary of
the rules above. "panel 0" means the panel scored the finding ABSENT despite the report
wording shown.
""".rstrip()

FOOTER = """
The pattern: the report MENTIONING a finding is not the same as the panel scoring it
present, and qualified amounts ("mild", "minimal", "small", "trace") plus
degenerative-context findings are the usual over-calls. These are corrections to
specific wordings - they are NOT an instruction to lower every probability. A finding
the report states plainly and without qualification should still score high.
""".rstrip()


def main():
    cells = pd.read_csv(OUT / "_cells.csv")
    picks = pd.read_csv(OUT / "fewshot.csv")
    rep = pd.read_csv(ROOT / "data/02_reports_en.csv").set_index("StudyInstanceUID")

    evidence = {}
    with open(ROOT / "data/03_report_labels_raw.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            evidence[d["text"]] = d.get("findings", {})

    # CONFIRMATIONS. Without these the block is 40 over-calls to 5 under-calls - the
    # labeler's own bias, and only 7 under-calls exist in all 58 - so the model reads it
    # as "say lower" rather than "apply this criterion". Measured: mean p fell
    # 0.422 -> 0.300 against a true positive rate of 0.399, macro AUROC 0.9001 -> 0.8805.
    # These show the other half: report explicit, panel agreed, keep it high.
    conf = cells[(cells.sid.isin(set(picks.StudyInstanceUID))) & (cells.state == "YES")
                 & (cells.y == 1) & (cells.err < 0.25)]
    conf = conf.sort_values("p", ascending=False).groupby("finding").head(2)

    lines, n_cells = [HEADER], 0
    for n, sid in enumerate(picks.StudyInstanceUID, 1):
        txt = rep.loc[sid, "Report_en"] if sid in rep.index else ""
        ev = evidence.get(txt, {})
        # only cells the report answers AND the labeler got materially wrong
        g = cells[(cells.sid == sid) & (cells.state != "SILENT") & (cells.err > 0.5)]
        if g.empty:
            continue
        parts = []
        for _, r in g.iterrows():
            quote = (ev.get(r.finding, {}) or {}).get("evidence", "") or ""
            quote = " ".join(str(quote).split())[:150]
            if not quote:
                continue
            parts.append(f'  report: "{quote}"\n'
                         f'     -> {r.finding}: panel {int(r.y)}  (a careless read gives '
                         f'{r.p:.2f})')
            n_cells += 1
        if parts:
            lines.append(f"\nExample {n}:")
            lines.extend(parts)
    lines.append("\nAnd the other direction - the report is explicit and the panel"
                 " AGREED, so these stay high:")
    n_conf = 0
    for _, r in conf.iterrows():
        txt = rep.loc[r.sid, "Report_en"] if r.sid in rep.index else ""
        quote = (evidence.get(txt, {}).get(r.finding, {}) or {}).get("evidence", "") or ""
        quote = " ".join(str(quote).split())[:150]
        if not quote:
            continue
        lines.append(f'  report: "{quote}"\n     -> {r.finding}: panel 1')
        n_conf += 1
    lines.append("\n" + FOOTER)

    block = "\n".join(lines)
    (OUT / "fewshot_block.txt").write_text(block, encoding="utf-8")
    print(f"{n_cells} corrections + {n_conf} confirmations from {len(picks)} studies")
    print(f"~{len(block) // 4:,} tokens  ->  {OUT / 'fewshot_block.txt'}")


if __name__ == "__main__":
    main()
