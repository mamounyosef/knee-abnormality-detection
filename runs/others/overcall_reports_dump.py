"""overcall_reports_dump.py - the audit-58 reports behind the Lateral OA and Synovitis over-calls (panel 0, v3 label >= 0.5),
plus the true positives (panel 1) for contrast, with every label source side by side. Writes a text file to read.

    python runs/others/overcall_reports_dump.py            # -> runs/others/logs/overcall_reports.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE.parent.parent / "data"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))   # noqa: E731
sg = lambda x: 1 / (1 + np.exp(-x))   # noqa: E731
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
u = list(aud.index)
q = pd.read_csv(D / "03_report_labels_qwen24t-med.csv").set_index("StudyInstanceUID").loc[u, T]
mg = pd.read_csv(D / "16_medgemma15_corpus_all.csv").set_index("StudyInstanceUID").loc[u, T]
rep = pd.read_csv(D / "02_reports_en.csv").set_index("StudyInstanceUID").reindex(u)
out = []
for t in ["Lateral OA", "Synovitis"]:
    v3 = sg(0.5 * lg(q[t].values) + 0.5 * lg(mg[t].values))
    df = pd.DataFrame({"panel": aud.loc[u, "official_" + t].astype(int).values, "v3": v3,
                       "qwen": q[t].values, "medgemma": mg[t].values}, index=u)
    for grp, mask in (("OVER-CALLED (panel 0, label >= 0.5)", (df.panel == 0) & (df.v3 >= 0.5)),
                      ("TRUE POSITIVES (panel 1)", df.panel == 1)):
        sub = df[mask].sort_values("v3", ascending=False)
        out.append(f"\n{'=' * 100}\n{t}: {grp}: {len(sub)} studies\n{'=' * 100}")
        for k, (s, r) in enumerate(sub.iterrows(), 1):
            out.append(f"\n[{t} {k}] {s[-12:]}  panel {r.panel}  v3 {r.v3:.2f}  qwen {r.qwen:.2f}  medgemma {r.medgemma:.2f}"
                       f"  lang {rep.loc[s, 'lang']}\n{rep.loc[s, 'Report_en']}")
p = HERE / "logs" / "overcall_reports.txt"
p.write_text("\n".join(out), encoding="utf-8")
print(p, len("\n".join(out)), "chars")
