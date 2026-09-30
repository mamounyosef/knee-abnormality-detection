"""make_audit58_blind.py - the 58 audit reports WITHOUT any label, in a folder outside the repo,
for a blind labelling session (a separate Claude Code instance labels them by hand).

Writes <out>/reports.csv (StudyInstanceUID, lang, Report, Report_en), shuffled with a fixed seed
so row order carries no information, and checks that no label column leaked in.

    python data_code/make_audit58_blind.py [--out C:\\temp1\\audit58_labeling]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=r"C:\temp1\audit58_labeling")
a = ap.parse_args()
ids = pd.read_csv(ROOT / "data/05_audit58.csv", usecols=["StudyInstanceUID"]).StudyInstanceUID
rep = pd.read_csv(ROOT / "data/02_reports_en.csv").set_index("StudyInstanceUID")
blind = rep.loc[ids, ["lang", "Report", "Report_en"]].reset_index()
blind = blind.sample(frac=1, random_state=7).reset_index(drop=True)
assert len(blind) == 58 and blind.StudyInstanceUID.is_unique
assert list(blind.columns) == ["StudyInstanceUID", "lang", "Report", "Report_en"]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
blind.to_csv(out / "reports.csv", index=False)
print(f"wrote {out / 'reports.csv'}: {len(blind)} reports, columns {list(blind.columns)}; "
      f"empty English reports: {int(blind.Report_en.isna().sum())}")
