"""compare_train_logs.py - per-epoch mean loss, gradient norm and lr of two or more runs.

    python runs/others/compare_train_logs.py run_a run_b [...]
"""
import sys
from pathlib import Path

import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
tabs = []
for r in sys.argv[1:]:
    t = pd.read_csv(RUNS / r / "results/train_log.csv")
    t = t[t.run_id == t.run_id.iloc[-1]].drop_duplicates("global_step", keep="last")
    g = t.groupby("epoch").agg(loss=("loss", "mean"), gnorm=("gnorm", "mean"),
                               gnorm_med=("gnorm", "median"), lr=("lr", "mean"), n=("loss", "size"))
    g.columns = pd.MultiIndex.from_product([[r[-30:]], g.columns])
    tabs.append(g)
pd.set_option("display.width", 250)
print(pd.concat(tabs, axis=1).round(4).to_string())
