"""duplicate_slot_stats.py - how often does plan_slots feed the SAME series into two slots (same-plane fallback)?

    python runs/others/crop_rois/duplicate_slot_stats.py
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, r"D:\My Projects\knee_abnormality_detection\runs")
import shared_code as sc
cfg = sc.load_config(Path(r"D:\My Projects\knee_abnormality_detection\runs\convnext_base_25d_v5_joint_zoom_030_6v30sl"), verbose=False)
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plan = sc.plan_slots(idx, cfg)
slots = cfg["input"]["slots"]
n = len(plan)
for a, b in (("ax_fluid", "ax_plain"), ("cor_fluid", "cor_plain"), ("sag_fluid", "sag_plain")):
    same = sum(1 for p in plan.values() if p.get(a) and p.get(a) == p.get(b))
    print(f"{a} and {b} SAME series in {same} of {n} studies ({same / n:.0%})")
exact = {sl: sum(1 for st, g in idx.groupby('study') if (g.slot == sl).any()) for sl in slots}
print("studies with a genuine series per slot:", exact)
