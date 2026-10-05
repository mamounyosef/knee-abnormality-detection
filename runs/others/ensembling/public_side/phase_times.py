"""phase_times.py - per-stage wall time of a public-side run, from its diagnostics/phase_events.jsonl.

    python runs/others/ensembling/public_side/phase_times.py runs/others/ensembling/public_side/test_out/v1
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
ev = [json.loads(l) for l in (d / "diagnostics" / "phase_events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
start = {}
for e in ev:
    st, s, t = e.get("stage"), e.get("status"), float(e.get("elapsed_seconds", 0))
    if s == "START":
        start[st] = t
    elif st in start:
        print(f"{st:28s} {start[st] / 60:6.1f} -> {t / 60:6.1f} min   = {(t - start[st]) / 60:5.1f} min  {s}")
print(f"last event at {max(float(e.get('elapsed_seconds', 0)) for e in ev) / 60:.1f} min")
