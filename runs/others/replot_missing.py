"""replot_missing.py - draw plots for every run that has training/validation logs but no plots.

Runs each run's OWN plot_results.py (its config, the engine's make_plots) in a subprocess, so
one broken run cannot stop the rest. --all redraws every run with logs.

    python runs/others/replot_missing.py [--all]
"""
import subprocess
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent
redo_all = "--all" in sys.argv
todo = []
for d in sorted(p for p in RUNS.iterdir() if (p / "config.yaml").exists()):
    res = d / "results"
    if not ((res / "train_log.csv").exists() or (res / "val_log.csv").exists()):
        continue
    plots = res / "plots"
    if redo_all or not (plots.exists() and any(plots.iterdir())):
        todo.append(d)
print(f"{len(todo)} run(s) to plot", flush=True)
ok, bad = [], []
for d in todo:
    script = d / "plot_results.py"
    if not script.exists():
        bad.append((d.name, "no plot_results.py"))
        continue
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=RUNS.parent)
    n = len(list((d / "results/plots").glob("*"))) if (d / "results/plots").exists() else 0
    if r.returncode == 0 and n:
        ok.append(d.name)
        print(f"  OK    {d.name}: {n} plot file(s)", flush=True)
    else:
        err = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
        bad.append((d.name, err[0][:160]))
        print(f"  FAIL  {d.name}: {err[0][:160]}", flush=True)
print(f"\nplotted {len(ok)}, failed {len(bad)}")
for n, e in bad:
    print(f"  {n}: {e}")
