"""config_diff.py - every key that differs between two runs' MERGED configs (shared + own).

    python checks/config_diff.py <run_a> <run_b>
"""
import importlib.util
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "runs"


def load(name):
    sys.path.insert(0, str(RUNS))
    spec = importlib.util.spec_from_file_location(f"t_{name}", RUNS / name / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    assert tm.cfg["experiment"]["name"] == name, (name, tm.cfg["experiment"]["name"])
    return tm.cfg


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


a, b = sys.argv[1], sys.argv[2]
fa, fb = flat(load(a)), flat(load(b))
diff = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
print(f"{a}  vs  {b}: {len(diff)} differing key(s)")
for k in diff:
    print(f"  {k}: {fa.get(k)!r} -> {fb.get(k)!r}")
