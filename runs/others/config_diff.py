"""config_diff.py - every field where two runs' FULLY MERGED configs (shared_config.yaml + the run's config.yaml)
differ, plus the values that matter for a new run whatever they are.

    python runs/others/config_diff.py <run A> <run B>
"""
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
import shared_code as sc  # noqa: E402


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


a, b = sys.argv[1], sys.argv[2]
A, B = flat(sc.load_config(R / a, verbose=False)), flat(sc.load_config(R / b, verbose=False))
print(f"DIFFERENCES  {a}  ->  {b}")
for k in sorted(set(A) | set(B)):
    if A.get(k, "<absent>") != B.get(k, "<absent>"):
        print(f"  {k:45s} {str(A.get(k, '<absent>'))[:45]:45s} -> {str(B.get(k, '<absent>'))[:60]}")
print(f"\nKEY VALUES of {b}")
for k in sorted(B):
    if any(s in k for s in ("laterality", "slots", "n_slices", "size", "mm_per_px", "window", "channels", "slices_as",
                            "norm_", "train_csv", "val_csv", "audit", "tasks", "amp", "bf16", "dtype", "compile",
                            "seed", "epochs", "batch", "accum", "lr", "gpu", "timeout", "early", "patience", "eval_every",
                            "augment", "flip", "label", "loss.name", "pos_weight", "series_index", "resume")):
        print(f"  {k:45s} {str(B[k])[:90]}")
