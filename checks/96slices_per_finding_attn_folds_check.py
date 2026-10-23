"""96slices_per_finding_attn_folds_check.py - the 5 PerFindingAttn-96 folds (12 Oct): light, CPU, no decoding.

  splits    each fold's train and val are disjoint, val == 27_fold{k}_val (v3-graded copy), audit-58 in neither,
            the 5 vals partition every train+val study once
  recipe    input / model / optimizer / loss / scheduler / epochs identical to fold 0 and to
            convnext_base_v5_96slices_per_finding_attn (LB 0.939); no SAM, no extension
  update    batch x accum = 8 studies per update in every fold
  cadence   the validation interval per epoch

    python checks/96slices_per_finding_attn_folds_check.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

D = ROOT / "data"
ref = sc.load_config(ROOT / "runs/convnext_base_v5_96slices_per_finding_attn", verbose=False)
aud = set(pd.read_csv(D / "05_audit58.csv").StudyInstanceUID)
fails, seen = [], []
for k in range(5):
    run = f"convnext_base_v5_96slices_per_finding_attn_f{k}"
    c = sc.load_config(ROOT / "runs" / run, verbose=False)
    tr = set(pd.read_csv(D / c["paths"]["train_csv"]).StudyInstanceUID)
    va = set(pd.read_csv(D / c["paths"]["val_csv"]).StudyInstanceUID)
    t, r = c["training"], ref["training"]
    checks = {
        "train/val disjoint": not (tr & va),
        f"val == 27_fold{k}_val": va == set(pd.read_csv(D / f"27_fold{k}_val.csv").StudyInstanceUID),
        "audit in neither": not (aud & (tr | va)),
        "input + model == convnext_base_v5_96slices_per_finding_attn": c["input"] == ref["input"] and c["model"] == ref["model"],
        "optimizer/loss/scheduler/epochs == convnext_base_v5_96slices_per_finding_attn": all(t[x] == r[x] for x in ("optimizer", "loss", "scheduler", "epochs")),
        "no SAM, no extension": not (t.get("sam") or {}).get("enable") and not (t.get("extend") or {}).get("enable"),
        "8 studies per update": c["dataloader"]["batch_size"] * sc.accum_steps(c) == 8,
    }
    seen.append(va)
    bad = [n for n, ok in checks.items() if not ok]
    fails += [f"{run}: {b}" for b in bad]
    print(f"{run}: train {len(tr)} val {len(va)} | batch {c['dataloader']['batch_size']} x accum {sc.accum_steps(c)} | "
          f"eval every {[sc.eval_interval(c, e) for e in (1, 6, 7, 10, 16)]} @ epochs 1/6/7/10/16 | "
          f"{'ok' if not bad else 'FAIL ' + ', '.join(bad)}")
allv = set().union(*seen)
part = sum(len(v) for v in seen) == len(allv)
print(f"5 vals partition {len(allv)} studies once: {'ok' if part else 'FAIL'}")
print("ALL OK" if not fails and part else "FAILED: " + "; ".join(fails))
