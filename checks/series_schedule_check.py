"""series_schedule_check.py - is convnext_base_25d_v5_duplicate_series its parent plus the series schedule, and does the schedule do what
it claims?

  1. loads its OWN config; differs from convnext_base_25d_18slices_v5_aug_base only by input.series_schedule
  2. plan_slot_candidates puts plan_slots' own pick first for EVERY study and slot (inference unchanged)
  3. scheduled_series: best series on odd epochs and the last 2; every duplicate shown equally often on even epochs
  4. the real train dataset follows the schedule: on epoch 1 its series equal the plan, on epoch 2 the studies with
     duplicates switch; the val dataset never switches; a switched study still loads a finite volume

    python checks/series_schedule_check.py
"""
import collections
import copy
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(RUNS))
RUN, PARENT = "convnext_base_25d_v5_duplicate_series", "convnext_base_25d_18slices_v5_aug_base"
bad = 0


def check(ok, msg):
    global bad
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'}  {msg}")


def load(run):
    spec = importlib.util.spec_from_file_location(run, RUNS / run / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
    return out


m, pm = load(RUN), load(PARENT)
cfg, sc = m.cfg, m.sc
print(RUN)
check(cfg["experiment"]["name"] == RUN and m.EXP_NAME == RUN, "1. loads its own config, EXP_NAME matches")
f, fp = flat(cfg), flat(pm.cfg)
diff = {k for k in set(f) | set(fp) if f.get(k) != fp.get(k)}
check(diff == {"experiment.name", "input.series_schedule.enable"}, f"1. differs from the parent only by the schedule: {sorted(diff)}")

idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plan, cands = sc.plan_slots(idx, cfg), sc.plan_slot_candidates(idx, cfg)
same = all(cands[s][sl][0] == plan[s][sl] for s in plan for sl in plan[s])
nd = sum(len(v) - 1 for c in cands.values() for v in c.values())
nst = sum(any(len(v) > 1 for v in c.values()) for c in cands.values())
check(same and set(plan) == set(cands), f"2. the first candidate is plan_slots' pick everywhere; {nd} duplicates in {nst} studies")

E = cfg["training"]["epochs"]
ok3 = True
for n in (2, 3, 4):
    c = [f"s{j}" for j in range(n)]
    seen = collections.Counter()
    for key in (f"k{q}" for q in range(60)):
        sched = [sc.scheduled_series(c, e, E, 2, key) for e in range(1, E + 1)]
        ok3 &= all(sched[e - 1] == "s0" for e in range(1, E + 1) if e % 2 == 1 or e > E - 2)
        seen.update(x for e, x in enumerate(sched, 1) if e % 2 == 0 and e <= E - 2)
    dup = [seen[f"s{j}"] for j in range(1, n)]
    ok3 &= seen["s0"] == 0 and max(dup) - min(dup) <= max(dup) * 0.25
    print(f"     {n} candidates, 60 study/slot keys: duplicate showings {dup}")
check(ok3, "3. best series on odd epochs and the last 2; duplicates share the even epochs evenly")

c2 = copy.deepcopy(cfg)
c2["dataloader"].update(num_workers=0, val_num_workers=0, persistent_workers=False, pin_memory=False)
L = sc.make_loaders(c2)
ds, vs = L["train"].dataset, L["val"].dataset


def series_of(dset, i, epoch):
    sc._EPOCH.value = epoch
    st = dset.df.StudyInstanceUID.iloc[i]
    ch = dset.plan.get(st, {})
    if dset.train and getattr(dset, "cands", None):
        ch = {sl: sc.scheduled_series(cc, epoch, E, 2, f"{st}|{sl}") for sl, cc in dset.cands.get(st, {}).items()}
    return ch


dups = [i for i, st in enumerate(ds.df.StudyInstanceUID) if any(len(v) > 1 for v in ds.cands.get(st, {}).values())]
e1 = all(series_of(ds, i, 1) == ds.plan[ds.df.StudyInstanceUID.iloc[i]] for i in dups[:300])
e2 = sum(series_of(ds, i, 2) != ds.plan[ds.df.StudyInstanceUID.iloc[i]] for i in dups)
check(e1 and e2 == len(dups), f"4. epoch 1 = the plan; epoch 2 switches all {len(dups)} train studies with duplicates")
check(not getattr(vs, "cands", None), "4. the val dataset has no schedule (always the best series)")
sc._EPOCH.value = 2
x = ds[dups[0]][0]
check(torch.isfinite(x.float()).all().item() and x.shape == ds[1][0].shape, f"4. a switched study loads a volume {tuple(x.shape)}")
sc._EPOCH.value = 0

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
