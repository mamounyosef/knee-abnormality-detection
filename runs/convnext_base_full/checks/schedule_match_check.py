"""Does convnext_base_full's lr curve actually match convnext_base_relabel's?

Matching means: at the same FRACTION of training done, both runs sit at the same multiple
of the base lr. The two have different step counts (454 vs 550 per epoch), so comparing
step-for-step is wrong; comparing epoch-for-epoch is the thing.
"""
import sys
from pathlib import Path

import torch

REPO = Path("d:/My Projects/knee_abnormality_detection")
sys.path.insert(0, str(REPO / "runs"))
import shared_code as sc  # noqa: E402

PARENT, FULL = "convnext_base_relabel", "convnext_base_full"
SPE = {PARENT: 454, FULL: 550}          # steps per epoch, from 3638/8 and 4400/8


def curve(run):
    cfg = sc.load_config(REPO / "runs" / run, verbose=False)
    ep = cfg["training"]["epochs"]
    total = SPE[run] * ep
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = sc.build_scheduler(opt, cfg, total)
    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return cfg, ep, total, lrs


ca, epa, ta, la = curve(PARENT)
cb, epb, tb, lb = curve(FULL)
print(f"{PARENT}: epochs {epa}, {ta} steps, schedule_epochs "
      f"{ca['training']['scheduler'].get('schedule_epochs')}")
print(f"{FULL}:   epochs {epb}, {tb} steps, schedule_epochs "
      f"{cb['training']['scheduler'].get('schedule_epochs')}")
print()
print("%-8s %14s %14s %10s" % ("epoch", PARENT[:14], FULL[:14], "diff"))
print("-" * 50)
worst = 0.0
for e in range(epb + 1):
    ia, ib = min(e * SPE[PARENT], ta - 1), min(e * SPE[FULL], tb - 1)
    d = abs(la[ia] - lb[ib])
    worst = max(worst, d)
    print("%-8d %14.6f %14.6f %10.2e" % (e, la[ia], lb[ib], d))
print("-" * 50)
print(f"worst mismatch at an epoch boundary: {worst:.2e}")
# 1e-3, not 1e-4: 454 and 550 steps per epoch do not land on identical fractions of their
# cosines, so a few 1e-4 of drift is integer rounding, not a schedule mismatch.
print("PASS - the curves match" if worst < 1e-3 else "FAIL - curves differ")

# control: without the knob the two would visibly diverge, or this test proves nothing
cb["training"]["scheduler"].pop("schedule_epochs")
p = torch.nn.Parameter(torch.zeros(1))
opt = torch.optim.SGD([p], lr=1.0)
sched = sc.build_scheduler(opt, cb, tb)
no_knob = []
for _ in range(tb):
    no_knob.append(opt.param_groups[0]["lr"])
    opt.step()
    sched.step()
gap = max(abs(la[min(e * SPE[PARENT], ta - 1)] - no_knob[min(e * SPE[FULL], tb - 1)])
          for e in range(epb + 1))
print(f"\nsanity: WITHOUT schedule_epochs the worst gap is {gap:.4f} "
      f"({'detected, so the knob is doing the work' if gap > 0.05 else 'NOT detected'})")
print(f"        final lr would be {no_knob[-1]:.6f} instead of {lb[-1]:.6f}")
