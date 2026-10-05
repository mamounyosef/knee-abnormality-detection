"""orthofoundation_run_checks.py - pre-launch checks for orthofoundation_vitl_cross_attention_v5 (an expensive run).

1. LATERALITY: for a LEFT knee, the study tensor with the run's flip ("left") must equal the unflipped one mirrored
   per plane: sagittal slots reverse the slice order, coronal/axial slots reverse the width. For a RIGHT knee the
   flip must change nothing. (eval mode, augmentation off, so only the flip differs)
2. TRAIN STEP: one forward+backward through the real model at a reduced 2 slices per slot (8 images), eager vs
   torch.compile: loss and gradient agreement, step time. bf16/fp16 autocast as the engine trains.

    python runs/others/orthofoundation_run_checks.py
"""
import copy
import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd
import torch

R = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ortho_train", R / "orthofoundation_vitl_cross_attention_v5" / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sc, cfg = mod.sc, mod.cfg

# ---------- 1. laterality ----------
lat = pd.read_csv(sc.resolve(cfg["paths"]["laterality_csv"], cfg))
print("laterality csv columns:", list(lat.columns)[:6])
side_col = next(c for c in lat.columns if c.lower() in ("laterality", "side", "knee_side"))
df = sc._eval_df(cfg, cfg["paths"]["train_csv"])
plan = sc.plan_slots(pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)), cfg)
uid_col = "StudyInstanceUID"
sides = dict(zip(lat[uid_col].astype(str), lat[side_col].astype(str).str.upper().str[0]))
ok_all = True
for want in ("L", "R"):
    uid = next(u for u in df[uid_col].astype(str) if sides.get(u) == want)
    row = df[df[uid_col].astype(str) == uid]
    c_on = copy.deepcopy(cfg)
    c_off = copy.deepcopy(cfg)
    c_off["input"]["laterality_flip"] = None
    x_on, _ = sc.KneeShardDataset(row, c_on, plan, False)[0]
    x_off, _ = sc.KneeShardDataset(row, c_off, plan, False)[0]
    for si, slot in enumerate(cfg["input"]["slots"]):
        a, b = x_on[si], x_off[si]                              # (D, H, W)
        if want == "R":
            good = torch.equal(a, b)
            rule = "unchanged"
        elif slot.startswith("sag"):
            good = torch.equal(a, b.flip(0))
            rule = "slice order reversed"
        else:
            good = torch.equal(a, b.flip(-1))
            rule = "width reversed"
        ok_all &= good
        print(f"  {want} knee {uid[-8:]} {slot:10s} expect {rule:22s} -> {'OK' if good else 'MISMATCH'}"
              f" (identical to unflipped: {torch.equal(a, b)})")
print("LATERALITY:", "PASS" if ok_all else "FAIL")

# ---------- 2. train step eager vs compile ----------
c2 = copy.deepcopy(cfg)
c2["input"]["n_slices"] = 2                              # 4 slots x 2 = 8 images: fits an 8 GB card with gradients
dev = torch.device("cuda")
torch.manual_seed(0)
base = mod.build_model(c2).to(dev).train()
x = torch.randint(0, 256, (1, 4, 2, 336, 336), dtype=torch.uint8)
y = torch.randint(0, 2, (1, 12)).float().to(dev)
lossf = torch.nn.BCEWithLogitsLoss()


def step(model):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)                                 # same dropout / drop-path draws in both
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = lossf(model(sc.prepare_batch(x, c2, dev)).float(), y)
    loss.backward()
    torch.cuda.synchronize()
    return loss.item(), time.perf_counter() - t


res = {}
for name in (("eager",) if "--no-compile" in sys.argv else ("eager", "compile")):
    m = base if name == "eager" else torch.compile(base)
    first = step(m)
    times = [step(m)[1] for _ in range(3)]
    g = torch.cat([p.grad.flatten().float() for p in base.parameters() if p.grad is not None])
    res[name] = (first[0], g.clone(), min(times), first[1])
    print(f"  {name:8s} loss {first[0]:.5f} | first call {first[1]:.1f} s | step {min(times):.3f} s"
          f" | peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
if "compile" not in res:
    sys.exit(0)
ge, gc = res["eager"][1], res["compile"][1]
cos = float(torch.nn.functional.cosine_similarity(ge, gc, dim=0))
print(f"  loss diff {abs(res['eager'][0] - res['compile'][0]):.2e} | grad cosine {cos:.5f}"
      f" | compile speed-up {res['eager'][2] / res['compile'][2]:.2f}x")
print("COMPILE:", "PASS (works, same gradients)" if cos > 0.999 else "CHECK (gradients differ)")
