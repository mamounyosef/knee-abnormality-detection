"""qwen_vision_smoke.py - run the real pipeline on real studies, end to end, on this PC.

qwen_vision_check.py proves the weights load and the tensors are right. It does NOT prove
the run trains: it never touches the dataset, the optimizer, or the LP-FT freeze. Those are
where the subtle bugs live - a backbone that silently stays frozen, a head learning rate
that means something different for a ViT than a ResNet, or a trunk whose features are so
flat that only the head moves.

This does a few real optimizer steps on real shards and checks that:
  - the loader hands the model the geometry the config promises
  - the LP-FT gate really freezes the trunk in epoch 1 and really unfreezes it after
  - the loss actually falls
  - the trunk's update-to-weight ratio lands in the same band as a healthy run

Run:  python checks/qwen_vision_smoke.py
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUN = "qwen3_5_vision_tower_336"

sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(RUNS / RUN))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %s %-60s %s" % ("PASS" if ok else "FAIL", name, detail))


cfg = sc.load_config(RUNS / RUN, verbose=False)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("\n  device %s   run %s" % (dev, RUN))

print("\n1. real shards through the real dataset")
df = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
shards = Path(cfg["paths"]["shards_dir"])
have = [s for s in df["StudyInstanceUID"] if (shards / ("%s.npz" % s)).exists()]
# only a handful of shards live on this PC; the rest are on the Modal volume. Two studies
# are enough to exercise the loader, prepare_batch, the freeze gate and a few real steps.
ck("local shards available", len(have) >= 2, "%d of %d studies in this split" % (len(have), len(df)))
if len(have) < 2:
    raise SystemExit("not enough local shards for a smoke test")
sub = df[df["StudyInstanceUID"].isin(have[:4])].reset_index(drop=True)

idx = __import__("pandas").read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plan = sc.plan_slots(idx, cfg)
ds = sc.KneeShardDataset(sub, cfg, plan, train=True)
item = ds[0]
x = item[0] if isinstance(item, (tuple, list)) else item["image"]
ck("one study is (slots, n_slices, H, W)",
   tuple(x.shape) == (len(cfg["input"]["slots"]), cfg["input"]["n_slices"],
                      cfg["input"]["size"], cfg["input"]["size"]), str(tuple(x.shape)))
# the dataset emits uint8; prepare_batch does the normalisation on the GPU
ck("the dataset emits uint8, normalisation happens in prepare_batch",
   str(x.dtype) == "torch.uint8", "dtype %s  min %d max %d" % (x.dtype, x.min(), x.max()))

print("\n2. the model consumes exactly what prepare_batch produces")
tm = importlib.import_module("train")
model = tm.build_model(cfg).to(dev)
loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))
xraw, yb = batch[0], batch[1]
xb = sc.prepare_batch(xraw, cfg, dev)
yb = yb.to(dev)
ck("prepare_batch normalises to about [-1, 1] using Qwen's 0.5/0.5",
   float(xb.min()) < -0.5 and float(xb.max()) > 0.5,
   "min %.3f max %.3f mean %.3f" % (xb.min(), xb.max(), xb.mean()))
per = sc.images_per_study(cfg)
ck("prepare_batch emits (B*windows, 3, size, size)",
   tuple(xb.shape) == (2 * per, 3, cfg["input"]["size"], cfg["input"]["size"]),
   str(tuple(xb.shape)))
model.eval()
with torch.no_grad():
    out = model(xb)
ck("forward gives (B, 12)", tuple(out.shape) == (2, len(cfg["tasks"])), str(tuple(out.shape)))
ck("outputs finite and varying", bool(torch.isfinite(out).all()) and float(out.std()) > 1e-4,
   "std %.4f" % out.std())

print("\n3. the LP-FT gate: epoch 1 must freeze the trunk, epoch 2 must release it")
trunk = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
ck("trunk and head are both non-empty", len(trunk) > 100 and len(head) > 0,
   "%d trunk, %d head tensors" % (len(trunk), len(head)))
sc.set_lp_ft_phase(cfg, model, epoch=1)
ck("epoch 1 freezes every trunk tensor", not any(p.requires_grad for p in trunk))
ck("epoch 1 leaves the head trainable", all(p.requires_grad for p in head))
sc.set_lp_ft_phase(cfg, model, epoch=2)
ck("epoch 2 unfreezes the trunk", all(p.requires_grad for p in trunk))

print("\n4. real optimizer steps with THIS RUN'S optimizer, not a hand-rolled one")
model.train()
sc.set_lp_ft_phase(cfg, model, epoch=2)          # phase 2: everything trainable
opt = sc.build_optimizer(model, cfg)
lrs = sorted({g["lr"] for g in opt.param_groups})
print("      %d param groups, learning rates %.3e .. %.3e (layer_decay %s)"
      % (len(opt.param_groups), lrs[0], lrs[-1],
         cfg["training"]["optimizer"].get("layer_decay")))
ck("layer-wise decay produced a spread of learning rates", len(lrs) > 3, "%d distinct" % len(lrs))

lossf = torch.nn.BCEWithLogitsLoss()
watch = ("backbone.blocks.0.attn.qkv.weight", "backbone.blocks.11.attn.qkv.weight",
         "head.weight")
before = {n: p.detach().clone() for n, p in model.named_parameters() if n in watch}
losses = []
for step in range(6):
    bt = next(iter(loader))
    xb = sc.prepare_batch(bt[0], cfg, dev)
    yb = bt[1].to(dev).float()
    loss = lossf(model(xb), yb)
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
    opt.step()
    losses.append(float(loss.detach()))
    print("      step %d  loss %.4f  grad_norm %.3f" % (step, losses[-1], gn))

ck("the loss falls over 6 steps", losses[-1] < losses[0],
   "%.4f -> %.4f" % (losses[0], losses[-1]))
ck("no NaN or inf in the loss", all(np.isfinite(losses)))

print("\n5. update-to-weight ratios, the test that the trunk is neither frozen nor blown up")
print("      a healthy run sits near 1e-4.6 per step; the baseline convnext logs -4.57")
for n in watch:
    old_p = before[n]
    new_p = dict(model.named_parameters())[n].detach()
    rel = ((new_p - old_p).norm() / old_p.norm().clamp_min(1e-12)).item()
    per_step = rel / 6.0
    lg = np.log10(max(per_step, 1e-30))
    ck("%s moved" % n.replace("backbone.", ""), per_step > 1e-9,
       "log10(update/weight) per step = %.2f" % lg)
    if "blocks" in n:
        ck("  and its rate is sane, not frozen and not exploding" + " " * len(n),
           -9.0 < lg < -2.0, "%.2f" % lg)

print("\n" + "=" * 74)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    for f in FAIL:
        print("    FAILED: %s" % f)
    raise SystemExit(1)
