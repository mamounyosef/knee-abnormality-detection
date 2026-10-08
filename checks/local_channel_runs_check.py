"""local_channel_runs_check.py - the three local runs of 8 Oct (multiwindow, slab_mmm, randconv) load their OWN configs
and data and build the input they claim. CPU only, local shards.

  A  each config: item (4 slots, 18 planes, 336, 336) uint8, the real model forward -> 12, local recipe (batch 2 x 4)
  B  multiwindow: the 3 channels of an image are ONE slice (corr > 0.8) under 3 windows: dark-window mean > today's >
     fluid-window mean; today's window channel equals the default window of that slice set
  C  slab_mmm: per pixel min <= mean <= max exactly; the bins use >= 80% of the stored slices (today 57-65%)
  D  laterality, both modes: a LEFT study's sagittal slot = its images in reverse order with the channel order KEPT;
     coronal/axial width-reversed; right study untouched
  E  randconv: absent and p 0 give identical training items (no rng drawn when off); p 1 changes the item, stays 0-255
  F  default 2.5D configs unchanged: aug_base's dataset has channel_mode "slices", grp == ctx == 1
  G  a tiny real run_experiment per mode trains and validates

    python checks/local_channel_runs_check.py
"""
import contextlib
import copy
import io
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.cuda.is_available = lambda: False
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails, n_all = [], [0]
RUNS = ["convnext_base_25d_v5_multiwindow", "convnext_base_25d_v5_slab_mmm", "convnext_base_25d_v5_randconv"]
have = {q.stem for q in (ROOT / "data/shards").glob("*.npz")}


def check(name, ok, detail=""):
    n_all[0] += 1
    print("  %-72s %s%s" % (name, "ok" if ok else "FAIL", "  " + str(detail) if detail else ""))
    if not ok:
        fails.append(name)


def cpu(cfg):
    c = copy.deepcopy(cfg)
    c["dataloader"].update(num_workers=0, val_num_workers=0, persistent_workers=False, prefetch_factor=None,
                           pin_memory=False)
    return c


def loaders(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return sc.make_loaders(cfg)


cfgs, L = {}, {}
print("A  configs, items, model")
for r in RUNS:
    c = cpu(sc.load_config(ROOT / "runs" / r, verbose=False))
    cfgs[r], L[r] = c, loaders(c)
    va = L[r]["val"].dataset
    x = va[0][0]
    d = c["dataloader"]
    check(f"A {r.split('_v5_')[1]}: item (4, 18, 336, 336) uint8; batch 2 x accum 4; grad ckpt",
          tuple(x.shape) == (4, 18, 336, 336) and x.dtype == torch.uint8 and d["batch_size"] == 2
          and sc.accum_steps(c) == 4 and c["training"].get("grad_checkpointing"), tuple(x.shape))
    sys.path.insert(0, str(ROOT / "runs" / r))
    import importlib
    tr = importlib.import_module("train")
    c2 = copy.deepcopy(c); c2["model"]["pretrained"] = False
    with torch.no_grad():
        o = tr.build_model(c2).eval()(sc.prepare_batch(x[None], c2, torch.device("cpu")))
    check(f"A {r.split('_v5_')[1]}: real model forward -> (1, 12)", tuple(o.shape) == (1, 12))
    sys.path.remove(str(ROOT / "runs" / r)); del sys.modules["train"]

mw = L["convnext_base_25d_v5_multiwindow"]["val"].dataset
sl = L["convnext_base_25d_v5_slab_mmm"]["val"].dataset
ids = [s for s in mw.df.StudyInstanceUID if s in have][:30]
print("B  multiwindow")
cors, order, eq = [], [], []
for st in ids:
    ser = mw.plan[st].get("sag_fluid")
    if not ser:
        continue
    arr = mw._read_series(st, ser)
    v = mw._load(st, ser, np.random.default_rng(0), 18).reshape(6, 3, 336, 336)
    cors.append(np.mean([np.corrcoef(v[g, 0].ravel(), v[g, 1].ravel())[0, 1] for g in range(6)]))
    order.append(v[:, 0].mean() > v[:, 1].mean() > v[:, 2].mean())
    idx = sc.pick_slices(arr.shape[0], 6, "even", None, mw.margin)
    ref = mw._resample_fit(arr[idx].astype(np.float32), ser)
    lo, hi = np.percentile(ref, 1.0), np.percentile(ref, 99.0)
    eq.append(np.allclose(v[:, 1], np.clip((ref - lo) / (hi - lo), 0, 1) * 255.0, atol=1e-3))
check("B channels are one slice (mean corr > 0.8)", np.nanmean(cors) > 0.8, f"{np.nanmean(cors):.3f}")
check("B dark-window mean > today's > fluid-window mean", np.mean(order) > 0.9, f"{np.mean(order):.0%} of studies")
check("B middle channel == today's [1, 99] window of the 6 picked slices", all(eq), f"{sum(eq)}/{len(eq)}")
print("C  slab_mmm")
mono, frac = [], []
for st in ids:
    ser = sl.plan[st].get("sag_fluid")
    if not ser:
        continue
    n = sl._read_series(st, ser).shape[0]
    v = sl._load(st, ser, np.random.default_rng(0), 18).reshape(6, 3, 336, 336)
    mono.append(bool((v[:, 0] <= v[:, 1] + 1e-3).all() and (v[:, 1] <= v[:, 2] + 1e-3).all()))
    lo = int(n * sl.margin)
    frac.append((n - 2 * lo) / n)
check("C min <= mean <= max on every pixel", all(mono), f"{sum(mono)}/{len(mono)}")
check("C slabs use >= 80% of stored slices", np.mean(frac) >= 0.8, f"mean {np.mean(frac):.0%}")

print("D  laterality")
for name, ds in (("multiwindow", mw), ("slab_mmm", sl)):
    dfi = ds.df.reset_index(drop=True)
    left = next(i for i, s in enumerate(dfi.StudyInstanceUID) if ds.sides.get(s) == "left" and s in have)
    right = next(i for i, s in enumerate(dfi.StudyInstanceUID) if ds.sides.get(s) == "right" and s in have)

    def item(i, flip):
        old = ds.flip_side; ds.flip_side = flip
        try:
            return ds[i][0].float()
        finally:
            ds.flip_side = old
    xl, xn = item(left, "left"), item(left, None)
    sag = ds.slots.index("sag_fluid")
    gl, gn = xl[sag].reshape(6, 3, 336, 336), xn[sag].reshape(6, 3, 336, 336)
    check(f"D {name}: left sag = images reversed, channel order kept", torch.equal(gl, torch.flip(gn, dims=[0])))
    others = [k for k, s in enumerate(ds.slots) if not s.startswith("sag")]
    check(f"D {name}: left cor/ax = width reversed",
          all(torch.equal(xl[k], torch.flip(xn[k], dims=[-1])) for k in others))
    check(f"D {name}: right study untouched", torch.equal(item(right, "left"), item(right, None)))

print("E  randconv")
rc = L["convnext_base_25d_v5_randconv"]["train"].dataset
c0 = copy.deepcopy(cfgs["convnext_base_25d_v5_randconv"]); c0["augmentation"].pop("randconv", None)
cz = copy.deepcopy(cfgs["convnext_base_25d_v5_randconv"]); cz["augmentation"]["randconv"]["p"] = 0.0
c1 = copy.deepcopy(cfgs["convnext_base_25d_v5_randconv"]); c1["augmentation"]["randconv"]["p"] = 1.0
d0, dz, d1 = (loaders(c)["train"].dataset for c in (c0, cz, c1))
i0 = next(i for i, s in enumerate(d0.df.StudyInstanceUID) if s in have)
a, b, c_ = d0[i0][0], dz[i0][0], d1[i0][0]
check("E randconv absent == p 0 (no rng drawn when off)", torch.equal(a, b))
check("E randconv p 1 changes the item, still uint8", not torch.equal(a, c_) and c_.dtype == torch.uint8,
      f"mean |diff| {(a.float() - c_.float()).abs().mean():.1f}")
check("E the run's own config: p 0.25, kernels 1/3/5, alpha 0.1-0.5",
      rc.aug["randconv"] == {"p": 0.25, "kernels": [1, 3, 5], "alpha": [0.1, 0.5]})

print("F  default configs unchanged")
cb = cpu(sc.load_config(ROOT / "runs/convnext_base_25d_18slices_v5_aug_base", verbose=False))
db = loaders(cb)["val"].dataset
check("F aug_base: channel_mode slices, grp == ctx == 1", db.channel_mode == "slices" and db.grp == db.ctx == 1)

print("G  tiny real runs")
import timm  # noqa: E402
for r in RUNS:
    tmp = Path(tempfile.mkdtemp(prefix="lcr_"))
    try:
        c = copy.deepcopy(cfgs[r])
        trn = pd.read_csv(ROOT / "data" / c["paths"]["train_csv"]); trn = trn[trn.StudyInstanceUID.isin(have)].head(6)
        vv = pd.read_csv(ROOT / "data" / c["paths"]["val_csv"]); vv = vv[vv.StudyInstanceUID.isin(have)].head(4)
        trn.to_csv(tmp / "tr.csv", index=False); vv.to_csv(tmp / "va.csv", index=False)
        c["experiment"] = dict(c["experiment"], name="lcr_tiny")
        c["model"] = dict(c["model"], name="resnet18", pretrained=False)
        c["input"] = dict(c["input"], size=32)
        c["dataloader"].update(batch_size=2, accum_steps=1, val_batch_size=2)
        c["training"] = dict(c["training"], epochs=1, grad_checkpointing=False)
        c["training"]["early_stopping"] = dict(c["training"]["early_stopping"], enable=False)
        c["training"]["extend"] = dict(c["training"].get("extend") or {}, enable=False)
        c["evaluation"] = dict(c["evaluation"], eval_every_steps_before=3, eval_every_steps_mid=3,
                               eval_every_steps_after=3, eval_every_steps_epoch_mid_threshold=None,
                               eval_every_steps_epoch_threshold=1, validate_before_training=False)
        c["evaluation"]["audit"] = dict(c["evaluation"].get("audit") or {}, enable=False)
        c["output"] = dict(c["output"], checkpoint_every_steps=0, save_debug_images=0, keep_last_n=1, keep_top_k=0,
                           upload_checkpoints=None)
        c["paths"] = dict(c["paths"], train_csv=str(tmp / "tr.csv"), val_csv=str(tmp / "va.csv"))
        c["wandb"] = {"enabled": False}; c["resume"] = None

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0, in_chans=3)
                self.head = nn.Linear(self.backbone.num_features, 12)

            def forward(self, xx):
                f = self.backbone(xx)
                return self.head(f.view(-1, 24, f.shape[-1]).mean(1))
        with contextlib.redirect_stdout(io.StringIO()):
            sc.run_experiment(c, Tiny(), tmp / "run")
        vl = pd.read_csv(tmp / "run/results/val_log.csv")
        check(f"G {r.split('_v5_')[1]}: tiny run trained and validated", len(vl) >= 1 and np.isfinite(vl.val_loss).all())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL OK (%d/%d)" % (n_all[0], n_all[0]) if not fails else "FAILED: %s" % ", ".join(fails)))
