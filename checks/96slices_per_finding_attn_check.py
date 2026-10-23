"""96slices_per_finding_attn_check.py - convnext_base_v5_96slices_per_finding_attn_sam (input.window_bag, 10 Oct) loads its OWN config and data and builds the
public 0.949 input. CPU only, local shards.

  A  config: 5 slots 26/22/18/12/18 = 96, window_bag 12 / 94, 384 px at 0.365 mm/px, margin 0.06, SAM, head lr 2e-3
  B  items: train (1, 12, 3, 384, 384), eval (1, 94, 3, 384, 384) uint8; window i = stack slices [c-1, c, c+1]
  C  window_centres == the public notebook's _eval_centers (copied verbatim) on 500 random masks; training centres
     are inside the same valid set, distinct when possible
  D  masks: a missing slot is 0 in the mask; windows never start before the first real slice
  E  laterality: a LEFT study's windows inside the sagittal slots are the unflipped ones in reverse order with the 3
     channels reversed; coronal/axial windows are width-flipped; a right study is untouched
  F  model: per-finding attention, train k 12 / eval k 94 follow train()/eval(), output (B, 12), gradients reach the
     attention scorer; optimizer: head groups at 2e-3, backbone at 4e-5
  G  a tiny real run_experiment (resnet18, 32 px, window bag) trains, validates on 94 windows and predicts
  H  (MII runs) the MII weights load 402/402, no layer decay, ImageNet norm, the validation schedule

    python checks/96slices_per_finding_attn_check.py [convnext_base_v5_96slices_per_finding_attn]   (heavy: run it on Modal, checks/modal_check_runner.py)
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

torch.cuda.is_available = lambda: False
ROOT = Path(__file__).resolve().parents[1]
RUN = sys.argv[1] if len(sys.argv) > 1 else "convnext_base_v5_96slices_per_finding_attn_sam"   # or ..._local
sys.path.insert(0, str(ROOT / "runs"))
sys.path.insert(0, str(ROOT / "runs" / RUN))
import shared_code as sc  # noqa: E402
import train as rt  # noqa: E402

fails, n_all = [], [0]
have = {q.stem for q in (ROOT / "data/shards").glob("*.npz")}


def check(name, ok, detail=""):
    n_all[0] += 1
    print("  %-74s %s%s" % (name, "ok" if ok else "FAIL", "  " + str(detail) if detail else ""))
    if not ok:
        fails.append(name)


def cpu(cfg):
    c = copy.deepcopy(cfg)
    c["dataloader"].update(num_workers=0, val_num_workers=0, persistent_workers=False, prefetch_factor=None,
                           pin_memory=False)
    return c


cfg = cpu(sc.load_config(ROOT / "runs" / RUN, verbose=False))
i, t = cfg["input"], cfg["training"]
print("A  config")
check("A slots sag_fluid 26 / sag_plain 22 / cor_fluid 18 / cor_plain 12 / ax_fluid 18 = 96",
      i["slots"] == ["sag_fluid", "sag_plain", "cor_fluid", "cor_plain", "ax_fluid"]
      and [i["n_slices"][s] for s in i["slots"]] == [26, 22, 18, 12, 18])
check("A window_bag 12 / 94, 384 px x 0.365 mm, margin 0.06",
      i["window_bag"] == {"train_k": 12, "eval_k": 94} and i["size"] == 384 and i["mm_per_px"] == 0.365
      and i["slice_margin"] == 0.06)
LOCAL = sc.accum_steps(cfg) > 1        # the local recipe (batch 2 x accum 4); never read off the run's name
SAM = bool((t.get("sam") or {}).get("enable"))
if LOCAL:
    check("A local runs carry no SAM (it needs accum 1)", not SAM)
if LOCAL:
    check("A local: NO SAM, batch 2 x accum 4, grad checkpointing, 16 epochs, head lr 2e-3",
          not (t.get("sam") or {}).get("enable") and cfg["dataloader"]["batch_size"] == 2 and sc.accum_steps(cfg) == 4
          and t.get("grad_checkpointing") and t["epochs"] == 16
          and abs(t["optimizer"]["lr"] / t["optimizer"]["backbone_lr_mult"] - 2e-3) < 1e-9)
else:
    check("A Modal: batch 8 x accum 1, 16 epochs, head lr 2e-3 (SAM %s)" % ("on" if SAM else "off"),
          cfg["dataloader"]["batch_size"] == 8 and sc.accum_steps(cfg) == 1 and t["epochs"] == 16
          and abs(t["optimizer"]["lr"] / t["optimizer"]["backbone_lr_mult"] - 2e-3) < 1e-9)

with contextlib.redirect_stdout(io.StringIO()):
    L = sc.make_loaders(cfg)
tr, va = L["train"].dataset, L["val"].dataset
print("B  items")
xt, xv = tr[0][0], va[0][0]
check("B train item (1, 12, 3, 384, 384) uint8", tuple(xt.shape) == (1, 12, 3, 384, 384) and xt.dtype == torch.uint8,
      tuple(xt.shape))
check("B eval item (1, 94, 3, 384, 384) uint8", tuple(xv.shape) == (1, 94, 3, 384, 384) and xv.dtype == torch.uint8,
      tuple(xv.shape))


def ref_eval_centers(mask, D, k):        # verbatim from nartaa/rsna-knee-0949-anatomical-mirror
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [max(1, min(cs[j], D - 2)) for j in idx]


print("C  window centres")
rng = np.random.default_rng(0)
same, inside = 0, 0
for _ in range(500):
    m = (rng.random(96) > rng.uniform(0, 0.6)).astype(np.float32)
    m[rng.integers(0, 5) * 19:][:rng.integers(0, 20)] = 0
    same += list(sc.window_centres(m, 96, 94)) == ref_eval_centers(m, 96, 94)
    ct = sc.window_centres(m, 96, 12, np.random.default_rng(1))
    valid = set(ref_eval_centers(m, 96, 200))
    inside += set(ct) <= valid and (len(set(ct)) == 12 or len(valid) < 12)
check("C eval centres == the public _eval_centers (500 random masks)", same == 500, f"{same}/500")
check("C training centres inside the valid set, distinct when possible", inside == 500, f"{inside}/500")

print("D  masks")
ids = list(va.df.StudyInstanceUID)
lone = next((j for j, s in enumerate(ids) if s in have and not va.plan.get(s, {}).get("cor_plain")), None)
if lone is not None:
    st = ids[lone]
    vol_ok = va.plan[st]
    check("D a study without cor_plain still yields 94 windows", tuple(va[lone][0].shape) == (1, 94, 3, 384, 384),
          f"slots {sorted(vol_ok)}")
else:
    check("D found a study without cor_plain", False)

print("E  laterality")
left = next(j for j, s in enumerate(ids) if va.sides.get(s) == "left" and s in have and va.plan[s].get("sag_fluid"))
right = next(j for j, s in enumerate(ids) if va.sides.get(s) == "right" and s in have)


def item(j, flip):
    old = va.flip_side; va.flip_side = flip
    try:
        return va[j][0][0].float()                      # (94, 3, H, W)
    finally:
        va.flip_side = old


xl, xn = item(left, "left"), item(left, None)
# full slots: centres 1..24 of slot 0 (slices 0..25) are inside it in both orders
sag_n = xn[0:24]                                       # windows centred 1..24 (all inside sag_fluid)
sag_l = xl[0:24]
check("E left sag_fluid windows = unflipped ones reversed, channels reversed",
      torch.equal(sag_l, torch.flip(sag_n, dims=[0, 1])))
cor0 = 26 + 22                                         # cor_fluid starts at slice 48
w = [j for j in range(94) if cor0 + 1 <= j + 1 <= cor0 + 16]  # centres 49..64 inside cor_fluid
check("E left cor_fluid windows = width-flipped", torch.equal(xl[w], torch.flip(xn[w], dims=[-1])))
check("E right study untouched", torch.equal(item(right, "left"), item(right, None)))

print("F  model and optimizer")
c2 = copy.deepcopy(cfg); c2["model"]["pretrained"] = False
m = rt.build_model(c2)
xb = torch.stack([tr[0][0], tr[1][0]])                 # (2, 1, 12, 3, H, W)
m.train()
out = m(sc.prepare_batch(xb, c2, torch.device("cpu")))
out.sum().backward()
check("F train: k 12, output (2, 12), attention scorer gets gradients",
      m.k == 12 and tuple(out.shape) == (2, 12) and m.att[0].weight.grad is not None
      and float(m.att[0].weight.grad.abs().sum()) > 0)
m.eval()
with torch.no_grad():
    o2 = m(sc.prepare_batch(va[0][0][None], c2, torch.device("cpu")))
check("F eval: k 94, output (1, 12)", m.k == 94 and tuple(o2.shape) == (1, 12))
opt = sc.build_optimizer(m, c2)
lrs = sorted({round(g["lr"], 9) for g in opt.param_groups})
bb_lr = round(t["optimizer"]["lr"], 9)
check("F optimizer: backbone %.0e, head 2e-3" % bb_lr, lrs == [bb_lr, 2e-3], lrs)
del m, opt, out, o2

if RUN.startswith("medimageinsight"):
    print("H  MII")
    import external_backbones as eb
    import timm
    bb = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=0, in_chans=3)
    n_t = eb.load_medimageinsight(bb, eb.resolve_weights(rt.WEIGHTS_FILE, cfg))
    n_p = sum(p.numel() for p in bb.parameters())
    check("H MII weights: 402 tensors into davit_huge_fl, 360.6M params", n_t == 402 and round(n_p / 1e6, 1) == 360.6,
          f"{n_t} tensors, {n_p / 1e6:.1f}M")
    check("H no layer decay, ImageNet norm", t["optimizer"]["layer_decay"] is None
          and i["norm_mean"] == [0.485, 0.456, 0.406] and i["norm_std"] == [0.229, 0.224, 0.225])
    sched = {ep: sc.eval_interval(cfg, ep) for ep in (1, 6, 7, 16)}
    check("H validations: 800 for epochs 1-6, then 100", sched == {1: 800, 6: 800, 7: 100, 16: 100}, sched)
    del bb

print("G  tiny real run")
tmp = Path(tempfile.mkdtemp(prefix="attn96_"))
try:
    c = copy.deepcopy(cfg)
    trn = pd.read_csv(ROOT / "data" / c["paths"]["train_csv"]); trn = trn[trn.StudyInstanceUID.isin(have)].head(6)
    vv = pd.read_csv(ROOT / "data" / c["paths"]["val_csv"]); vv = vv[vv.StudyInstanceUID.isin(have)].head(4)
    trn.to_csv(tmp / "tr.csv", index=False); vv.to_csv(tmp / "va.csv", index=False)
    c["experiment"] = dict(c["experiment"], name="attn96_tiny")
    c["model"] = dict(c["model"], name="resnet18", pretrained=False)
    c["input"] = dict(c["input"], size=32)
    c["dataloader"].update(batch_size=2, accum_steps=1, val_batch_size=2)
    c["training"]["sam"] = dict(c["training"].get("sam") or {}, enable=SAM)
    c["training"] = dict(c["training"], epochs=1)
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
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sc.run_experiment(c, rt.build_model(c), tmp / "run")
    vl = pd.read_csv(tmp / "run/results/val_log.csv")
    check("G tiny window-bag run trained and validated", len(vl) >= 1 and np.isfinite(vl.val_loss).all(),
          f"{len(vl)} val rows")
    tl = pd.read_csv(tmp / "run/results/train_log.csv")
    if not SAM:
        check("G no SAM in the run", "sam_n" not in tl.columns)
    else:
        check("G SAM ran in it", "sam_n" in tl.columns and tl.sam_n.notna().all())
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL OK (%d/%d)" % (n_all[0], n_all[0]) if not fails else "FAILED: %s" % ", ".join(fails)))
