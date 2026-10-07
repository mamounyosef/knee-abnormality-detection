"""round2_v3val_check.py - do the 5 round-2 ConvNeXt folds train on v4, validate on v3, mirror left knees, and
resume across the val change with a ONE-TIME early-stopping reset? CPU only (no GPU), local shards. (7 Oct)

PART A, the real configs through the engine's own loaders (sc.make_loaders, num_workers 0):
  A1 train rows = data/27_fold{k}_train.csv, labels = v4 (22_*-w50), as the engine's dataset holds them
  A2 val rows = data/21_fold{k}_val.csv = the same studies as 27_fold{k}_val, labels = v3, no train/val overlap,
     audit = the 58 panel studies in neither
  A3 knee flip: flip_side 'left', every study sided, and on a real LEFT study the engine's item equals the unflipped
     item mirrored per plane (coronal/axial: width reversed; sagittal: slice order reversed); a RIGHT study is untouched
  A4 the resume guard: the folds set resume_force (only paths.val_csv moved) and nothing that trains changed
PART B, a tiny real run_experiment on CPU (resnet18 32 px, 6 studies):
  leg 1: val graded by file A, early stopping on, no reset tag      -> best = X
  leg 2: val graded by file B, reset_tag "v3val", resume_force       -> prints the reset ONCE, scores against B,
                                                                       patience counts from 0
  leg 3: same tag, resumed again                                     -> no second reset

    python checks/round2_v3val_check.py
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
     "Synovitis", "Baker's", "Contusion", "Fracture"]
fails, n_all = [], [0]


def check(name, ok, detail=""):
    n_all[0] += 1
    print("  %-72s %s%s" % (name, "ok" if ok else "FAIL", "  " + str(detail) if detail else ""))
    if not ok:
        fails.append(name)


v4 = pd.read_csv(ROOT / "data/22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
aud_ids = set(pd.read_csv(ROOT / "data/05_audit58.csv").StudyInstanceUID)
have = {q.stem for q in (ROOT / "data/shards").glob("*.npz")}


def cpu(cfg):
    c = copy.deepcopy(cfg)
    c["dataloader"].update(num_workers=0, val_num_workers=0, persistent_workers=False, prefetch_factor=None,
                           pin_memory=False)
    return c


print("PART A - the real fold configs through the engine's loaders")
for k in range(5):
    name = f"convnext_base_25d_18slices_v4_flip_f{k}"
    cfg = sc.load_config(ROOT / "runs" / name, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        L = sc.make_loaders(cpu(cfg))
    tr, va = L["train"].dataset, L["val"].dataset
    v3 = pd.read_csv(ROOT / f"data/21_fold{k}_val.csv").set_index("StudyInstanceUID")
    t27 = pd.read_csv(ROOT / f"data/27_fold{k}_train.csv")
    v27 = pd.read_csv(ROOT / f"data/27_fold{k}_val.csv")
    trd, vad = tr.df.set_index("StudyInstanceUID"), va.df.set_index("StudyInstanceUID")
    print(f" fold {k}:")
    check("A1 train = 27_fold train rows", len(trd) == len(t27) and set(trd.index) == set(t27.StudyInstanceUID),
          f"{len(trd)} rows")
    check("A1 train labels = v4", (trd[T] - v4.loc[trd.index, T]).abs().to_numpy().max() < 1e-12)
    check("A2 val = same studies as 27_fold val", set(vad.index) == set(v27.StudyInstanceUID), f"{len(vad)} rows")
    check("A2 val labels = v3 (21_fold val)", (vad[T] - v3.loc[vad.index, T]).abs().to_numpy().max() < 1e-12)
    check("A2 val labels are NOT v4", (vad[T] - v4.loc[vad.index, T]).abs().to_numpy().mean() > 0.01,
          f"mean |v3 - v4| {(vad[T] - v4.loc[vad.index, T]).abs().to_numpy().mean():.3f}")
    check("A2 no train/val overlap, audit in neither",
          not (set(trd.index) & set(vad.index)) and not ((set(trd.index) | set(vad.index)) & aud_ids))
    if "audit" in L:
        ad = L["audit"].dataset.df
        check("A2 audit loader = the 58 panel studies", set(ad.StudyInstanceUID) == aud_ids, f"{len(ad)} rows")
    check("A3 flip_side 'left' on train and val", tr.flip_side == "left" and va.flip_side == "left")
    sided = sum(s in tr.sides for s in trd.index) + sum(s in va.sides for s in vad.index)
    check("A3 every study has a known side", sided == len(trd) + len(vad), f"{sided}/{len(trd) + len(vad)}")
    if k in (0, 1):
        need = set(trd.index) | set(vad.index) | aud_ids
        check("A3 local fold: every needed shard is on this PC", need <= have, f"{len(need - have)} missing")
    check("A4 resume_force set (the val file changed)", bool(cfg.get("resume_force")))
    es = cfg["training"]["early_stopping"]
    check("A4 early stopping on, patience 7, tag v3val, on val macro AUROC",
          es["enable"] and es["patience"] == 7 and es.get("reset_tag") == "v3val" and es["monitor"] == "val_macro_auroc")

# A3 mirror: a real LEFT and a real RIGHT val study of fold 2, item with the flip vs without
print(" mirror, fold 2 val items (no augmentation on val):")
cfg = cpu(sc.load_config(ROOT / "runs/convnext_base_25d_18slices_v4_flip_f2", verbose=False))
with contextlib.redirect_stdout(io.StringIO()):
    va = sc.make_loaders(cfg)["val"].dataset
ids = list(va.df.StudyInstanceUID)
left = next(i for i, s in enumerate(ids) if va.sides.get(s) == "left" and s in have)
right = next(i for i, s in enumerate(ids) if va.sides.get(s) == "right" and s in have)


def item(ds, i, flip):
    old = ds.flip_side
    ds.flip_side = flip
    try:
        out = ds[i]
    finally:
        ds.flip_side = old
    x = out[0] if isinstance(out, (tuple, list)) else out["x"]
    return torch.as_tensor(x).float(), out


xl, outl = item(va, left, "left")
xn, _ = item(va, left, None)
S, C = len(va.slots), int(cfg["input"]["slices_as_channels"])
xl_, xn_ = xl.reshape(S, -1, *xl.shape[-2:]), xn.reshape(S, -1, *xn.shape[-2:])
for s_i, slot in enumerate(va.slots):
    a, b = xl_[s_i], xn_[s_i]
    if not b.abs().sum():
        continue
    exp = torch.flip(b, dims=[0]) if slot.startswith("sag") else torch.flip(b, dims=[-1])
    check(f"A3 left study, {slot}: {'slice order' if slot.startswith('sag') else 'width'} reversed",
          torch.allclose(a, exp, atol=1e-5), f"max diff {(a - exp).abs().max():.2e}")
xr, _ = item(va, right, "left")
xr0, _ = item(va, right, None)
check("A3 right study untouched by the flip", torch.equal(xr, xr0))
lab = outl[1] if isinstance(outl, (tuple, list)) else outl["y"]
row = va.df.iloc[left][T].to_numpy(float)
check("A3 the item's labels = its v3 CSV row", np.allclose(np.asarray(lab, float)[:12], row, atol=1e-6))

print("\nPART B - tiny real run on CPU: resume across a re-graded val, one-time reset")
tmp = Path(tempfile.mkdtemp(prefix="v3val_"))
try:
    base = sc.load_config(ROOT / "runs/convnext_base_25d_18slices_v4_flip_f2", verbose=False)
    loc = sorted(have)
    trn = pd.read_csv(ROOT / "data/27_fold2_train.csv")
    trn = trn[trn.StudyInstanceUID.isin(loc)].head(6)
    vA = pd.read_csv(ROOT / "data/27_fold2_val.csv")
    vA = vA[vA.StudyInstanceUID.isin(loc)].head(4)
    vB = vA.copy()
    vB[T] = 1 - vB[T]                     # a DIFFERENT grading of the same studies, so its AUROC differs
    trn.to_csv(tmp / "tr.csv", index=False)
    vA.to_csv(tmp / "vaA.csv", index=False)
    vB.to_csv(tmp / "vaB.csv", index=False)

    def cfg_for(epochs, val, tag, resume):
        c = cpu(base)
        c["experiment"] = dict(base["experiment"], name="v3val_test")
        c["model"] = dict(base["model"], name="resnet18", pretrained=False)
        c["input"] = dict(base["input"], size=32, n_slices=3, slices_as_channels=1, mm_per_px=None)
        c["dataloader"].update(batch_size=2, accum_steps=1, val_batch_size=2, drop_last=False)
        c["training"] = dict(base["training"], epochs=epochs)
        c["training"]["early_stopping"] = dict(base["training"]["early_stopping"], enable=True, patience=3,
                                               reset_tag=tag)
        c["training"]["extend"] = dict(base["training"].get("extend") or {}, enable=False)
        c["evaluation"] = dict(base["evaluation"], eval_every_steps_before=3, eval_every_steps_mid=3,
                               eval_every_steps_after=3, eval_every_steps_epoch_mid_threshold=None,
                               eval_every_steps_epoch_threshold=1, validate_before_training=False)
        c["evaluation"]["audit"] = dict(base["evaluation"]["audit"], enable=False)
        c["output"] = dict(base["output"], checkpoint_every_steps=0, save_debug_images=0, keep_last_n=3,
                           keep_top_k=0, upload_checkpoints=None)
        c["paths"] = dict(base["paths"], train_csv=str(tmp / "tr.csv"), val_csv=str(tmp / val))
        c["wandb"] = {"enabled": False}
        c["resume"] = resume
        c["resume_force"] = True          # epochs/val_csv move between legs (as in the real resume)
        return c

    import timm

    def build(c):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0, in_chans=3)
                self.per = len(c["input"]["slots"]) * int(c["input"]["n_slices"]) // 3   # 3 slices per 3-channel image
                self.head = nn.Linear(self.backbone.num_features, sc.num_output_logits(c))

            def forward(self, x):
                f = self.backbone(x)
                return self.head(f.view(-1, self.per, f.shape[-1]).mean(1))
        return Tiny()

    def leg(c):
        sc.set_seed(42, False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sc.run_experiment(c, build(c), tmp / "run")
        return buf.getvalue()

    o1 = leg(cfg_for(1, "vaA.csv", None, None))
    o2 = leg(cfg_for(2, "vaB.csv", "v3val", "last"))
    o3 = leg(cfg_for(3, "vaB.csv", "v3val", "last"))
    if "--debug" in sys.argv:
        import re
        for nm, o in (("leg2", o2), ("leg3", o3)):
            print(f"--- {nm} ---")
            print("\n".join(l for l in o.splitlines()
                            if re.search(r"restor|reset|resum|new best|CONFIG|checkpoint", l))[:3000])
    vl = pd.read_csv(tmp / "run/results/val_log.csv")
    check("B leg 2 printed the one-time reset", "early stopping reset for metric tag 'v3val'" in o2)
    check("B leg 1 had no reset (no tag)", "early stopping reset" not in o1)
    check("B leg 3 (same tag) did NOT reset again", "early stopping reset" not in o3)
    check("B leg 2 restored then reset: a new best was set after the resume", "new best val_macro_auroc" in o2)
    steps = vl.drop_duplicates("step", keep="last")
    check("B validations ran in every leg", len(steps) >= 3, f"steps {list(steps.step)}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + ("ALL OK (%d/%d)" % (n_all[0], n_all[0]) if not fails else "FAILED: %s" % ", ".join(fails)))
