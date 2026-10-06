"""joint_zoom_check.py - prove convnext_base_25d_v5_joint_zoom loads ITS OWN config and data and trains, before launch.

Checks: config values; seed load (trunk/head = parent, localizer = parent, frozen); optimizer sees only trainable
params; one train item and one audit item through prepare_batch -> model (forward + backward, amp); crop centres
inside the frame; jitter only in train mode; the saved crops drawn to checks/joint_zoom_crops.png for an eye check.

    python checks/joint_zoom_check.py [run]      run: convnext_base_25d_v5_joint_zoom (default) | ..._dual
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
NAME = sys.argv[1] if len(sys.argv) > 1 else "convnext_base_25d_v5_joint_zoom"
RUN = ROOT / "runs" / NAME
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

spec = importlib.util.spec_from_file_location("jz_train", RUN / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.cfg
REMOTE = os.environ.get("KNEE_REMOTE") == "1"            # run on Modal (runs/others/crop_rois/modal_check.py)
if REMOTE:
    cfg = sc.remote_cfg(cfg)                              # container paths: shards and data mounts
ok = lambda c, m: print(("  OK   " if c else "  FAIL ") + m) or c
res = []
res.append(ok(cfg["experiment"]["name"] == NAME, f"experiment name {NAME}"))
i = cfg["input"]
res.append(ok(abs(i["size"] * i["mm_per_px"] - 134.4) < 0.5 and i["mm_per_px"] < 0.40, f"input frame 134 mm ({i['size']}, {i['mm_per_px']})"))
res.append(ok(len(i["slots"]) in (4, 6), f"slots {i['slots']}"))
res.append(ok(i.get("resize_to") in (None, 0), "no resize_to (the trunk must get the full-detail frame)"))
res.append(ok(i["laterality_flip"] == "left", "laterality flip left"))
res.append(ok(len(cfg["tasks"]) == 12, "12 tasks"))

model = mod.build_model(cfg)
_ck = (lambda rel: Path(cfg["modal"]["runs_mount"]) / rel if REMOTE else ROOT / "runs" / rel)
par = torch.load(_ck(mod.PARENT_CKPT), map_location="cpu", weights_only=False)["model"]
sd = model.state_dict()
k = "backbone.stages.3.blocks.2.mlp.fc2.weight"
# the trunk may come from another member (TRUNK_CKPT, e.g. the DINOv3 zoom); the localizer is always the parent
tr = torch.load(_ck(mod.TRUNK_CKPT), map_location="cpu", weights_only=False)["model"] if hasattr(mod, "TRUNK_CKPT") else par
tname = mod.TRUNK_CKPT.split("/")[0] if hasattr(mod, "TRUNK_CKPT") else "parent"
res.append(ok(torch.equal(sd[k], tr[k]) and torch.equal(sd["loc_" + k], par[k]), f"trunk = {tname}, localizer = parent"))
res.append(ok(torch.equal(sd["head.weight"], tr["head.weight"]) and torch.equal(sd["loc_head.weight"], par["head.weight"]), f"head = {tname} rows, loc head = parent rows"))
if "feat_norm.weight" in sd:
    res.append(ok(torch.equal(sd["feat_norm.weight"], tr["feat_norm.weight"]), "feat_norm seeded"))
    res.append(ok(not torch.equal(sd[k], par[k]), "trunk differs from the localizer"))
res.append(ok(all(not p.requires_grad for n, p in model.named_parameters() if n.startswith("loc_")), "localizer frozen"))
opt = sc.build_optimizer(model, cfg)
n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
n_tr = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("loc_"))
res.append(ok(n_opt == n_tr, f"optimizer holds exactly the trainable params ({n_opt / 1e6:.2f}M)"))
model.train()
res.append(ok(not model.loc_backbone.training and model.backbone.training, "train(): localizer stays eval"))

dev = torch.device("cuda")
model.backbone.set_grad_checkpointing(True)   # local 8 GB only; the H100 run does not need it
model.to(dev)
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plan = sc.plan_slots(idx, cfg)
tr = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg)).head(4)
ds_tr = sc.KneeShardDataset(tr, cfg, plan, train=True)
x, y = ds_tr[0]
res.append(ok(tuple(x.shape) == (len(i["slots"]), i["n_slices"], i["size"], i["size"]) and x.dtype == torch.uint8, f"train item {tuple(x.shape)} {x.dtype}"))
xb = sc.prepare_batch(x[None], cfg, dev)
res.append(ok(tuple(xb.shape) == (len(i["slots"]) * i["slices_as_channels"], 3, i["size"], i["size"]), f"prepared batch {tuple(xb.shape)}"))
with torch.autocast("cuda"):
    out = model(xb)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y[None].to(dev))
loss.backward()
g = model.backbone.stem[0].weight.grad
res.append(ok(out.shape == (1, 12) and g is not None and torch.isfinite(g).all(), f"forward+backward, loss {loss.item():.4f}"))
res.append(ok(model.loc_backbone.stem[0].weight.grad is None, "no gradient into the localizer"))
opt.zero_grad(set_to_none=True)

model.eval()
au = sc._eval_df(cfg, cfg["evaluation"]["audit"]["csv"]).head(3)
ds_au = sc.KneeShardDataset(au, cfg, plan, train=False)
xa, _ = ds_au[0]
xab = sc.prepare_batch(xa[None], cfg, dev)
with torch.no_grad():
    c1 = model.centres(xab).cpu()
    c2 = model.centres(xab).cpu()
    o = model(xab)
res.append(ok(torch.equal(c1, c2), "eval centres deterministic"))
res.append(ok(bool(((c1 >= 0) & (c1 <= i["size"])).all()), f"centres inside the frame: {c1.round().tolist()}"))
res.append(ok(torch.isfinite(o).all().item(), "eval logits finite"))
if i.get("short_stack") == "pad":
    # SCAN: on 120 real val/audit studies, every view's non-blank images must equal ceil(min(native slices, n_slices) / 3)
    # (0 for an absent view), blank images must be exactly zero, and no real image may be blank
    import pandas as _pd2
    _ix = _pd2.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)).set_index("series")
    _vd = sc._eval_df(cfg, cfg["paths"]["val_csv"]).sample(120, random_state=3).reset_index(drop=True)
    _ds = sc.KneeShardDataset(_vd, cfg, plan, train=False)
    _bad, _nviews, _short, _abs = [], 0, 0, 0
    for _j in range(len(_vd)):
        _x, _ = _ds[_j]
        _x = _x.view(len(i["slots"]), i["slices_as_channels"], 3, i["size"], i["size"])
        _st = _vd.loc[_j, "StudyInstanceUID"]
        for _k, _sl in enumerate(i["slots"]):
            _se = plan.get(_st, {}).get(_sl)
            _n = int(_ix.loc[_se, "n_slices"]) if _se else 0
            _want = -(-min(_n, i["n_slices"]) // 3)
            _flat = _x[_k].flatten(2).float()
            _real = ((_flat.amax(2) - _flat.amin(2)) > 0).any(1)
            _got = int(_real.sum())
            _nviews += 1; _short += int(0 < _n < i["n_slices"]); _abs += int(_n == 0)
            # padded slices must be exactly 0 (uint8): count all-zero SLICES = n_slices - min(n, n_slices)
            _zero_sl = int((_x[_k].reshape(-1, i["size"] * i["size"]).amax(1) == 0).sum())
            if _got != _want or (_n and _zero_sl < i["n_slices"] - min(_n, i["n_slices"])):
                _bad.append((_st[-8:], _sl, _n, _got, _want, _zero_sl))
    res.append(ok(not _bad, f"padding scan: {_nviews} views of 120 studies ({_short} short, {_abs} absent): "
                            f"non-blank images = real-slice groups everywhere" + (f"; FIRST BAD {_bad[:4]}" if _bad else "")))
    # short-stack padding: blank images must be exactly the padded / absent ones, and adding blanks must not move logits
    import pandas as _pd
    _idx = _pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg)).set_index("series")
    st = au.iloc[0]["StudyInstanceUID"]
    want = []
    for sl in i["slots"]:
        se = plan.get(st, {}).get(sl)
        nreal = min(int(_idx.loc[se, "n_slices"]), i["n_slices"]) if se else 0
        want.append(-(-nreal // 3))                      # images holding at least one real slice
    got = model.valid_images(xab).view(len(i["slots"]), -1).sum(1).tolist()
    res.append(ok(got == want, f"valid images per view {got} = real-slice groups {want}"))
    xz = xab.clone().view(len(i["slots"]), i["slices_as_channels"], *xab.shape[1:])
    keep = model.valid_images(xab).view(len(i["slots"]), -1)
    with torch.no_grad():
        # blank out one REAL image of view 0: it must now be skipped (logits change) - proves real images count
        k0 = int(keep[0].nonzero()[0])
        xz[0, k0] = xz[0, k0].flatten(1).amin(1)[:, None, None]
        o_drop = model(xz.view_as(xab))
    res.append(ok((o_drop - o).abs().max().item() > 1e-4, "a blanked real image is skipped (logits move)"))
    # a study whose padding is replaced by a DIFFERENT constant still gives the same logits (blanks carry nothing)
    xb2 = xab.clone()
    blank = ~model.valid_images(xab)
    if blank.any():
        xb2[blank] = 7.0
        with torch.no_grad():
            o_b = model(xb2)
        res.append(ok((o_b - o).abs().max().item() < 1e-3, f"blank images carry nothing (max diff {(o_b - o).abs().max().item():.1e})"))
if "findings" in cfg["model"]["zoom"]:
    # dual: at step 0 the trunk IS the localizer, so the full-view findings must equal the frozen parent on its view
    with torch.no_grad():
        lo = torch.nn.functional.interpolate(xab, size=(336, 336), mode="area")
        f = model.loc_backbone(lo).view(1, -1, model.loc_backbone.num_features).mean(1)
        ref = model.loc_head(f)
    m = model.zoom_mask[0].cpu()
    d_full = (o.cpu()[0, ~m] - ref.cpu()[0, ~m]).abs().max().item()
    d_zoom = (o.cpu()[0, m] - ref.cpu()[0, m]).abs().max().item()
    zf = [t for t, b in zip(cfg["tasks"], m.tolist()) if b]
    res.append(ok(zf == ["ACL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"], f"zoom findings {zf}"))
    res.append(ok(d_full < 1e-3, f"full-view findings = parent at step 0 (max diff {d_full:.2e})"))
    res.append(ok(d_zoom > 1e-3, f"zoom findings differ from the parent (crops read; max diff {d_zoom:.3f})"))
    res.append(ok(cfg["dataloader"]["batch_size"] * cfg["dataloader"]["accum_steps"] == 8, "8 studies per update"))
# parent on its own view vs zoom model at step 0, same study (the step-0 validation will show the full picture)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
NS = len(i["slots"])
fig, ax = plt.subplots(2, NS, figsize=(4 * NS, 8))
s0 = model.crop
for kk in range(NS):
    cy, cx = c1[kk].tolist()
    s = getattr(model, "wins", [s0] * NS)[kk]   # per-view window when the run has one
    y0, x0 = int(np.clip(round(cy - s / 2), 0, i["size"] - s)), int(np.clip(round(cx - s / 2), 0, i["size"] - s))
    img = xa[kk, i["n_slices"] // 2].numpy()
    ax[0, kk].imshow(img, cmap="gray")
    ax[0, kk].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="lime", lw=2))
    ax[0, kk].set_title(cfg["input"]["slots"][kk])
    ax[1, kk].imshow(img[y0:y0 + s, x0:x0 + s], cmap="gray")
    for a in ax[:, kk]:
        a.axis("off")
fig.tight_layout()
fig.savefig((Path("/tmp") if REMOTE else ROOT / "checks") / f"{NAME}_crops.png", dpi=60)
print(f"\n{'ALL OK' if all(res) else 'SOME CHECKS FAILED'} ({sum(res)}/{len(res)}); crops -> checks/{NAME}_crops.png")
print(f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
