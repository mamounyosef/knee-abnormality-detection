"""joint_zoom_check.py - prove convnext_base_25d_v5_joint_zoom loads ITS OWN config and data and trains, before launch.

Checks: config values; seed load (trunk/head = parent, localizer = parent, frozen); optimizer sees only trainable
params; one train item and one audit item through prepare_batch -> model (forward + backward, amp); crop centres
inside the frame; jitter only in train mode; the saved crops drawn to checks/joint_zoom_crops.png for an eye check.

    python checks/joint_zoom_check.py [run]      run: convnext_base_25d_v5_joint_zoom (default) | ..._dual
"""
import importlib.util
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
ok = lambda c, m: print(("  OK   " if c else "  FAIL ") + m) or c
res = []
res.append(ok(cfg["experiment"]["name"] == NAME, f"experiment name {NAME}"))
i = cfg["input"]
res.append(ok(abs(i["size"] * i["mm_per_px"] - 134.4) < 0.5 and i["mm_per_px"] < 0.40, f"input frame 134 mm ({i['size']}, {i['mm_per_px']})"))
res.append(ok(len(i["slots"]) == 4, f"slots {i['slots']}"))
res.append(ok(i.get("resize_to") in (None, 0), "no resize_to (the trunk must get the full-detail frame)"))
res.append(ok(i["laterality_flip"] == "left", "laterality flip left"))
res.append(ok(len(cfg["tasks"]) == 12, "12 tasks"))

model = mod.build_model(cfg)
par = torch.load(ROOT / "runs" / mod.PARENT_CKPT, map_location="cpu", weights_only=False)["model"]
sd = model.state_dict()
k = "backbone.stages.3.blocks.2.mlp.fc2.weight"
# the trunk may come from another member (TRUNK_CKPT, e.g. the DINOv3 zoom); the localizer is always the parent
tr = torch.load(ROOT / "runs" / mod.TRUNK_CKPT, map_location="cpu", weights_only=False)["model"] if hasattr(mod, "TRUNK_CKPT") else par
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
res.append(ok(tuple(x.shape) == (4, 18, i["size"], i["size"]) and x.dtype == torch.uint8, f"train item {tuple(x.shape)} {x.dtype}"))
xb = sc.prepare_batch(x[None], cfg, dev)
res.append(ok(tuple(xb.shape) == (24, 3, i["size"], i["size"]), f"prepared batch {tuple(xb.shape)}"))
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
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
s0 = model.crop
for kk in range(4):
    cy, cx = c1[kk].tolist()
    s = getattr(model, "wins", [s0] * 4)[kk]   # per-view window when the run has one
    y0, x0 = int(np.clip(round(cy - s / 2), 0, i["size"] - s)), int(np.clip(round(cx - s / 2), 0, i["size"] - s))
    img = xa[kk, 9].numpy()
    ax[0, kk].imshow(img, cmap="gray")
    ax[0, kk].add_patch(plt.Rectangle((x0, y0), s, s, fill=False, ec="lime", lw=2))
    ax[0, kk].set_title(cfg["input"]["slots"][kk])
    ax[1, kk].imshow(img[y0:y0 + s, x0:x0 + s], cmap="gray")
    for a in ax[:, kk]:
        a.axis("off")
fig.tight_layout()
fig.savefig(ROOT / "checks" / f"{NAME}_crops.png", dpi=60)
print(f"\n{'ALL OK' if all(res) else 'SOME CHECKS FAILED'} ({sum(res)}/{len(res)}); crops -> checks/{NAME}_crops.png")
print(f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
