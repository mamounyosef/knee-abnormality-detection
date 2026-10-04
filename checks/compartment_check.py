"""compartment_check.py - is convnext_base_25d_v5_compartment exactly its parent plus compartment pooling, and is the
pooling wired the way it claims?

  1. loads its OWN config; differs from the parent in exactly the expected keys; 12 findings, the parent's 4 slots
  2. the branch seed: every backbone tensor and all 12 head rows equal the parent's top_step5900
  3. one REAL study through both models (CPU, batch 1): the 7 non-compartment findings give exactly the parent's
     logits; the 5 compartment findings differ; each compartment logit equals a hand computation (head applied to the
     average of coronal/axial half-maps and sagittal whole maps, lateral = image LEFT half, medial = image RIGHT half)
  4. the orientation premise still holds on a sample of shards (image right = patient left on coronal/axial)

    python checks/compartment_check.py
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
sys.path.insert(0, str(RUNS))
RUN, PARENT, PCK = "convnext_base_25d_v5_compartment", "convnext_base_25d_18slices_v5_aug_base", "top_step5900.pt"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LAT, MED = {"Lateral Meniscus", "Lateral OA"}, {"Medial Meniscus", "Medial OA", "MCL"}
EXPECTED = {"experiment.name", "training.epochs", "training.optimizer.lr", "evaluation.eval_every_steps_before",
            "evaluation.eval_every_steps_mid", "evaluation.validate_before_training"}
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
check(diff == EXPECTED, "1. differs from the parent in exactly the expected keys"
      + ("" if diff == EXPECTED else f": extra {sorted(diff - EXPECTED)}, missing {sorted(EXPECTED - diff)}"))
check(cfg["tasks"] == T and cfg["input"]["slots"] == pm.cfg["input"]["slots"], f"1. 12 findings, slots {cfg['input']['slots']}")
ev = cfg["evaluation"]
check(ev["eval_every_steps_before"] == ev["eval_every_steps_mid"] == ev["eval_every_steps_after"] == 100
      and ev["validate_before_training"] is True, "1. validates every 100 steps, and at step 0")

c2 = copy.deepcopy(cfg)
c2["model"]["pretrained"] = False
model = m.build_model(c2).eval()
psd = torch.load(RUNS / PARENT / "results" / "checkpoints" / PCK, map_location="cpu", weights_only=False)["model"]
sd = model.state_dict()
check(set(sd) == set(psd) and all(torch.equal(sd[k], psd[k]) for k in sd), f"2. all {len(sd)} tensors equal the parent's")
p2 = copy.deepcopy(pm.cfg)
p2["model"]["pretrained"] = False
parent = pm.build_model(p2).eval()
parent.load_state_dict(psd)

c2["dataloader"].update(num_workers=0, val_num_workers=0, batch_size=1, val_batch_size=1,
                        persistent_workers=False, pin_memory=False)
L = sc.make_loaders(c2)
xb = next(iter(L["val"]))[0]
x = sc.prepare_batch(xb, c2, torch.device("cpu"))
with torch.no_grad():
    a, b = model(x)[0], parent(x)[0]
    fm = model.backbone.forward_features(x)
    w = fm.shape[-1]
    whole = model.backbone.forward_head(fm, pre_logits=True)
    lat = model.backbone.forward_head(fm[..., : w // 2], pre_logits=True)
    med = model.backbone.forward_head(fm[..., w - w // 2:], pre_logits=True)
plane = [str(s).split("_")[0] for s in cfg["input"]["slots"] for _ in range(model.per_slot)]
hv = torch.tensor([p in ("cor", "ax") for p in plane]).view(-1, 1)
hand_lat = model.head(torch.where(hv, lat, whole).mean(0))
hand_med = model.head(torch.where(hv, med, whole).mean(0))
rest = [j for j, t in enumerate(T) if t not in LAT | MED]
comp = [j for j, t in enumerate(T) if t in LAT | MED]
check(torch.allclose(a[rest], b[rest], atol=1e-4), f"3. the 7 whole-image findings equal the parent's logits "
      f"(max diff {(a[rest] - b[rest]).abs().max():.1e})")
check(all(abs(float(a[j] - b[j])) > 1e-4 for j in comp), "3. the 5 compartment findings differ from the parent")
ok_lat = all(torch.allclose(a[T.index(t)], hand_lat[T.index(t)], atol=1e-4) for t in LAT)
ok_med = all(torch.allclose(a[T.index(t)], hand_med[T.index(t)], atol=1e-4) for t in MED)
check(ok_lat and ok_med, f"3. lateral logits = left halves (+ sagittal whole), medial = right halves, by hand; "
      f"{sum(p in ('cor', 'ax') for p in plane)} of {len(plane)} images are split")
print("  logits parent -> compartment: " + ", ".join(f"{T[j]} {float(b[j]):+.2f} -> {float(a[j]):+.2f}" for j in comp))

shards = sorted(Path(sc.resolve(cfg["paths"]["shards_dir"], cfg)).glob("*.npz"))[::40]
import pandas as pd  # noqa: E402
idx = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
plane_of = dict(zip(idx["series"].astype(str), idx["plane"]))
h0 = []
for fsh in shards:
    with np.load(fsh, allow_pickle=True) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    for sid, s in (meta.get("series") or {}).items():
        if plane_of.get(str(sid)) in ("coronal", "axial") and s.get("orientation"):
            h0.append(float(s["orientation"][0]))
check(len(h0) > 300 and min(h0) > 0, f"4. {len(h0)} coronal/axial series in {len(shards)} shards: image right = patient left in all (min h0 {min(h0):.2f})")

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
