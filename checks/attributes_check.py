"""attributes_check.py - is convnext_base_25d_v5_attributes its parent plus the attribute head, and is the engine wiring right?

  1. loads its OWN config; differs from convnext_base_25d_18slices_v5_aug_base in exactly the expected keys
  2. the attribute file: K columns, values within each attribute's grade range, one row per study
  3. the dataset returns (x, y, [mask(K), value(K)]) for training and (x, y) for validation; values in [0, 1]
  4. the model returns (logits[12], aux[K]) in training mode and plain logits under eval(); check_aux_wiring passes
  5. the engine's masked loss equals a hand computation, and silent cells change nothing

Uses data/25_report_attributes.csv when it exists; otherwise a temporary stand-in with random grades (so the wiring can
be tested before the extraction runs), clearly reported as such.

    python checks/attributes_check.py
"""
import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RUNS, D = ROOT / "runs", ROOT / "data"
sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(ROOT / "data_code"))
RUN, PARENT = "convnext_base_25d_v5_attributes", "convnext_base_25d_18slices_v5_aug_base"
EXPECTED = {"experiment.name", "training.aux_attributes.enable"}
bad = 0


def check(ok, msg):
    global bad
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'}  {msg}")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
    return out


ex = load(ROOT / "data_code" / "25_extract_report_attributes.py", "attrx")
real = D / "25_report_attributes.csv"
if real.exists():
    path, label = real, "the REAL extraction"
else:
    tr = pd.read_csv(D / "23_train_w50.csv").StudyInstanceUID
    rng = np.random.default_rng(0)
    df = pd.DataFrame({k: rng.integers(0, m + 1, len(tr)).astype(float) for k, (m, _) in ex.ATTRS.items()}, index=tr)
    df = df.mask(rng.random(df.shape) < 0.5)              # half the cells silent
    df.insert(0, "bad", 0)
    df.insert(0, "model", "stand-in")
    path = Path(tempfile.gettempdir()) / "25_report_attributes_standin.csv"
    df.rename_axis("StudyInstanceUID").to_csv(path)
    label = "a TEMPORARY random stand-in (the extraction has not been built yet)"
print(f"{RUN}  (attribute file: {label})")

m, pm = load(RUNS / RUN / "train.py", RUN), load(RUNS / PARENT / "train.py", PARENT)
cfg, sc = m.cfg, m.sc
check(cfg["experiment"]["name"] == RUN and m.EXP_NAME == RUN, "1. loads its own config, EXP_NAME matches")
f, fp = flat(cfg), flat(pm.cfg)
diff = {k for k in set(f) | set(fp) if f.get(k) != fp.get(k)}
check(diff == EXPECTED and cfg["training"]["aux_attributes"]["weight"] == 0.3 and cfg["training"]["epochs"] == 12,
      "1. differs from the parent only by aux_attributes on (weight 0.3), 12 epochs"
      + ("" if diff == EXPECTED else f": extra {sorted(diff - EXPECTED)}, missing {sorted(EXPECTED - diff)}"))

c2 = copy.deepcopy(cfg)
c2["training"]["aux_attributes"]["path"] = str(path)
c2["model"]["pretrained"] = False
K = sc.aux_attribute_count(c2)
A = pd.read_csv(path).set_index("StudyInstanceUID")
rng_ok = all(A[k].dropna().between(0, mx).all() for k, (mx, _) in ex.ATTRS.items())
check(K == len(ex.ATTRS) and rng_ok and A.index.is_unique, f"2. {K} attributes, all within their grade ranges, unique studies")

c2["dataloader"].update(num_workers=0, val_num_workers=0, batch_size=2, val_batch_size=1,
                        persistent_workers=False, pin_memory=False)
L = sc.make_loaders(c2)
b = next(iter(L["train"]))
vb = next(iter(L["val"]))
check(len(b) == 3 and tuple(b[2].shape) == (2, 2 * K) and len(vb) == 2,
      f"3. train batch (x, y, aux {tuple(b[2].shape)}), val batch (x, y)")
msk, val = b[2][:, :K], b[2][:, K:]
check(set(msk.unique().tolist()) <= {0.0, 1.0} and float(val.min()) >= 0 and float(val.max()) <= 1
      and float((val * (1 - msk)).abs().sum()) == 0, "3. mask is 0/1, values in [0, 1], silent cells are 0")

model = m.build_model(c2)
model.train()
x = sc.prepare_batch(b[0], c2, torch.device("cpu"))
with torch.no_grad():
    out = model(x)
model.eval()
with torch.no_grad():
    out_e = model(x)
check(isinstance(out, tuple) and tuple(out[0].shape) == (2, 12) and tuple(out[1].shape) == (2, K)
      and torch.is_tensor(out_e) and tuple(out_e.shape) == (2, 12), "4. (logits, aux) in training, plain logits in eval")
try:
    sc.check_aux_wiring(c2, model)
    check(True, "4. the engine's check_aux_wiring passes")
except SystemExit as e:
    check(False, f"4. check_aux_wiring: {e}")

# 5. the loss exactly as the engine computes it, vs by hand
aux_p = out[1]
bce = F.binary_cross_entropy_with_logits(aux_p, val, reduction="none")
eng = (bce * msk).sum() / msk.sum().clamp(min=1.0)
hand = np.mean([float(F.binary_cross_entropy_with_logits(aux_p[i, j], val[i, j]))
                for i in range(2) for j in range(K) if msk[i, j] == 1])
v2 = val.clone()
v2[msk == 0] = 0.77                                       # change only the silent cells
eng2 = (F.binary_cross_entropy_with_logits(aux_p, v2, reduction="none") * msk).sum() / msk.sum().clamp(min=1.0)
check(abs(float(eng) - hand) < 1e-5 and abs(float(eng2) - float(eng)) < 1e-7,
      f"5. masked loss {float(eng):.4f} = hand {hand:.4f}; silent cells do not change it")
src = (RUNS / "shared_code.py").read_text(encoding="utf-8")
check("aux_attr_bce" in src and "and aux_attr:" in src, "5. the engine logs aux_attr_bce / _shuf / _gap and uses the masked branch")

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}" + ("" if real.exists() else "  (re-run after --build for the real file)"))
sys.exit(1 if bad else 0)
