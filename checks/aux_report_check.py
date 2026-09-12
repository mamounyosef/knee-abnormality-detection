"""
aux_report_check.py - does the auxiliary report loss actually do what the config claims?

Every step here is a way this feature can fail SILENTLY, which is the real risk: a run that
trains exactly like the baseline while its config says otherwise looks like a negative
result about report supervision rather than a wiring bug.

Runs on CPU with a tiny stand-in backbone. No GPU, no Modal, no shards.

    python checks/aux_report_check.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

ok, fail = [], []


def check(name, cond, detail=""):
    (ok if cond else fail).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (("   " + detail) if detail else ""))


cfg = sc.load_config(ROOT / "runs" / "convnext_base_report", verbose=False)
ax = cfg["training"]["aux_report"]
print("1. config")
check("aux_report.enable is true", ax["enable"] is True)
check("weight is non-zero", float(ax["weight"]) > 0, f"weight={ax['weight']}")

print("\n2. the target file")
z = np.load(sc.resolve(ax["path"], cfg), allow_pickle=True)
E, mu, sd = z["emb"].astype(np.float32), z["mean"], z["std"]
check("dim matches config", E.shape[1] == int(ax["dim"]), f"{E.shape[1]} vs {ax['dim']}")
check("no NaN or inf in the raw target", np.isfinite(E).all())
S = (E - mu) / sd
check("standardised target is finite", np.isfinite(S).all())
check("standardised sd is ~1 on train dims", abs(float(S.std()) - 1.0) < 0.35,
      f"sd={S.std():.3f}")

print("\n3. every split has a target")
import pandas as pd  # noqa: E402
have = set(map(str, z["study_ids"]))
for f in ("05_train_qwen24t-med.csv", "05_val_qwen24t-med.csv", "05_audit58.csv"):
    ids = pd.read_csv(ROOT / "data" / f).StudyInstanceUID.astype(str)
    check(f"{f}: all studies have an embedding", (~ids.isin(have)).sum() == 0,
          f"{(~ids.isin(have)).sum()} missing of {len(ids)}")

print("\n4. the model's contract")
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location(
    "aux_train", ROOT / "runs" / "convnext_base_report" / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# a 2-slice, 1-slot, 8px stand-in so this runs on a laptop CPU in a second
tiny = dict(cfg)
tiny["input"] = dict(cfg["input"], slots=["sag_plain"], n_slices=2, size=8)
tiny["model"] = dict(cfg["model"], pretrained=False)
m = mod.SliceModel(tiny)
# swap the real backbone for something tiny: we are testing WIRING, not the trunk
feat = m.head.in_features
m.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, feat))
m.per_study = 2

x = torch.randn(2, 3, 8, 8)
m.train()
out_train = m(x)
m.eval()
out_eval = m(x)
check("returns a tuple in training mode", isinstance(out_train, tuple))
check("returns plain logits under eval()", torch.is_tensor(out_eval),
      "this is what keeps validation, the ensemble and the Kaggle notebook working")
check("aux head width matches the target",
      isinstance(out_train, tuple) and out_train[1].shape[1] == int(ax["dim"]))

print("\n5. the gradient actually reaches the TRUNK")
m.train()
logits, aux_p = m(x)
tgt = torch.randn(1, int(ax["dim"]))
w = torch.ones(1, 1)
cos = torch.nn.functional.cosine_similarity(aux_p.float(), tgt.float(), dim=1)
aux_loss = ((1.0 - cos) * w.squeeze(1)).sum() / w.sum().clamp(min=1.0)
m.zero_grad()
aux_loss.backward()
g_trunk = m.backbone[1].weight.grad
g_head = m.head.weight.grad
check("auxiliary loss produces trunk gradient",
      g_trunk is not None and float(g_trunk.abs().sum()) > 0,
      f"|grad|={float(g_trunk.abs().sum()):.3e}" if g_trunk is not None else "None")
check("auxiliary loss does NOT touch the label head",
      g_head is None or float(g_head.abs().sum()) == 0,
      "the aux term must not train the 12-way classifier directly")

print("\n6. a study with no report contributes nothing")
# fresh forward: the graph above was freed by its own backward
_, aux_p0 = m(x)
cos0 = torch.nn.functional.cosine_similarity(aux_p0.float(), tgt.float(), dim=1)
w0 = torch.zeros(1, 1)
loss0 = ((1.0 - cos0) * w0.squeeze(1)).sum() / w0.sum().clamp(min=1.0)
check("zero weight gives exactly zero loss", float(loss0) == 0.0)
m.zero_grad()
loss0.backward()
g0 = m.backbone[1].weight.grad
check("zero weight gives zero trunk gradient",
      g0 is None or float(g0.abs().sum()) == 0.0)

print("\n7. the guard catches an unwired model")
plain = mod.SliceModel(dict(tiny, training={**tiny["training"],
                                            "aux_report": {"enable": False}}))
plain.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, feat))
plain.per_study = 2
try:
    sc.check_aux_wiring(tiny, plain)
    check("check_aux_wiring rejects a model with no aux head", False, "it did not raise")
except SystemExit:
    check("check_aux_wiring rejects a model with no aux head", True)

print("\n%d passed, %d failed" % (len(ok), len(fail)))
if fail:
    print("FAILED: " + ", ".join(fail))
sys.exit(1 if fail else 0)
