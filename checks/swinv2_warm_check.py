"""swinv2_warm_check.py - is swinv2_base_in22k_v5_warm exactly its donor plus the planned changes, and does it load?

  1. loads its OWN config through its train.py
  2. differs from swinv2_base_in22k in EXACTLY the expected keys
  3. train = data/24 (4,279 on v4 labels), val = the 121 clean studies (63 v3 + 58 panel), none seen by the donor
  4. the donor's best.pt loads into the new model with NO missing or unexpected tensors (strict warm start)
  5. one real study from the run's own train loader, knee flip on (no forward pass: light on the CPU)

    python checks/swinv2_warm_check.py
"""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS, D = ROOT / "runs", ROOT / "data"
sys.path.insert(0, str(RUNS))
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RUN, DONOR = "swinv2_base_in22k_v5_warm", "swinv2_base_in22k"
EXPECTED = {"experiment.name", "paths.train_csv", "paths.val_csv", "model.init_from", "input.laterality_flip",
            "training.epochs", "training.optimizer.lr", "training.scheduler.warmup_frac",
            "training.early_stopping.patience", "modal.timeout_hours", "evaluation.eval_every_steps_before",
            "evaluation.eval_every_steps_mid",   # _after: shared default is already 100
            "evaluation.validate_before_training"}
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


m = load(RUN)
cfg, sc = m.cfg, m.sc
print(RUN)
check(cfg["experiment"]["name"] == RUN and m.EXP_NAME == RUN, "1. loads its own config, EXP_NAME matches")
f, fd = flat(cfg), flat(load(DONOR).cfg)
diff = {k for k in set(f) | set(fd) if f.get(k) != fd.get(k)}
check(diff == EXPECTED, "2. differs from the donor in exactly the expected keys"
      + ("" if diff == EXPECTED else f": extra {sorted(diff - EXPECTED)}, missing {sorted(EXPECTED - diff)}"))

w50 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
v5val = pd.read_csv(D / "23_val_v3_audit.csv").set_index("StudyInstanceUID")
old = set(pd.read_csv(D / "05_train_qwen24t-med.csv").StudyInstanceUID)
tr = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
va = pd.read_csv(sc.resolve(cfg["paths"]["val_csv"], cfg))
check(len(tr) == 4279 and np.allclose(tr[T].values, w50.loc[tr.StudyInstanceUID, T].values), "3. 4,279 train rows == v4 labels")
check(len(va) == 121 and (va.source == "audit").sum() == 58
      and np.allclose(va[T].values, v5val.loc[va.StudyInstanceUID, T].values), "3. 121 val rows == the v5 val's own labels")
check(not set(va.StudyInstanceUID) & old and not set(va.StudyInstanceUID) & set(tr.StudyInstanceUID),
      "3. no val study in the donor's training set or in this train set")
check(cfg["input"].get("laterality_flip") == "left", "3. knee flip on")
es = cfg["training"]["early_stopping"]
check(es["enable"] and es["patience"] == 7, "3. early stopping on, patience 7 validations")
check(not cfg["input"].get("slices_as_channels") and cfg["input"].get("channels") == 3
      and cfg["input"]["n_slices"] == fd["input.n_slices"],
      "3. 2D like the donor: each slice its own image, grey copied to 3 channels (no 2.5D)")
ev = cfg["evaluation"]
check(ev["eval_every_steps_before"] == ev["eval_every_steps_mid"] == ev["eval_every_steps_after"] == 100
      and ev["validate_before_training"] is True, "3. validates every 100 steps, and at step 0")
eff = cfg["dataloader"]["batch_size"] * cfg["dataloader"]["accum_steps"]
check(eff == 8, f"3. {eff} studies per update")

c2 = copy.deepcopy(cfg)
c2["model"]["pretrained"] = False
model = m.build_model(c2)
src = sc.resolve_init_from(cfg["model"]["init_from"], RUNS / RUN, cfg)
ck = torch.load(src, map_location="cpu", weights_only=False)
res = sc._load_model_state(model, ck.get("model", ck))
check(not res.missing_keys and not res.unexpected_keys,
      f"4. donor {src.name} (epoch {ck.get('epoch')}, score {ck.get('score')}) loads strictly")
del model, ck

c2["dataloader"].update(num_workers=0, val_num_workers=0, batch_size=1, val_batch_size=1,
                        persistent_workers=False, pin_memory=False)
L = sc.make_loaders(c2)
xb = next(iter(L["train"]))[0]
check(L["train"].dataset.flip_side == "left" and torch.isfinite(xb).all().item(),
      f"5. a real study {tuple(xb.shape)} from the train loader, left knees mirrored")
check(len(L["val"].dataset) == 121, "5. the val loader holds the 121")

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
