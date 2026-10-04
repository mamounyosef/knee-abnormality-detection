"""specialists_v5_check.py - are the three v5 specialists exactly their parent plus the planned changes, and do they load?

Parent: convnext_base_25d_18slices_v5_aug_base (top_step5900). Runs: convnext_base_25d_v5_spec_{lateral,pfoa,fluid,trauma}.
  1. loads its OWN config through its train.py
  2. differs from the parent in EXACTLY the expected keys
  3. its findings and slots are the planned ones; v5 split, flip, early stopping, 8 studies per update
  4. the branch seed: every backbone tensor equals the parent's, and the head rows equal the parent's rows for
     exactly these findings, in this order
  5. one real study from its own train loader: its slots, 18 slices, left knees mirrored, targets = its findings'
     v4 labels (no forward pass: light on the CPU)

    python checks/specialists_v5_check.py
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
ALL = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PARENT, PARENT_CK = "convnext_base_25d_18slices_v5_aug_base", "top_step5900.pt"
PLAN = {
    "convnext_base_25d_v5_spec_lateral": (["Lateral Meniscus", "Lateral OA"], ["cor_fluid", "cor_plain", "sag_fluid", "sag_plain"]),
    "convnext_base_25d_v5_spec_pfoa": (["PF OA"], ["ax_fluid", "ax_plain", "sag_plain", "sag_fluid"]),
    "convnext_base_25d_v5_spec_fluid": (["Effusion", "Synovitis", "Baker's"], ["ax_fluid", "sag_fluid", "cor_fluid"]),
    "convnext_base_25d_v5_spec_trauma": (["Fracture", "Contusion"],
                                         ["sag_plain", "sag_fluid", "cor_plain", "cor_fluid", "ax_plain", "ax_fluid"]),
}
# six slots do not fit 8 studies at once: batch 4 x accum 2 (still 8 per update)
EXTRA = {"convnext_base_25d_v5_spec_trauma": {"dataloader.batch_size", "dataloader.accum_steps", "dataloader.val_batch_size"}}
EXPECTED = {"experiment.name", "input.slots", "tasks", "training.epochs", "training.optimizer.lr",
            "evaluation.eval_every_steps_before", "evaluation.eval_every_steps_mid", "evaluation.validate_before_training"}
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


pm = load(PARENT)
fp = flat(pm.cfg)
psd = torch.load(RUNS / PARENT / "results" / "checkpoints" / PARENT_CK, map_location="cpu", weights_only=False)["model"]
w50 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
for run, (tasks, slots) in PLAN.items():
    print(f"\n{run}")
    m = load(run)
    cfg, sc = m.cfg, m.sc
    check(cfg["experiment"]["name"] == run and m.EXP_NAME == run, "1. loads its own config, EXP_NAME matches")
    f = flat(cfg)
    diff = {k for k in set(f) | set(fp) if f.get(k) != fp.get(k)}
    exp = EXPECTED | EXTRA.get(run, set())
    check(diff == exp, "2. differs from the parent in exactly the expected keys"
          + ("" if diff == exp else f": extra {sorted(diff - exp)}, missing {sorted(exp - diff)}"))
    check(cfg["tasks"] == tasks and cfg["input"]["slots"] == slots, f"3. findings {tasks}, slots {slots}")
    check(Path(cfg["paths"]["train_csv"]).name == "23_train_w50.csv" and Path(cfg["paths"]["val_csv"]).name == "23_val_v3_audit.csv",
          "3. v5 split (train v4 labels, val 346 v3 + 58 audit)")
    check(cfg["input"].get("laterality_flip") == "left", "3. knee flip on")
    es = cfg["training"]["early_stopping"]
    check(es["enable"] and es["patience"] == 6, "3. early stopping on, patience 6 validations")
    check(cfg["training"]["epochs"] == 4 and cfg["training"]["optimizer"]["lr"] == 2.0e-5, "3. 4 epochs, lr 2e-5")
    check(cfg["dataloader"]["batch_size"] * cfg["dataloader"]["accum_steps"] == 8, "3. 8 studies per update")
    ev = cfg["evaluation"]
    check(ev["eval_every_steps_before"] == ev["eval_every_steps_mid"] == ev["eval_every_steps_after"] == 100
          and ev["validate_before_training"] is True, "3. validates every 100 steps, and at step 0")

    c2 = copy.deepcopy(cfg)
    c2["model"]["pretrained"] = False
    model = m.build_model(c2)
    sd = model.state_dict()
    rows = [ALL.index(t) for t in tasks]
    trunk_same = all(torch.equal(sd[k], psd[k]) for k in sd if not k.startswith("head."))
    head_same = (torch.equal(sd["head.weight"], psd["head.weight"][rows])
                 and torch.equal(sd["head.bias"], psd["head.bias"][rows]))
    check(trunk_same and len(sd) == len(psd), f"4. all {len(sd) - 2} backbone tensors equal the parent's")
    check(head_same and tuple(sd["head.weight"].shape) == (len(tasks), 1024), f"4. head = parent rows {rows}")
    del model

    c2["dataloader"].update(num_workers=0, val_num_workers=0, batch_size=1, val_batch_size=1,
                            persistent_workers=False, pin_memory=False)
    L = sc.make_loaders(c2)
    b = next(iter(L["train"]))
    xb, yb = b[0], b[1]
    sid = L["train"].dataset.df.StudyInstanceUID.iloc[0] if not getattr(L["train"], "shuffle", False) else None
    check(tuple(xb.shape)[1:3] == (len(slots), 18) and torch.isfinite(xb).all().item(),
          f"5. a real study {tuple(xb.shape)}: {len(slots)} slots x 18 slices")
    check(tuple(yb.shape) == (1, len(tasks)), f"5. targets {tuple(yb.shape)}: one per finding")
    check(L["train"].dataset.flip_side == "left", "5. the loader mirrors left knees")
    ds = L["train"].dataset
    ycols = ds.df[tasks].values if hasattr(ds, "df") else None
    if ycols is not None:
        check(np.allclose(ycols, w50.loc[ds.df.StudyInstanceUID, tasks].values), "5. train targets == v4 labels for these findings")

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
