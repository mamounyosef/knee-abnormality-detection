"""batch1_relabel_check.py - are the batch-1 relabel runs exactly their donors with only the planned changes,
on the right labels and split, and do they build and load?

Runs (2-3 Oct):
  dinov3_convnext_base_v5                       Modal  donor dinov3_convnext_base
  convnext_base_v5   Modal  donor convnext_base_full_all_knees_same_orientation
  convnext_base_25d_poolattn_v5                 LOCAL  donor convnext_base_25d_poolattn
  convnext_base_25d_full_v5   LOCAL  donor ..._fulldata (the shipped member)

  1. each loads its OWN config through its train.py
  2. it differs from its donor in EXACTLY the expected keys
  3. v5 runs: train 23_train_w50.csv (v4 labels), val 23_val_v3_audit.csv, flip left, early stopping on,
     patience 6, audit reported; the local run: all 4,400 studies on v4 labels, no validation, 12 epochs,
     8 studies per update
  4. label files: every v5 train row equals the v4 (w50) label file; the full-data file is the v4 file itself
     and its 58 audit rows carry the panel labels
  5. the model builds (no pretrained download) and one real batch from the run's own loader goes through it

    python checks/batch1_relabel_check.py                 # all runs, with the model + batch test (heavy on CPU)
    python checks/batch1_relabel_check.py --light <run>   # config and label checks only, one run
"""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
D = ROOT / "data"
sys.path.insert(0, str(RUNS))
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
V5 = {"experiment.name", "paths.train_csv", "paths.val_csv"}
LOCAL = {"training.grad_checkpointing", "dataloader.batch_size", "dataloader.accum_steps", "dataloader.val_batch_size",
         "dataloader.num_workers", "dataloader.val_num_workers", "dataloader.prefetch_factor", "dataloader.pin_memory",
         "output.keep_last_n", "output.keep_top_k", "output.upload_checkpoints.modal_config",
         "output.upload_checkpoints.patterns"}
RUNS_EXPECTED = {
    "dinov3_convnext_base_v5": ("dinov3_convnext_base",
        V5 | {"input.laterality_flip", "training.early_stopping.patience", "training.extend.enable"}),
    "convnext_base_v5": ("convnext_base_full_all_knees_same_orientation",
        V5 | {"training.epochs", "training.scheduler.schedule_epochs", "training.early_stopping.enable",
              "training.early_stopping.patience", "evaluation.enable", "evaluation.eval_every_steps_before",
              "evaluation.eval_every_steps_after", "evaluation.audit.enable"}),
    "convnext_base_25d_poolattn_v5": ("convnext_base_25d_poolattn",          # LOCAL since 3 Oct (user)
        V5 | {"input.laterality_flip", "training.early_stopping.patience", "modal.gpu", "modal.cpu"}),   # local to step 4100, then Modal
    "convnext_base_25d_full_v5": ("convnext_base_25d_all_knees_same_orientation_fulldata",
        {"experiment.name", "paths.train_csv", "training.grad_checkpointing", "dataloader.batch_size",
         "dataloader.accum_steps", "dataloader.val_batch_size", "dataloader.num_workers", "dataloader.val_num_workers",
         "dataloader.prefetch_factor", "dataloader.pin_memory", "output.keep_last_n", "output.keep_top_k",
         "output.upload_checkpoints.modal_config", "output.upload_checkpoints.patterns",
         "output.checkpoint_every_steps", "output.upload_checkpoints.volume"}),   # 4 Oct: save every 50, upload to backup
    "convnext_base_25d_perfinding_v5": ("convnext_base_25d_perfinding_slots_and_pooling_fulldata_flip",   # batch 3, LOCAL
        V5 | {"training.grad_checkpointing", "training.epochs", "training.early_stopping.enable",
              "training.early_stopping.patience", "evaluation.enable", "dataloader.batch_size", "dataloader.accum_steps",
              "dataloader.val_batch_size", "dataloader.num_workers", "dataloader.val_num_workers",
              "dataloader.prefetch_factor", "dataloader.pin_memory", "output.keep_last_n", "output.keep_top_k",
              "output.checkpoint_every_steps", "output.upload_checkpoints.modal_config",
              "output.upload_checkpoints.volume", "output.upload_checkpoints.patterns"}),
    "mricore_vitb_mae_cross_attention_v5": ("mricore_vitb_mae_cross_attention",            # batch 2
        V5 | {"input.laterality_flip", "training.early_stopping.patience", "modal.gpu"}),
    "mricore_vitb_cross_attention_v5": ("mricore_vitb_cross_attention",                    # batch 2
        V5 | {"input.laterality_flip", "training.early_stopping.patience", "modal.gpu"}),
}
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


w50 = pd.read_csv(D / "22_train_full_qwen-mg-convnext-mii-w50.csv").set_index("StudyInstanceUID")
aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")

LIGHT = "--light" in sys.argv
ONLY = [a for a in sys.argv[1:] if not a.startswith("--")]
for run, (donor, expected) in RUNS_EXPECTED.items():
    if ONLY and run not in ONLY:
        continue
    print(f"\n{run}   (donor {donor})")
    m = load(run)
    cfg, sc = m.cfg, m.sc
    check(cfg["experiment"]["name"] == run and m.EXP_NAME == run, "1. loads its own config, EXP_NAME matches")
    f, fd = flat(cfg), flat(load(donor).cfg)
    diff = {k for k in set(f) | set(fd) if f.get(k) != fd.get(k)}
    check(diff == expected, f"2. differs from the donor in exactly the expected keys"
                            + ("" if diff == expected else f": extra {sorted(diff - expected)}, missing {sorted(expected - diff)}"))
    local = run.endswith("_full_v5")
    tr = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
    if not local:
        check(Path(cfg["paths"]["train_csv"]).name == "23_train_w50.csv" and
              Path(cfg["paths"]["val_csv"]).name == "23_val_v3_audit.csv", "3. v5 split files")
        check(cfg["input"].get("laterality_flip") == "left", "3. knee flip on")
        es = cfg["training"]["early_stopping"]
        check(es["enable"] and es["patience"] == 6, "3. early stopping on, patience 6")
        check(cfg["evaluation"].get("enable", True) and cfg["evaluation"]["audit"]["enable"], "3. validates, audit-58 reported")
        check(not (cfg["training"].get("extend") or {}).get("enable"), "3. no extension")
        ids = list(tr.StudyInstanceUID)
        check(len(ids) == 3996 and np.allclose(tr[T].values, w50.loc[ids, T].values), "4. 3,996 train rows == the v4 label file")
    else:
        check(Path(cfg["paths"]["train_csv"]).name == "22_train_full_qwen-mg-convnext-mii-w50.csv", "3. the v4 full-data file")
        check(len(tr) == 4400 and np.allclose(tr[T].values, w50.loc[tr.StudyInstanceUID, T].values), "4. all 4,400 rows, v4 labels")
        a = tr.set_index("StudyInstanceUID").loc[aud.index, T].values
        check(np.array_equal(a, aud[["official_" + t for t in T]].values.astype(float)), "4. the 58 audit rows carry the panel labels")
        check(cfg["evaluation"].get("enable") is False and not cfg["training"]["early_stopping"]["enable"],
              "3. no validation, no early stopping (nothing held out)")
        check(cfg["training"]["epochs"] == 12, "3. fixed 12 epochs (its v5 twin peaked at the end of 12)")
        check(cfg["input"].get("laterality_flip") == "left", "3. knee flip on")
    eff = cfg["dataloader"]["batch_size"] * cfg["dataloader"]["accum_steps"]
    check(eff == 8, f"3. {eff} studies per update")
    if LIGHT:
        continue
    # 5. build + one real batch through the model
    c2 = copy.deepcopy(cfg)
    c2["model"]["pretrained"] = False
    c2["dataloader"].update(num_workers=0, val_num_workers=0, batch_size=1, val_batch_size=1,
                            persistent_workers=False, pin_memory=False)
    model = m.build_model(c2).eval()
    L = sc.make_loaders(c2)
    xb, yb = next(iter(L["train"]))[:2]
    with torch.no_grad():
        out = model(sc.prepare_batch(xb, c2, torch.device("cpu")))
    check(tuple(out.shape) == (1, 12) and torch.isfinite(out).all().item(),
          f"5. model builds and a real batch {tuple(xb.shape)} gives logits {tuple(out.shape)}")
    check(L["train"].dataset.flip_side == "left", "5. the loader mirrors left knees")

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
