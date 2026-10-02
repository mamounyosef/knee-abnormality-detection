"""v3_flip_folds_check.py - are the new ConvNeXt teacher folds exactly the old folds with v3 labels and the flip?

For runs/convnext_base_25d_18slices_v3_flip_f0..f4 against their donors convnext_base_25d_18slices_f0..f4:
  1. each run loads its OWN config through its train.py; experiment.name matches the folder
  2. train / val CSVs are 21_fold{k}_*: labels equal the v3 file (17_train_full_qwen-mg-raw) cell by cell,
     same studies in the same order as the donor's 13_fold{k}_*, train and val disjoint, no audit study
  3. laterality_flip "left", patience 7, everything else equal to the donor except the expected keys:
     the local recipe (f0: same 8 studies per update) or the Modal H100 recipe (f1-f4)
  4. local runs: every training and validation study's shard is on this PC
  5. the 5 validation sets are disjoint and together cover all 4,342 non-audit studies

    python checks/v3_flip_folds_check.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
RUNS = ROOT / "runs"
sys.path.insert(0, str(RUNS))
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMMON = {"experiment.name", "paths.train_csv", "paths.val_csv", "input.laterality_flip",
          "evaluation.eval_every_steps_before",
          "training.early_stopping.patience"}
LOCAL_KEYS = {"training.grad_checkpointing", "dataloader.batch_size", "dataloader.accum_steps",
              "dataloader.val_batch_size", "dataloader.num_workers", "dataloader.val_num_workers",
              "dataloader.prefetch_factor", "dataloader.pin_memory", "output.keep_last_n",
              "output.keep_top_k", "output.checkpoint_every_steps", "output.upload_checkpoints.modal_config",
              "output.upload_checkpoints.patterns"}
MODAL_KEYS = {"modal.gpu", "modal.cpu"}
bad = 0


def check(ok, msg):
    global bad
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'}  {msg}")


def load(run):
    spec = importlib.util.spec_from_file_location(run, RUNS / run / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm.cfg, tm.sc


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


v3 = pd.read_csv(D / "17_train_full_qwen-mg-raw.csv").set_index("StudyInstanceUID")
audit = set(pd.read_csv(D / "05_audit58.csv").StudyInstanceUID)
shards = {p.stem for p in (D / "shards").glob("*.npz")}
vals = []
for k in range(5):
    run, donor = f"convnext_base_25d_18slices_v3_flip_f{k}", f"convnext_base_25d_18slices_f{k}"
    local = k == 0                     # f1, f2 moved to Modal H100 on 2 Oct (f1 resumes from local step 3500)
    print(f"\n{run}  ({'LOCAL' if local else 'Modal H100'})")
    cfg, sc = load(run)
    dcfg, _ = load(donor)
    check(cfg["experiment"]["name"] == run, "1. loads its own config")
    f, fd = flat(cfg), flat(dcfg)
    diff = {x for x in set(f) | set(fd) if f.get(x) != fd.get(x)}
    pat = 6 if k in (1, 2) else 7          # f1, f2: 6 for the Modal finish (user, 2 Oct) = the donor's
    common = COMMON - {"training.early_stopping.patience"} if pat == 6 else COMMON
    allowed = common | (LOCAL_KEYS if local else MODAL_KEYS)
    check(diff <= allowed and common <= diff, f"3. differs from {donor} only in expected keys: {sorted(diff)}")
    check(cfg["input"].get("laterality_flip") == "left", "3. knee flip on (left knees mirrored)")
    check(cfg["training"]["early_stopping"]["patience"] == pat, f"3. patience {pat}")
    check(cfg["evaluation"]["eval_every_steps_before"] == 800, "3. validates every 800 steps in epochs 1-4")
    eff = cfg["dataloader"]["batch_size"] * cfg["dataloader"]["accum_steps"]
    check(eff == dcfg["dataloader"]["batch_size"] * dcfg["dataloader"]["accum_steps"] == 8,
          f"3. {eff} studies per update, same as the donor")
    if not local:
        check(cfg["modal"]["gpu"] == "H100", f"3. Modal GPU {cfg['modal']['gpu']}, cpu {cfg['modal'].get('cpu')}")
    parts = {}
    for part in ("train", "val"):
        p = Path(sc.resolve(cfg["paths"][f"{part}_csv"], cfg))
        df = pd.read_csv(p).set_index("StudyInstanceUID")
        parts[part] = df
        check(p.name == f"21_fold{k}_{part}.csv", f"2. {part}: reads {p.name}")
        ids = list(df.index)
        dv = np.abs(df[T].values - v3.loc[ids, T].values).max()
        check(dv == 0, f"2. {part}: {len(ids)} rows, labels == v3 exactly (max |diff| {dv:.1e})")
        check(ids == list(pd.read_csv(D / f"13_fold{k}_{part}.csv").StudyInstanceUID),
              f"2. {part}: same studies, same order as the donor's 13_fold{k}_{part}")
        check(not (set(ids) & audit), f"2. {part}: no audit study")
        if local:
            miss = [s for s in ids if s not in shards]
            check(not miss, f"4. {part}: all {len(ids)} shards on this PC ({len(miss)} missing)")
    check(not (set(parts["train"].index) & set(parts["val"].index)), "2. train and val disjoint")
    vals.append(set(parts["val"].index))
union = set().union(*vals)
nonaud = {s for s in v3.index if s not in audit}
check(sum(map(len, vals)) == len(union) and union == nonaud,
      f"\n  5. the 5 val sets are disjoint and cover all {len(union)} non-audit studies")
print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
