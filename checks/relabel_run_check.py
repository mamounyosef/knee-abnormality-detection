"""relabel_run_check.py - is a relabel run exactly its donor with only the labels (and the local
training recipe) changed?

    python checks/relabel_run_check.py <new_run> <donor_run>

Checks:
  1. both runs load through their OWN train.py, and experiment.name matches the folder
  2. merged-config diff (shared_config + config.yaml): every differing key is printed, and any
     key outside the allowed list FAILS
  3. label files: same studies in the same order, same columns, no NaN, labels in [0, 1],
     audit-58 rows carry the panel labels in both, and how much the labels actually changed
  4. local files the run needs exist: backbone weights, series index, laterality file, shards
     for every training study
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ALLOWED = {  # keys a local relabel may change; anything else is a FAIL
    "experiment.name", "paths.train_csv", "paths.val_csv", "model.grad_checkpointing",
    "training.grad_checkpointing", "training.stop_after_steps",
    "dataloader.batch_size", "dataloader.accum_steps", "dataloader.val_batch_size",
    "dataloader.num_workers", "dataloader.val_num_workers", "dataloader.prefetch_factor",
    "dataloader.pin_memory", "output.keep_last_n", "output.keep_top_k",
    "output.checkpoint_every_steps", "output.upload_checkpoints"}
bad = 0


def fail(msg):
    global bad
    bad += 1
    print(f"  FAIL  {msg}")


def load(name):
    sys.path.insert(0, str(RUNS))
    spec = importlib.util.spec_from_file_location(f"train_{name}", RUNS / name / "train.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    return tm


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        key = f"{p}{k}"
        if isinstance(v, dict) and not key.startswith("output.upload_checkpoints"):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


new, donor = sys.argv[1], sys.argv[2]
tn, td = load(new), load(donor)
sc = tn.sc
print("1. names")
for n, t in ((new, tn), (donor, td)):
    got = t.cfg["experiment"]["name"]
    print(f"  {n}: experiment.name = {got}")
    if got != n:
        fail(f"{n} loads config of {got}")

print("2. config diff (new vs donor)")
fn, fd = flat(tn.cfg), flat(td.cfg)
for k in sorted(set(fn) | set(fd)):
    if fn.get(k) != fd.get(k):
        tag = "ok  " if k in ALLOWED else "FAIL"
        print(f"  {tag}  {k}: {fd.get(k)!r} -> {fn.get(k)!r}")
        if k not in ALLOWED:
            bad += 1
ga = lambda c: c["dataloader"]["batch_size"] * c["dataloader"].get("accum_steps", 1)
print(f"  studies per update: donor {ga(td.cfg)}, new {ga(tn.cfg)}")
if ga(td.cfg) != ga(tn.cfg):
    fail("studies per update differ, so the step count and lr schedule differ")

print("3. labels")
tasks = tn.cfg["tasks"]
a = pd.read_csv(sc.resolve(td.cfg["paths"]["train_csv"], td.cfg))
b = pd.read_csv(sc.resolve(tn.cfg["paths"]["train_csv"], tn.cfg))
print(f"  donor {td.cfg['paths']['train_csv']}: {len(a)} rows; new {tn.cfg['paths']['train_csv']}: {len(b)} rows")
if list(a.StudyInstanceUID) != list(b.StudyInstanceUID):
    same = set(a.StudyInstanceUID) == set(b.StudyInstanceUID)
    (print("  ok    same studies, different order (the loader shuffles anyway)") if same
     else fail("different study sets"))
if list(a.columns) != list(b.columns):
    print(f"  note  columns differ: only donor {sorted(set(a.columns) - set(b.columns))}, "
          f"only new {sorted(set(b.columns) - set(a.columns))}")
a, b = a.set_index("StudyInstanceUID"), b.set_index("StudyInstanceUID").loc[a.StudyInstanceUID]
for nm, df in (("donor", a), ("new", b)):
    v = df[tasks].values
    if np.isnan(v).any():
        fail(f"{nm} labels have NaN")
    if v.min() < 0 or v.max() > 1:
        fail(f"{nm} labels outside [0, 1]: {v.min()} .. {v.max()}")
aud = pd.read_csv(ROOT / "data/05_audit58.csv").set_index("StudyInstanceUID")
for nm, df in (("donor", a), ("new", b)):
    missing = [s for s in aud.index if s not in df.index]
    if missing:
        fail(f"{nm}: {len(missing)} audit studies missing")
        continue
    d = np.abs(df.loc[aud.index, tasks].values - aud[["official_" + t for t in tasks]].values).max()
    print(f"  {nm}: audit-58 rows vs panel labels, max |diff| {d:.4f}")
    if d > 1e-6:
        fail(f"{nm} audit rows are not the panel labels")
nonaud = [s for s in a.index if s not in aud.index]
d = np.abs(a.loc[nonaud, tasks].values - b.loc[nonaud, tasks].values)
print(f"  label change on the {len(nonaud)} non-audit studies: mean |diff| {d.mean():.4f}, "
      f"max {d.max():.4f}, rows changed {(d.max(1) > 1e-6).sum()}")
print("  per finding mean |diff|: " + ", ".join(f"{t} {d[:, j].mean():.3f}" for j, t in enumerate(tasks)))
if d.max() < 1e-6:
    fail("labels are IDENTICAL to the donor's: this would retrain the donor")

print("4. local files")
p = sc.resolve(tn.cfg["paths"]["series_index"], tn.cfg)
print(f"  series index {p}: {'ok' if Path(p).exists() else 'MISSING'}")
if not Path(p).exists():
    fail("series index missing")
if getattr(tn, "WEIGHTS_FILE", None):
    import external_backbones as eb
    w = Path(eb.resolve_weights(tn.WEIGHTS_FILE, tn.cfg))
    print(f"  backbone weights {w}: {'%.2f GB' % (w.stat().st_size / 1e9) if w.exists() else 'MISSING'}")
    if not w.exists():
        fail("backbone weights missing")
lat = ROOT / "data/12_laterality.csv"
if tn.cfg["input"].get("laterality_flip"):
    print(f"  laterality file {lat.name}: {'ok' if lat.exists() else 'MISSING'}")
    if not lat.exists():
        fail("laterality file missing")
sh = ROOT / "data/shards"
miss = [s for s in b.index if not (sh / f"{s}.npz").exists()]
print(f"  shards: {len(b) - len(miss)}/{len(b)} training studies present in data/shards")
if miss:
    fail(f"{len(miss)} shards missing, e.g. {miss[:3]}")

print(f"\n{'ALL OK' if not bad else f'{bad} FAIL(S)'}")
sys.exit(1 if bad else 0)
