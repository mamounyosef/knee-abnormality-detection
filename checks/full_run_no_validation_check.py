"""full_run_no_validation_check.py - prove a full-data run checkpoints WITHOUT validating.

convnext_base_full_all_knees_same_orientation trains on train + val + audit, so any
"validation" it prints is in-sample: it climbs whatever the model does and reads like a
score when it is not one. `evaluation.enable: false` is supposed to skip the evaluation and
KEEP the checkpointing, which the engine otherwise couples to it.

Both halves of that matter and neither is obvious from reading the config:
  - if the guard is in the wrong place, validations still print
  - if the guard returns too early, the run produces NO checkpoints at all, which would be
    discovered only after eight epochs of GPU time

So run the real engine for a handful of steps on a deliberately tiny geometry, capture
stdout, and check both.

Run:  python checks/full_run_no_validation_check.py
"""
import contextlib
import io as _io
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RUN = "convnext_base_full_all_knees_same_orientation"

sys.path.insert(0, str(RUNS))
sys.path.insert(0, str(RUNS / RUN))
import shared_code as sc  # noqa: E402

PASS, FAIL = [], []


def ck(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %s %-58s %s" % ("PASS" if ok else "FAIL", name, detail))


cfg = sc.load_config(RUNS / RUN, verbose=False)
print("\n1. the config says what it should")
ck("evaluation.enable is false", cfg["evaluation"]["enable"] is False)
ck("audit is off (those 58 studies are in the training set)",
   cfg["evaluation"]["audit"]["enable"] is False)
ck("early stopping is off (nothing held out to stop on)",
   cfg["training"]["early_stopping"]["enable"] is False)
ck("laterality_flip is 'left'", cfg["input"]["laterality_flip"] == "left",
   repr(cfg["input"]["laterality_flip"]))
ck("trains on the full split", cfg["paths"]["train_csv"] == "08_train_full.csv",
   cfg["paths"]["train_csv"])

print("\n2. the engine's guard reads the flag before it evaluates")
src = (RUNS / "shared_code.py").read_text(encoding="utf-8")
i_guard = src.find('cfg["evaluation"].get("enable", True)')
i_eval = src.find('vm = evaluate(model, loaders["val"]')
ck("the guard exists", i_guard > 0)
ck("and it comes BEFORE the evaluate() call", 0 < i_guard < i_eval,
   "guard at %d, evaluate at %d" % (i_guard, i_eval))
ck("the guard still writes last.pt", 'ckpt_dir / "last.pt"' in src[i_guard:i_eval])
ck("the guard still writes best.pt",
   'cfg["output"]["best_name"]' in src[i_guard:i_eval])

print("\n3. a real run of the real engine, tiny geometry, a few steps")
shards = Path(cfg["paths"]["shards_dir"])
df = pd.read_csv(sc.resolve(cfg["paths"]["train_csv"], cfg))
have = [s for s in df["StudyInstanceUID"] if (shards / ("%s.npz" % s)).exists()]
ck("local shards available", len(have) >= 2, "%d studies" % len(have))
if len(have) < 2:
    print("\n  cannot run the engine without shards")
    raise SystemExit(1)

tmp = Path(tempfile.mkdtemp())
sub = df[df["StudyInstanceUID"].isin(have)].reset_index(drop=True)
sub.to_csv(tmp / "tiny_train.csv", index=False)

t = dict(cfg)
t["paths"] = dict(cfg["paths"], train_csv=str(tmp / "tiny_train.csv"),
                  val_csv=str(tmp / "tiny_train.csv"))
t["input"] = dict(cfg["input"], size=96, n_slices=2)
t["training"] = dict(cfg["training"], epochs=1, amp=False)
t["dataloader"] = dict(cfg["dataloader"], batch_size=1, accum_steps=1, num_workers=0,
                       persistent_workers=False, prefetch_factor=None, val_batch_size=1)
t["output"] = dict(cfg["output"], run_dir=str(tmp / "results"))
t["evaluation"] = dict(cfg["evaluation"], eval_every_steps_before=1,
                       eval_every_steps_after=1)
t["experiment"] = dict(cfg["experiment"], name="tiny_full_check")
t["wandb"] = dict(cfg.get("wandb", {}), enabled=False)

import importlib  # noqa: E402
tm = importlib.import_module("train")
buf = _io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        sc.run_experiment(t, tm.build_model(t), RUNS / RUN)
    err = None
except Exception as e:                       # noqa: BLE001
    err = e
out = buf.getvalue()
if err is not None:
    print(out[-1500:])
    ck("the engine ran", False, "%s: %s" % (type(err).__name__, str(err)[:80]))
else:
    ck("the engine ran", True, "%d lines of output" % len(out.splitlines()))
    ck("NO validation was printed", "VALIDATION (" not in out,
       "found it" if "VALIDATION (" in out else "clean")
    ck("no VAL metric block was printed", "\U0001F50E" not in out or "VALIDATION (" not in out)
    ck("it said why it was skipping", "no validation, this run trains on every study" in out)
    res = Path(t["output"]["run_dir"])
    cks = res / cfg["output"]["checkpoints_dir"]
    got = sorted(p.name for p in cks.glob("*.pt")) if cks.exists() else []
    ck("last.pt was still written", "last.pt" in got, str(got))
    ck("best.pt was still written", cfg["output"]["best_name"] in got, str(got))
    vlog = res / "val_log.csv"
    rows = len(pd.read_csv(vlog)) if vlog.exists() and vlog.stat().st_size > 2 else 0
    ck("val_log.csv has no rows", rows == 0, "%d rows" % rows)
    preds = res / cfg["output"].get("predictions_dir", "predictions")
    npz = list(preds.glob("*.npz")) if preds.exists() else []
    ck("no in-sample predictions were saved", not npz, "%d files" % len(npz))

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + "=" * 72)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
