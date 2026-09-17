r"""submission_guards_check.py - do the submission guards actually fire?

Every guard here exists because the failure it catches already happened, and each one cost a
kernel run or shipped something the record misdescribes. A guard that silently stops working
is worse than none, because the whole point is that nobody is watching at 2am.

  1. DROPPED UPLOAD. kaggle printed "Upload successful: m2_model.pt (357MB)", dropped it
     during ingest, and `_wait_for_dataset` accepted the dataset because the string
     "model.pt" appeared - it was matching m0. The notebook then died on a missing file.
  2. STALE STAGING. The build directory was only ever mkdir(exist_ok=True), so an 8-member
     build inherited m8_model.pt and m9_model.pt from a previous 12-member one.
  3. STALE NOTE. Submission 56234763 shipped the 18-slice member while its public
     description named the 24-slice one. Kaggle descriptions cannot be edited afterwards.
  4. DEAD MEMBER. A member carrying zero weight on all 12 findings is shipped, paid for in
     runtime, and changes nothing.

No network, no Kaggle account needed.

    python checks/submission_guards_check.py
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print("  %-62s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


# --- 1. the dataset readiness check must name every file ---------------------------
print("\n1. dropped-upload guard (_wait_for_dataset)")
listing = {"full": "m0_model.pt m1_model.pt m2_model.pt",
           "missing_m2": "m0_model.pt m1_model.pt m2_config.json"}
state = {"mode": "missing_m2"}


calls = []


def fake_kaggle(*args, **kw):
    calls.append(args)
    return listing[state["mode"]]


real_kaggle, real_sleep = sc._kaggle, None
import time as _time
real_sleep = _time.sleep
sc._kaggle = fake_kaggle
_time.sleep = lambda s: None
try:
    sc._wait_for_dataset("u", "s", tries=2, secs=1, expect_models=3)
    check("refuses to continue when m2_model.pt never lands", False, "it returned happily")
except SystemExit as e:
    check("refuses to continue when m2_model.pt never lands", "m2_model.pt" in str(e))

state["mode"] = "full"
try:
    sc._wait_for_dataset("u", "s", tries=2, secs=1, expect_models=3)
    check("accepts a complete dataset", True)
except SystemExit as e:
    check("accepts a complete dataset", False, str(e)[:80])

# the exact bug: the old check passed on any "model.pt", so prove the new one does not
state["mode"] = "missing_m2"
try:
    sc._wait_for_dataset("u", "s", tries=1, secs=1, expect_models=3)
    check("the OLD 'any model.pt' behaviour is gone", False)
except SystemExit:
    check("the OLD 'any model.pt' behaviour is gone", True)
# the listing MUST be requested with a large page size. It paginates at 20 rows by default,
# and this dataset has ~30 files, so m5/m6/m7_model.pt land on page 2 and look dropped.
check("asks kaggle for a large page (pagination bug)",
      any("--page-size" in a for a in calls), str(calls[:1]))
sc._kaggle, _time.sleep = real_kaggle, real_sleep

# --- 2. the staging directory is wiped, not merged ---------------------------------
print("\n2. stale-staging guard (build_kaggle_ensemble_assets)")
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
i = src.index("def build_kaggle_ensemble_assets")
head = src[i:i + 1600]
check("the builder rmtree's the staging dir before writing",
      "rmtree" in head and head.index("rmtree") < head.index('mkdir(parents=True'))

# --- 3 and 4. the preflight in submit_ensemble -------------------------------------
print("\n3+4. preflight (stale NOTE, dead member, wrong file count)")
sub = (ROOT / "runs" / "others" / "ensembling" / "submit_ensemble.py").read_text(encoding="utf-8")
for label, needle in [
        ("checks the built file count against MEMBERS", "built %d model files for %d MEMBERS"),
        ("checks NOTE actually names the members", "NOTE names only"),
        ("checks no member carries zero weight everywhere", "carries ZERO weight"),
        ("raises rather than warning", 'raise SystemExit("preflight failed')]:
    check(label, needle in sub)

# the preflight must run BEFORE anything is uploaded
check("preflight runs before kaggle_publish",
      sub.index("preflight failed") < sub.index("sc.kaggle_publish"))

print("\n" + "=" * 76)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("all submission guards fire as intended")
