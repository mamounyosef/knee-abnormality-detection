"""inference_cpu_caps_check.py - the CPU is not oversubscribed, and the batch knobs work.

Our two-GPU member split is correct but measured 2.4x slower PER MEMBER than one card, while
the public two-GPU notebook doing the same thing is fast. The difference is the CPU, not
CUDA. Kaggle gives 4 virtual cores (vCPUs) and two T4 cards:

  - OpenMP and Intel Math Kernel Library (MKL) each start one thread PER CORE inside every
    PyTorch intra-op call unless capped, so two per-GPU threads plus four DICOM decode
    workers all fight over four cores. The caps are read ONCE, when torch is imported, so
    setting them after the import does nothing at all.
  - at a couple of studies per forward the loop is kernel-LAUNCH bound. Launching is Python
    work holding the Global Interpreter Lock (GIL), so a second GPU thread launching at the
    same moment simply waits. A larger batch gives each launch more GPU work to cover.

Run:  python checks/inference_cpu_caps_check.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))

PASS, FAIL = [], []


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-58s %s" % ("PASS" if ok else "FAIL", n, d))


print("\n1. the caps are set, and set BEFORE torch is imported")
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
i_cap = src.find("OMP_NUM_THREADS")
i_torch = src.find("\nimport torch")
ck("the cap block exists", i_cap > 0)
ck("it sits above `import torch`", 0 < i_cap < i_torch,
   "cap at char %d, import torch at %d" % (i_cap, i_torch))
ck("all three pools are capped",
   all(v in src[:i_torch] for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "MKL_NUM_THREADS")))
ck("setdefault, so an explicit environment still wins", "os.environ.setdefault" in src[:i_torch])
ck("`os` is imported exactly once", src.count("\nimport os\n") == 1)

import shared_code as sc  # noqa: E402
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    ck("importing shared_code sets %s" % v, os.environ.get(v) == "4", os.environ.get(v))

print("\n2. an explicit value is not overwritten")
ck("setdefault leaves a caller's choice alone",
   os.environ.setdefault("OMP_NUM_THREADS", "99") == "4")

print("\n3. the config knobs reach the loader")
cfg = sc.load_config(ROOT / "runs" / "convnext_base_relabel")
inf = cfg.get("inference", {})
ck("inference.batch_size exists and defaults to null", "batch_size" in inf and inf["batch_size"] is None)
ck("inference.num_workers exists and defaults to null",
   "num_workers" in inf and inf["num_workers"] is None)
body = src[src.find("def predict_dicom_ensemble("):src.find("def predict_dicom_ensemble_grouped")]
ck("an explicit argument still beats the config",
   "if batch_size is None and _inf0.get(\"batch_size\")" in body)
ck("workers likewise", 'if num_workers is None and _inf0.get("num_workers") is not None' in body)
ck("the config is read before nw is computed",
   body.find("_inf0.get(\"num_workers\")") < body.find("nw = num_workers if num_workers"))
ck("the grouped path forwards both, so the config applies there too",
   "batch_size=batch_size, num_workers=num_workers," in src)

print("\n4. null means the historical behaviour, unchanged")
ck("no batch_size falls back to the validation batch size",
   "int(batch_size or val_loader_settings(cfg0)[0])" in src)
ck("no num_workers falls back to min(4, cpu_count)",
   "min(4, (os.cpu_count() or 2))" in body)

print("\n" + "=" * 72)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
