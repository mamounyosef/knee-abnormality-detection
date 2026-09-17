r"""mcl_probe_check.py - are the two zero-training probes actually zero-training, and is the
control really a control?

The probes exist to separate two things every MCL run so far has mixed together: adapting
the trunk to a scan mix it never saw, and learning MCL. They score the SEEDED weights at
step 0 and stop.

  dinov3_mcl_probe_parent  the parent's OWN slots and slice count. The sliced head
                           reproduces the parent's MCL row exactly, so on the parent's own
                           inputs this MUST return the parent's own 0.8855. If it does not,
                           something in the seeding, the head slicing or the eval path is
                           broken - which is the "is there a bug?" question, answered.
  dinov3_mcl_probe_new     the MCL slot set at 30 slices. Its gap from 0.8855 is the cost or
                           benefit of the slot change, with no training in the way.

What would make the probes worthless and is therefore checked here:
  * the control's geometry drifting from the parent's, so it is no longer a control
  * validate_before_training not set to "only", so the probe quietly trains and bills for it
  * the two probes differing in anything but geometry
  * the head not being the parent's MCL row, which would make both numbers meaningless

CPU only, no GPU and no data needed.

    python checks/mcl_probe_check.py
"""

import csv
import importlib.util
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

CONTROL = "dinov3_mcl_probe_parent"
TREAT = "dinov3_mcl_probe_new"
PARENT = "dinov3_convnext_base"
SEED = ROOT / "runs" / PARENT / "results/checkpoints/step5600.pt"
PARENT_TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
                'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
                'Contusion', 'Fracture']
# geometry keys that decide what pixels reach the model - the control must match the parent
# on every one of them or it proves nothing
GEOM = ("slots", "n_slices", "size", "mm_per_px", "window", "window_pct", "channels",
        "slice_margin", "slice_sampling", "interpolation", "resize_to",
        "fallback_same_plane", "duplicate_rule")
fails = []


def check(name, ok, detail=""):
    print("  %-62s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


cfgs = {r: sc.load_config(ROOT / "runs" / r, verbose=False) for r in (CONTROL, TREAT)}
pcfg = sc.load_config(ROOT / "runs" / PARENT, verbose=False)

# --- 1. neither probe can train -------------------------------------------------------
print("\n1. they are zero-training")
for r in (CONTROL, TREAT):
    v = cfgs[r]["evaluation"]["validate_before_training"]
    check("%s is validate_before_training='only'" % r, str(v).lower() == "only", repr(v))
src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
check("the engine honours 'only' by returning before the loop",
      'if str(_vbt).lower() == "only":' in src and "return {\"probe\": True" in src)
check("the step-0 validation happens BEFORE the epoch loop",
      src.index("validate_before_training") < src.index("for epoch in range(start_epoch"))
check("it is skipped on a resume, so it cannot fire mid-run",
      'cfg["evaluation"].get("validate_before_training") and not ck' in src
      or "_vbt and not ck" in src)
check("the default stays off everywhere else",
      sc.load_config(ROOT / "runs" / "convnext_base_relabel",
                     verbose=False)["evaluation"]["validate_before_training"] is False)

# --- 2. the CONTROL is genuinely the parent's geometry --------------------------------
print("\n2. the control reproduces the parent's input exactly")
for k in GEOM:
    check("input.%s matches %s" % (k, PARENT),
          cfgs[CONTROL]["input"].get(k) == pcfg["input"].get(k),
          "%s vs %s" % (cfgs[CONTROL]["input"].get(k), pcfg["input"].get(k)))
check("augmentation is OFF for eval, so the pass is deterministic",
      cfgs[CONTROL]["augmentation"].get("eval_enable") is False)
check("eval slice sampling is 'even', not random",
      cfgs[CONTROL]["input"]["slice_sampling"] == "even")

# --- 3. the two probes differ ONLY in geometry ----------------------------------------
print("\n3. the probes differ only where intended")
fa, fb = flat(cfgs[CONTROL]), flat(cfgs[TREAT])
diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
want = {"experiment.name", "input.slots", "input.n_slices"}
check("changed exactly the geometry and the name", set(diff) == want,
      "unexpected %s" % sorted(set(diff) - want))
check("the treatment adds cor_plain", "cor_plain" in cfgs[TREAT]["input"]["slots"])
check("the control does NOT have cor_plain", "cor_plain" not in cfgs[CONTROL]["input"]["slots"])

# --- 4. both are the same seeded model ------------------------------------------------
print("\n4. both load the parent's MCL row, unchanged")
ck = torch.load(SEED, map_location="cpu", weights_only=False)
rows = [r for r in csv.DictReader(open(ROOT / "runs" / PARENT / "results" / "val_log.csv",
                                       encoding="utf-8")) if r.get("val_auroc_MCL")]
best_mcl = max(rows, key=lambda r: float(r["val_auroc_MCL"]))
target = float(best_mcl["val_auroc_MCL"])
check("the seed is the parent's best MCL step (%s)" % best_mcl["step"],
      (ck.get("global_step") or ck.get("step")) == int(best_mcl["step"]))
print("     -> the control must return MCL %.4f" % target)

pw = ck["model"]["head.weight"][PARENT_TASKS.index("MCL")]
pb = ck["model"]["head.bias"][PARENT_TASKS.index("MCL")]
for r in (CONTROL, TREAT):
    txt = (ROOT / "runs" / r / "train.py").read_text(encoding="utf-8")
    check("%s declares EXP_NAME as its own literal" % r,
          set(re.findall(r'EXP_NAME = "([^"]*)"', txt)) == {r},
          str(set(re.findall(r'EXP_NAME = "([^"]*)"', txt))))
    check("%s seeds from step5600" % r, "step5600.pt" in txt)
    spec = importlib.util.spec_from_file_location("p_" + r, ROOT / "runs" / r / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    check("%s uses HEAD_INIT=slice" % r, getattr(m, "HEAD_INIT", None) == "slice")
    b = dict(cfgs[r])
    b["model"] = dict(cfgs[r]["model"], pretrained=False)
    b.setdefault("modal", {})["runs_mount"] = str(ROOT / "runs")
    mdl = m.build_model(b)
    check("%s head EQUALS the parent's MCL row" % r,
          torch.allclose(mdl.head.weight.detach()[0], pw, atol=0)
          and torch.allclose(mdl.head.bias.detach()[0], pb, atol=0),
          "max diff %.2e" % float((mdl.head.weight.detach()[0] - pw).abs().max()))
    bb = [(k, v) for k, v in mdl.state_dict().items() if not k.startswith("head.")]
    ref = ck["model"]
    ident = sum(1 for k, v in bb if k in ref and torch.equal(v, ref[k]))
    check("%s backbone is bit-identical (%d/%d)" % (r, ident, len(bb)), ident == len(bb))
    want_imgs = len(cfgs[r]["input"]["slots"]) * cfgs[r]["input"]["n_slices"]
    check("%s feeds %d images/study" % (r, want_imgs), mdl.per_study == want_imgs)

print("\n" + "=" * 80)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("both probes are one validation pass over the same seeded weights;")
print("%s is a true control and must return MCL %.4f" % (CONTROL, target))
