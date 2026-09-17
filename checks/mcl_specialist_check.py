r"""mcl_specialist_check.py - is the MCL specialist built the way it claims?

History this guards, all of it paid for once already:
  * the first build seeded from convnext_base_25d_18slices, the WORST parent in the pool at
    MCL (0.8384 against dinov3's 0.8736), because the seed was picked for runtime
  * the rebuild then took best.pt, the best MACRO checkpoint (MCL 0.8736), when step5600 is
    the best MCL one (0.8855) - giving away +0.0119 before training started
  * three fold runs once trained their PARENT because EXP_NAME was not a literal, at a cost
    of $12.46 and 3.5 A100 hours
  * the seed checkpoint being absent from the target volume, which trains from the plain
    pretrained backbone while logging entirely normal-looking numbers

The live run is dinov3_mcl_slice_lowlr, a clean restart of dinov3_mcl_slice at a lower rate.
dinov3_mcl_fresh is retired: it peaked at 0.8743 against the sliced arm's 0.8837, so the
head question is answered and one arm is enough.

CPU only, no GPU and no data needed.

    python checks/mcl_specialist_check.py
"""

import csv
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUN = "dinov3_mcl_slice_lowlr"
PRIOR = "dinov3_mcl_slice"                 # the attempt it restarts
PARENT = "dinov3_convnext_base"
SEED = ROOT / "runs" / PARENT / "results/checkpoints/step5600.pt"
PARENT_TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
                'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
                'Contusion', 'Fracture']
WANT_SLOTS = ["cor_fluid", "cor_plain", "ax_fluid", "sag_fluid"]
WANT_SLICES = 30
# the ONLY things the restart is allowed to move
WANT_DIFF = {"experiment.name", "training.grad_clip", "training.optimizer.lr",
             "training.optimizer.backbone_lr_mult", "evaluation.validate_on_resume"}
fails = []


def check(name, ok, detail=""):
    print("  %-60s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
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


# --- 1. geometry and task ------------------------------------------------------------
print("\n1. the config")
cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
check("tasks is exactly [MCL]", cfg["tasks"] == ["MCL"], str(cfg["tasks"]))
check("slots are the MCL set", cfg["input"]["slots"] == WANT_SLOTS, str(cfg["input"]["slots"]))
check("both coronal scans are present",
      {"cor_fluid", "cor_plain"} <= set(cfg["input"]["slots"]))
check("full resolution kept (512 px @ 0.40 mm/px)",
      cfg["input"]["size"] == 512 and cfg["input"]["mm_per_px"] == 0.40,
      "%s px, %s mm/px" % (cfg["input"]["size"], cfg["input"]["mm_per_px"]))
check("NOT 2.5D - one slice per image", not cfg["input"].get("slices_as_channels"))
check("n_slices is %d" % WANT_SLICES, cfg["input"]["n_slices"] == WANT_SLICES,
      str(cfg["input"]["n_slices"]))
check("the parent's own extend block is OFF", cfg["training"]["extend"]["enable"] is False)

# --- 2. the restart moved ONLY what it meant to --------------------------------------
print("\n2. what the restart changed against %s" % PRIOR)
fa, fb = flat(cfg), flat(sc.load_config(ROOT / "runs" / PRIOR, verbose=False))
diff = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
check("changed exactly the intended keys", diff == WANT_DIFF,
      "unexpected %s, missing %s" % (sorted(diff - WANT_DIFF), sorted(WANT_DIFF - diff)))
o = cfg["training"]["optimizer"]
head_lr = o["lr"] / o["backbone_lr_mult"]
check("backbone lr is 6.0e-6", abs(o["lr"] - 6.0e-6) < 1e-12, "%.2e" % o["lr"])
check("head lr is 8.0e-6", abs(head_lr - 8.0e-6) < 1e-12, "%.2e" % head_lr)
check("the head is no longer on a fresh-head multiplier", head_lr / o["lr"] < 2.0,
      "head is %.2fx the trunk" % (head_lr / o["lr"]))
check("grad_clip is 20, not a throttle", cfg["training"]["grad_clip"] == 20.0,
      str(cfg["training"]["grad_clip"]))
check("grad_clip stays above the floor of 3", cfg["training"]["grad_clip"] >= 3.0)
check("validate_on_resume is OFF for a clean start",
      cfg["evaluation"]["validate_on_resume"] is False)

# grad_clip has to sit ABOVE the typical gradient, or it throttles every step instead of
# catching spikes. Measured from the first attempt's own log.
tl = ROOT / "runs" / PRIOR / "results" / "train_log.csv"
if tl.exists():
    g = sorted(float(r["gnorm"]) for r in csv.DictReader(open(tl, encoding="utf-8"))
               if r.get("gnorm") not in (None, "") and 0 < float(r["gnorm"]) < 1e30)
    if g:
        med = g[len(g) // 2]
        clipped = 100.0 * sum(1 for x in g if x > cfg["training"]["grad_clip"]) / len(g)
        check("grad_clip is above the MEDIAN gradient norm (%.1f)" % med,
              cfg["training"]["grad_clip"] > med, "clips %.1f%% of steps" % clipped)
        check("it still clips something, so it is a real guard", clipped > 1.0,
              "%.1f%%" % clipped)

# --- 3. EXP_NAME is a LITERAL that resolves the way the CONTAINER resolves it ---------
print("\n3. EXP_NAME resolves in the container, not just here")
txt = (ROOT / "runs" / RUN / "train.py").read_text(encoding="utf-8")
check("declares EXP_NAME as the literal %s" % RUN,
      set(re.findall(r'EXP_NAME = "([^"]*)"', txt)) == {RUN},
      str(set(re.findall(r'EXP_NAME = "([^"]*)"', txt))))
check("has a config.yaml at that name", (ROOT / "runs" / RUN / "config.yaml").exists())
check("never derives the name from __file__", "parent.name" not in txt)
for f in ("evaluate.py", "plot_results.py", "submit.py"):
    p = ROOT / "runs" / RUN / f
    if p.exists():
        check("%s points at itself too" % f,
              set(re.findall(r'EXP_NAME = "([^"]*)"', p.read_text(encoding="utf-8"))) == {RUN})

# --- 4. the seed checkpoint is the best MCL one, not the best macro one ---------------
print("\n4. the seed checkpoint")
check("present locally", SEED.exists(), str(SEED.name))
ck = torch.load(SEED, map_location="cpu", weights_only=False)
check("its head is (12, 1024)", tuple(ck["model"]["head.weight"].shape) == (12, 1024))
rows = [r for r in csv.DictReader(open(ROOT / "runs" / PARENT / "results" / "val_log.csv",
                                       encoding="utf-8"))
        if r.get("val_auroc_MCL") and r.get("val_macro_auroc")]
best_mcl = max(rows, key=lambda r: float(r["val_auroc_MCL"]))
best_macro = max(rows, key=lambda r: float(r["val_macro_auroc"]))
want_step = int(best_mcl["step"])
got_step = ck.get("global_step") or ck.get("step")
check("is the parent's best MCL step (%d, MCL %.4f)"
      % (want_step, float(best_mcl["val_auroc_MCL"])), got_step == want_step,
      "got step %s" % got_step)
check("is NOT the best-macro checkpoint (step %s, MCL %.4f)"
      % (best_macro["step"], float(best_macro["val_auroc_MCL"])),
      got_step != int(best_macro["step"]) or want_step == int(best_macro["step"]))
check("BRANCH_FROM points at step%d.pt" % want_step, "step%d.pt" % want_step in txt)
slotted = [k for k in ck["model"]
           if any(t in k for t in ("pos_slot", "pos_slice", "pool_attn", "encoder"))]
check("NO slot- or slice-shaped parameters, so the slot change is safe", not slotted,
      str(slotted))
check("the parent pools with a plain mean",
      sc.load_config(ROOT / "runs" / PARENT, verbose=False)["model"]["pool"] == "mean")

# --- 5. build it and verify what actually transferred ---------------------------------
print("\n5. what actually lands in the weights")
spec = importlib.util.spec_from_file_location("t_" + RUN, ROOT / "runs" / RUN / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
check("declares HEAD_INIT=slice", getattr(mod, "HEAD_INIT", None) == "slice",
      str(getattr(mod, "HEAD_INIT", None)))
check("a missing parent is FATAL while training", getattr(mod, "REQUIRE_PARENT", None) is False
      and "REQUIRE_PARENT:" in txt and txt.count("REQUIRE_PARENT = True") == 2)

bcfg = dict(cfg)
bcfg["model"] = dict(cfg["model"], pretrained=False)
bcfg.setdefault("modal", {})["runs_mount"] = str(ROOT / "runs")
model = mod.build_model(bcfg)
mcl_row = PARENT_TASKS.index("MCL")
pw, pb = ck["model"]["head.weight"][mcl_row], ck["model"]["head.bias"][mcl_row]
hw = model.head.weight.detach()
check("head is (1, 1024)", tuple(hw.shape) == (1, 1024), str(tuple(hw.shape)))
check("head EQUALS the parent's MCL row", torch.allclose(hw[0], pw, atol=1e-6),
      "max diff %.2e" % float((hw[0] - pw).abs().max()))
check("bias equals the parent's MCL bias",
      torch.allclose(model.head.bias.detach()[0], pb, atol=1e-6))
bb = [(k, v) for k, v in model.state_dict().items() if not k.startswith("head.")]
ref = ck["model"]
matched = sum(1 for k, v in bb if k in ref and ref[k].shape == v.shape
              and torch.allclose(v, ref[k], atol=1e-6))
check("backbone transferred (%d/%d tensors identical)" % (matched, len(bb)),
      matched == len(bb), "%d mismatched" % (len(bb) - matched))
want = len(cfg["input"]["slots"]) * cfg["input"]["n_slices"]
check("per_study is %d images" % want, model.per_study == want, str(model.per_study))

# --- 6. the slice budget still buys distinct slices -----------------------------------
print("\n6. n_slices=%d is not paying for duplicate images" % WANT_SLICES)
rng = np.random.default_rng(0)
for native in (30, 32, 36):
    idx = sc.pick_slices(native, WANT_SLICES, "even", rng,
                         cfg["input"].get("slice_margin", 0.05))
    check("a %d-slice stack yields %d distinct of %d"
          % (native, len(set(idx.tolist())), WANT_SLICES),
          len(set(idx.tolist())) == WANT_SLICES)

print("\n" + "=" * 78)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("%s: sliced head off the parent's BEST MCL step, at the lower rate" % RUN)
