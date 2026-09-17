r"""mcl_continue_check.py - does dinov3_mcl_slice_ft really start from the 0.8837 model?

The point of this run is to continue from a checkpoint that is ALREADY adapted to the MCL
slot set, instead of from dinov3_convnext_base step5600, which had never seen cor_plain. If
the weights do not land bit-identical, the run is not continuing from 0.8837 and the whole
premise is gone - and it would still train, and still log plausible numbers.

The specific trap here is the head. The three earlier arms sliced a (12, 1024) head down to
one row. This checkpoint's head is ALREADY (1, 1024), so:
    "slice" would index row 1 of a 1-row matrix   -> wrong weights or a crash
    "fresh" would discard the head entirely       -> throws away the 0.8837
    "keep"  loads it untouched                    -> what this run needs

CPU only, no GPU and no data needed.

    python checks/mcl_continue_check.py
"""

import importlib.util
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUN = "dinov3_mcl_slice_ft"
PRIOR = "dinov3_mcl_slice_lowlr"
CKPT = ROOT / "runs/dinov3_mcl_slice/results/checkpoints/best.pt"
WANT_DIFF = {"experiment.name", "training.epochs", "training.optimizer.lr",
             "training.optimizer.backbone_lr_mult",
             "evaluation.eval_every_steps_before",
             "evaluation.eval_every_steps_epoch_threshold"}
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


# --- 1. the checkpoint this continues from -------------------------------------------
print("\n1. the checkpoint being continued")
check("it is present locally", CKPT.exists(), str(CKPT))
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
score = ck.get("score")
check("it is the 0.8837 checkpoint", abs(float(score) - 0.883718) < 1e-5, str(score))
check("it is step 1000, epoch 3",
      (ck.get("global_step") or ck.get("step")) == 1000 and ck.get("epoch") == 3,
      "step %s epoch %s" % (ck.get("global_step") or ck.get("step"), ck.get("epoch")))
check("its head is ALREADY (1, 1024)", tuple(ck["model"]["head.weight"].shape) == (1, 1024),
      str(tuple(ck["model"]["head.weight"].shape)))
ock = (ck.get("cfg") or {})
check("it was trained on tasks [MCL]", ock.get("tasks") == ["MCL"], str(ock.get("tasks")))

# --- 2. the new run feeds it the SAME inputs it was trained on -------------------------
# a continuation that changes the input is not a continuation.
print("\n2. the inputs match what that checkpoint saw")
cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
oi = ock.get("input") or {}
for k in ("slots", "n_slices", "size", "mm_per_px", "window", "channels"):
    check("input.%s is unchanged" % k, cfg["input"].get(k) == oi.get(k),
          "%s -> %s" % (oi.get(k), cfg["input"].get(k)))
check("tasks unchanged", cfg["tasks"] == ock.get("tasks"), str(cfg["tasks"]))

# --- 3. only the intended knobs moved --------------------------------------------------
print("\n3. what changed against %s" % PRIOR)
fa, fb = flat(cfg), flat(sc.load_config(ROOT / "runs" / PRIOR, verbose=False))
diff = {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
check("changed exactly the intended keys", diff == WANT_DIFF,
      "unexpected %s, missing %s" % (sorted(diff - WANT_DIFF), sorted(WANT_DIFF - diff)))
o = cfg["training"]["optimizer"]
head_lr = o["lr"] / o["backbone_lr_mult"]
check("backbone lr is 2.0e-6", abs(o["lr"] - 2.0e-6) < 1e-12, "%.2e" % o["lr"])
check("head lr is 3.0e-6", abs(head_lr - 3.0e-6) < 1e-12, "%.2e" % head_lr)
# the rates live at step 1000 were backbone 1.147e-5, head 2.867e-5
check("backbone rate is well below what was live at step 1000", o["lr"] < 1.147e-5 * 0.25,
      "%.0f%% of it" % (100 * o["lr"] / 1.147e-5))
check("head rate is well below what was live at step 1000", head_lr < 2.867e-5 * 0.25,
      "%.0f%% of it" % (100 * head_lr / 2.867e-5))
check("grad_clip still above the floor of 3", cfg["training"]["grad_clip"] >= 3.0,
      str(cfg["training"]["grad_clip"]))
check("validate_on_resume is off", cfg["evaluation"]["validate_on_resume"] is False)

# --- 4. EXP_NAME literal ----------------------------------------------------------------
print("\n4. EXP_NAME resolves in the container")
txt = (ROOT / "runs" / RUN / "train.py").read_text(encoding="utf-8")
check("declares EXP_NAME as the literal %s" % RUN,
      set(re.findall(r'EXP_NAME = "([^"]*)"', txt)) == {RUN},
      str(set(re.findall(r'EXP_NAME = "([^"]*)"', txt))))
check("has a config.yaml at that name", (ROOT / "runs" / RUN / "config.yaml").exists())
check("BRANCH_FROM points at the specialist checkpoint",
      "dinov3_mcl_slice/results/checkpoints/best.pt" in txt)
check("it does NOT still point at the old step5600 seed", "step5600.pt" not in txt)

# --- 5. the weights land BIT-IDENTICAL --------------------------------------------------
print("\n5. the model starts as exactly that checkpoint")
spec = importlib.util.spec_from_file_location("t_" + RUN, ROOT / "runs" / RUN / "train.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
check("declares HEAD_INIT=keep", getattr(mod, "HEAD_INIT", None) == "keep",
      str(getattr(mod, "HEAD_INIT", None)))

bcfg = dict(cfg)
bcfg["model"] = dict(cfg["model"], pretrained=False)
bcfg.setdefault("modal", {})["runs_mount"] = str(ROOT / "runs")
model = mod.build_model(bcfg)
sd, ref = model.state_dict(), ck["model"]
check("same tensor set", set(sd) == set(ref), "%d vs %d" % (len(sd), len(ref)))
exact = sum(1 for k in sd if k in ref and sd[k].shape == ref[k].shape
            and torch.equal(sd[k], ref[k]))
check("ALL %d tensors are bit-identical, head included" % len(sd), exact == len(sd),
      "%d differ" % (len(sd) - exact))
check("the head specifically is untouched",
      torch.equal(model.head.weight.detach(), ref["head.weight"])
      and torch.equal(model.head.bias.detach(), ref["head.bias"]))

# --- 6. the keep guard actually rejects a wrong head -------------------------------------
# without this, a future 12-logit checkpoint would load silently and train the wrong thing
print("\n6. the keep guard rejects a mismatched head")
src = (ROOT / "runs" / RUN / "train.py").read_text(encoding="utf-8")
check("the guard exists in the code", "head_init=keep wants a" in src)
bad = dict(cfg)
bad["tasks"] = ["ACL", "MCL"]                       # would need a (2, 1024) head
bad["model"] = dict(cfg["model"], pretrained=False)
bad.setdefault("modal", {})["runs_mount"] = str(ROOT / "runs")
try:
    mod.build_model(bad)
    check("it raises when the head shape is wrong", False, "it loaded silently")
except RuntimeError as e:
    check("it raises when the head shape is wrong", "head_init=keep" in str(e))
except Exception as e:
    check("it raises when the head shape is wrong", False, "%s: %s" % (type(e).__name__, e))

print("\n" + "=" * 80)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("%s starts as the 0.8837 model, byte for byte, at 2.0e-6 / 3.0e-6" % RUN)
