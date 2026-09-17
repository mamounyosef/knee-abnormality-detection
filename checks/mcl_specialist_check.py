r"""mcl_specialist_check.py - prove the two MCL arms differ ONLY in how the head is seeded.

The whole point of running two arms is that one variable changes. If anything else drifts -
slots, slices, tasks, the seed checkpoint - the comparison answers nothing and costs two GPU
hours to say so.

What would fail silently without this:
  * the seed checkpoint not found, so a run trains from the pretrained weights and looks
    completely normal
  * "slice" not actually copying the parent's MCL row, so both arms are really "fresh"
  * "fresh" accidentally inheriting the head, so both arms are really "slice"
  * the backbone not transferring, leaving a half-random model
  * the two configs differing in something other than the run name
  * EXP_NAME not being a literal, so the container resolves EXP_DIR to /root/runs/root and
    trains SOMETHING ELSE - this already cost $12.46 once
  * a slot- or slice-shaped parameter in the checkpoint, which would make the slot change
    silently reinterpret weights instead of just feeding different images

CPU only, no GPU and no data needed.

    python checks/mcl_specialist_check.py
"""

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

ARMS = ["dinov3_mcl_slice", "dinov3_mcl_fresh"]
SEED = ROOT / "runs/dinov3_convnext_base/results/checkpoints/best.pt"
PARENT_TASKS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
                'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
                'Contusion', 'Fracture']
WANT_SLOTS = ["cor_fluid", "cor_plain", "ax_fluid", "sag_fluid"]
WANT_SLICES = 30
fails = []


def check(name, ok, detail=""):
    print("  %-60s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


def load_mod(run):
    spec = importlib.util.spec_from_file_location("t_" + run, ROOT / "runs" / run / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- 1. the configs differ only where intended ------------------------------------
print("\n1. the two configs")
cfgs = {r: sc.load_config(ROOT / "runs" / r, verbose=False) for r in ARMS}
a, b = cfgs[ARMS[0]], cfgs[ARMS[1]]


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


fa, fb = flat(a), flat(b)
diff = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
check("differ ONLY in experiment.name", diff == ["experiment.name"], str(diff))
check("tasks is exactly [MCL]", a["tasks"] == ["MCL"], str(a["tasks"]))
check("slots are the MCL set", a["input"]["slots"] == WANT_SLOTS, str(a["input"]["slots"]))
check("both coronal scans are present",
      {"cor_fluid", "cor_plain"} <= set(a["input"]["slots"]))
check("full resolution kept (512 px @ 0.40 mm/px)",
      a["input"]["size"] == 512 and a["input"]["mm_per_px"] == 0.40,
      "%s px, %s mm/px" % (a["input"]["size"], a["input"]["mm_per_px"]))
check("NOT 2.5D - one slice per image",
      not a["input"].get("slices_as_channels"), str(a["input"].get("slices_as_channels")))
check("n_slices is %d" % WANT_SLICES, a["input"]["n_slices"] == WANT_SLICES,
      str(a["input"]["n_slices"]))
check("the parent's own extend block is OFF",
      a["training"]["extend"]["enable"] is False)
check("grad_clip is at or above 3.0", a["training"]["grad_clip"] >= 3.0,
      str(a["training"]["grad_clip"]))

# --- 2. EXP_NAME is a LITERAL and resolves the way the CONTAINER resolves it ---------
# Modal copies the entrypoint to /root/train.py, so anything derived from the file's parent
# folder becomes "root" and EXP_DIR points at a config that does not exist.
print("\n2. EXP_NAME resolves in the container, not just here")
for r in ARMS:
    txt = (ROOT / "runs" / r / "train.py").read_text(encoding="utf-8")
    declared = set(re.findall(r'EXP_NAME = "([^"]*)"', txt))
    check("%s declares EXP_NAME as the literal %s" % (r, r), declared == {r}, str(declared))
    check("%s has a config.yaml at that name" % r,
          (ROOT / "runs" / r / "config.yaml").exists())
    check("%s never derives the name from __file__" % r,
          "__file__" not in txt.split("EXP_NAME")[0].rsplit("\n", 1)[0]
          or "parent.name" not in txt)

# --- 3. the seed checkpoint ---------------------------------------------------------
print("\n3. the seed checkpoint")
check("the parent checkpoint is present locally", SEED.exists(), str(SEED))
ck = torch.load(SEED, map_location="cpu", weights_only=False)
check("its head is (12, 1024)", tuple(ck["model"]["head.weight"].shape) == (12, 1024))
# a slot- or slice-shaped parameter would mean the slot change reinterprets weights rather
# than simply feeding different images. pool="mean" must hold for that to be true.
slotted = [k for k in ck["model"]
           if any(t in k for t in ("pos_slot", "pos_slice", "pool_attn", "encoder"))]
check("NO slot- or slice-shaped parameters in the checkpoint", not slotted, str(slotted))
check("the parent pools with a plain mean",
      sc.load_config(ROOT / "runs" / "dinov3_convnext_base",
                     verbose=False)["model"]["pool"] == "mean")

# --- 4. HEAD_INIT is what each arm claims -------------------------------------------
print("\n4. each arm declares the right head_init")
mods = {}
for r in ARMS:
    m = load_mod(r)
    mods[r] = m
    want = r.rsplit("_", 1)[1]
    check("%s declares HEAD_INIT=%s" % (r, want), getattr(m, "HEAD_INIT", None) == want,
          str(getattr(m, "HEAD_INIT", None)))
    check("%s points at the dinov3 parent" % r,
          "dinov3_convnext_base" in getattr(m, "BRANCH_FROM", ""),
          getattr(m, "BRANCH_FROM", ""))

# --- 5. build each model and verify what actually transferred -----------------------
print("\n5. what actually lands in the weights")
mcl_row = PARENT_TASKS.index("MCL")
parent_w = ck["model"]["head.weight"][mcl_row]
parent_b = ck["model"]["head.bias"][mcl_row]

for r in ARMS:
    cfg = dict(cfgs[r])
    cfg["model"] = dict(cfg["model"], pretrained=False)
    cfg.setdefault("modal", {})["runs_mount"] = str(ROOT / "runs")
    model = mods[r].build_model(cfg)
    hw = model.head.weight.detach()
    check("%s head is (1, 1024)" % r, tuple(hw.shape) == (1, 1024), str(tuple(hw.shape)))

    same_head = torch.allclose(hw[0], parent_w, atol=1e-6)
    if r.endswith("slice"):
        check("slice: head EQUALS the parent's MCL row", same_head,
              "max diff %.2e" % float((hw[0] - parent_w).abs().max()))
        check("slice: bias equals the parent's MCL bias",
              torch.allclose(model.head.bias.detach()[0], parent_b, atol=1e-6))
    else:
        check("fresh: head does NOT equal the parent's MCL row", not same_head,
              "max diff %.2e" % float((hw[0] - parent_w).abs().max()))

    # the BACKBONE must transfer in both arms - that is the whole point of seeding
    bb = [(k, v) for k, v in model.state_dict().items() if not k.startswith("head.")]
    ref = ck["model"]
    matched = sum(1 for k, v in bb if k in ref and ref[k].shape == v.shape
                  and torch.allclose(v, ref[k], atol=1e-6))
    check("%s: backbone transferred (%d/%d tensors identical)" % (r, matched, len(bb)),
          matched == len(bb), "%d mismatched" % (len(bb) - matched))

    # the reshape the model does per study must match the new slot/slice product
    want = len(cfgs[r]["input"]["slots"]) * cfgs[r]["input"]["n_slices"]
    check("%s: per_study is %d images" % (r, want), model.per_study == want,
          str(model.per_study))

# --- 6. the slice budget actually buys distinct slices ------------------------------
# pick_slices REPEATS indices once n_want passes the usable band, so a too-high n_slices
# spends forward passes on duplicates. 30 must still be all-distinct on a median stack.
print("\n6. n_slices=%d is not paying for duplicate images" % WANT_SLICES)
rng = np.random.default_rng(0)
for native in (30, 32, 36):
    idx = sc.pick_slices(native, WANT_SLICES, "even", rng, a["input"].get("slice_margin", 0.05))
    check("a %d-slice stack yields %d distinct of %d" % (native, len(set(idx.tolist())),
                                                         WANT_SLICES),
          len(set(idx.tolist())) == WANT_SLICES)

print("\n" + "=" * 76)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("both arms seed from dinov3_convnext_base and differ ONLY in the head")
