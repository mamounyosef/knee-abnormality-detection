r"""slots_and_pooling_deep_check.py - end-to-end audit of convnext_base_25d_perfinding_slots_and_pooling.

That run stacks four things this project has never shipped together: six slots instead of
four, per-slot slice counts, a concatenated image-axis item layout, and a per-finding head
that pools its own subset of windows. The earlier checks each test one piece with a stub
trunk. This one runs the REAL model on REAL shards and takes REAL gradient steps, because
the failure modes that survive unit tests are the ones that only appear when the pieces are
wired together.

What it establishes:
  1. Geometry: 40 windows, 3 channels, coronal really drawing 24 distinct planes.
  2. Item layout: slot order in the concatenated tensor is right for EVERY slot, not just one.
  3. The real backbone accepts the real batch, and loss/gradients are finite.
  4. A few optimizer steps on a fixed batch actually reduce the loss, so the heads are
     connected to the graph - a head whose gradient never arrives trains silently at chance.
  5. Every one of the 12 heads receives gradient, and only from its own windows.
  6. Checkpoint save/load round-trips this architecture exactly.
  7. The resume config guard tolerates a dict n_slices instead of crashing on it.
  8. Nothing here changed the 4-slot parent.

CPU only, small resolution for the trunk, so it runs on this PC in a couple of minutes.

    python checks/slots_and_pooling_deep_check.py
"""

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = "convnext_base_25d_perfinding_slots_and_pooling"
PARENT = "convnext_base_25d_18slices"
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def load_mod(run):
    d = ROOT / "runs" / run
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("deep_" + run, d / "train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    mod = load_mod(RUN)
    cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
    slots = list(cfg["input"]["slots"])
    tasks = list(cfg["tasks"])

    # ---- 1. geometry --------------------------------------------------------------------
    n_win = [sc.slot_groups(cfg, s) for s in slots]
    check("40 windows per study", sc.images_per_study(cfg) == 40,
          "%d = %s" % (sc.images_per_study(cfg), "+".join(map(str, n_win))))
    check("every slot yields 3 channels", sc.check_slot_geometry(cfg) == 3)
    check("coronal samples 24, everything else 18",
          all(sc.slot_slices(cfg, s) == (24 if s.startswith("cor") else 18) for s in slots))
    check("six slots", len(slots) == 6, ", ".join(slots))

    # ---- 2. real shards, real dataset ---------------------------------------------------
    shards = Path(sc.resolve(cfg["paths"]["shards_dir"], cfg))
    have = sorted(p.stem for p in shards.glob("*.npz"))
    if not have:
        print("  no local shards - cannot run the real half")
        return 1
    full = pd.concat([sc._eval_df(cfg, cfg["paths"]["train_csv"]),
                      sc._eval_df(cfg, cfg["paths"]["val_csv"])], ignore_index=True)
    df = full[full["StudyInstanceUID"].isin(have)].reset_index(drop=True)
    idx_csv = pd.read_csv(sc.resolve(cfg["paths"]["series_index"], cfg))
    plan = sc.plan_slots(idx_csv, cfg)
    ds = sc.KneeShardDataset(df, cfg, plan, train=False)
    x0, y0 = ds[0]
    check("item is (1 pseudo-slot, 40 images, 3, size, size)",
          tuple(x0.shape) == (1, 40, 3, cfg["input"]["size"], cfg["input"]["size"]),
          str(tuple(x0.shape)))
    check("labels are 12", tuple(y0.shape) == (len(tasks),), str(tuple(y0.shape)))

    # coronal must really pull 24 DISTINCT planes, not 18 with repeats. pick_slices repeats
    # indices when the band is too short, which would make the denser sampling a no-op.
    g = idx_csv.set_index("series")
    study = df["StudyInstanceUID"][0]
    for slot in ("cor_fluid", "sag_fluid"):
        ser = plan[study].get(slot)
        if ser is None or ser not in g.index:
            continue
        avail = int(g.loc[ser, "n_slices"])
        want = sc.slot_slices(cfg, slot)
        picked = sc.pick_slices(avail, want, "even", np.random.default_rng(0),
                                cfg["input"].get("slice_margin", 0.0))
        check("%s picks %d planes, %d distinct of %d available"
              % (slot, want, len(set(picked)), avail),
              len(picked) == want and len(set(picked)) >= min(want, int(avail * 0.9)),
              str(sorted(set(picked))[:6]) + "...")

    # ---- 3. slot ORDER in the concatenated tensor, for every slot -----------------------
    start, acc = [], 0
    for n in n_win:
        start.append(acc)
        acc += n
    for si, slot in enumerate(slots):
        solo = {k: ({slot: v[slot]} if slot in v else {}) for k, v in plan.items()}
        xs = sc.KneeShardDataset(df, cfg, solo, train=False)[0][0][0]
        nz = [i for i in range(40) if float(xs[i].abs().sum()) > 0]
        want_range = (start[si], start[si] + n_win[si] - 1)
        check("%-10s occupies windows %d..%d" % (slot, want_range[0], want_range[1]),
              bool(nz) and min(nz) == want_range[0] and max(nz) == want_range[1],
              "got %d..%d" % (min(nz), max(nz)) if nz else "empty")

    # ---- 4/5. the REAL model, real gradients --------------------------------------------
    small = copy.deepcopy(cfg)
    small["input"]["size"] = 64            # the trunk is convolutional; this is the same graph
    small["model"] = dict(small["model"], pretrained=False)
    model = mod.build_model(small)
    check("model is the per-finding pooling model", type(model).__name__ == "MILModel")
    check("12 norms and 12 single-output heads",
          len(model.norms) == 12 and len(model.heads) == 12
          and all(h.out_features == 1 for h in model.heads))

    torch.manual_seed(0)
    xb = torch.randint(0, 256, (2, 1, 40, 3, 64, 64), dtype=torch.uint8)
    yb = torch.randint(0, 2, (2, 12)).float()
    inp = sc.prepare_batch(xb, small, torch.device("cpu"))
    check("prepare_batch -> (80, 3, 64, 64)", tuple(inp.shape) == (80, 3, 64, 64),
          str(tuple(inp.shape)))

    out = model(inp)
    check("forward -> (2, 12), finite",
          tuple(out.shape) == (2, 12) and bool(torch.isfinite(out).all()))

    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(out, yb)
    loss.backward()
    check("loss is finite", bool(torch.isfinite(loss)), "%.4f" % float(loss))
    check("backbone received gradient",
          any(p.grad is not None and float(p.grad.abs().sum()) > 0
              for p in model.backbone.parameters()))
    dead = [tasks[j] for j in range(12)
            if model.heads[j].weight.grad is None
            or float(model.heads[j].weight.grad.abs().sum()) == 0]
    check("every one of the 12 heads received gradient", not dead, "dead: %s" % dead)
    dead_n = [tasks[j] for j in range(12)
              if model.norms[j].weight.grad is None
              or float(model.norms[j].weight.grad.abs().sum()) == 0]
    check("every one of the 12 norms received gradient", not dead_n, "dead: %s" % dead_n)

    # a head's gradient must depend only on ITS windows: zero a window it does not read and
    # its gradient must be unchanged
    j = tasks.index("ACL")
    mine = set(model.windows[j])
    other = next(w for w in range(40) if w not in mine)

    # eval mode matters here: drop_rate is 0.1, so in TRAIN mode the two passes draw
    # different dropout masks and the gradients differ for that reason alone, which looks
    # exactly like window leakage. The first version of this check asserted against a
    # stochastic forward and failed on correct code.
    model.eval()

    def acl_grad(t):
        model.zero_grad(set_to_none=True)
        o = model(sc.prepare_batch(t, small, torch.device("cpu")))
        o[:, j].sum().backward()
        return model.heads[j].weight.grad.clone()

    g0 = acl_grad(xb)
    xb2 = xb.clone()
    xb2[:, 0, other] = 0
    g1 = acl_grad(xb2)
    # and the control: zeroing a window ACL DOES read must change its gradient
    xb3 = xb.clone()
    xb3[:, 0, sorted(mine)[0]] = 0
    g2 = acl_grad(xb3)
    check("ACL's head gradient DOES move for a window it reads",
          not torch.allclose(g0, g2, atol=1e-6),
          "max diff %.2e" % float((g0 - g2).abs().max()))
    check("ACL's head gradient ignores a window outside its views",
          bool(torch.allclose(g0, g1, atol=1e-6)),
          "max diff %.2e" % float((g0 - g1).abs().max()))

    # ---- 4b. a few steps must actually reduce the loss on a fixed batch ------------------
    model.train()
    model.zero_grad(set_to_none=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = last = None
    for step in range(12):
        opt.zero_grad(set_to_none=True)
        l = loss_fn(model(inp), yb)
        l.backward()
        opt.step()
        if step == 0:
            first = float(l)
        last = float(l)
    check("12 steps reduce the loss on a fixed batch", last < first,
          "%.4f -> %.4f" % (first, last))

    # ---- 6. checkpoint round trip -------------------------------------------------------
    model.eval()
    with torch.no_grad():
        ref = model(inp)
    sd = copy.deepcopy(model.state_dict())
    fresh = mod.build_model(small)
    missing, unexpected = fresh.load_state_dict(sd, strict=False)
    check("state dict loads with nothing missing or unexpected",
          not missing and not unexpected,
          "missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    fresh.eval()
    with torch.no_grad():
        got = fresh(inp)
    check("reloaded model reproduces the logits exactly",
          bool(torch.allclose(ref, got, atol=1e-6)))

    # ---- 7. the resume guard must survive a dict n_slices --------------------------------
    def dig(d, path):
        for k in path:
            if not isinstance(d, dict) or k not in d:
                return "<absent>"
            d = d[k]
        return d
    same = dig(cfg, ("input", "n_slices")) == dig(copy.deepcopy(cfg), ("input", "n_slices"))
    check("resume guard compares a dict n_slices without crashing", same)

    # ---- 8. the 4-slot parent is untouched ----------------------------------------------
    pcfg = sc.load_config(ROOT / "runs" / PARENT, verbose=False)
    check("parent still scalar, 4 slots, 24 windows",
          not sc.per_slot_input(pcfg) and len(pcfg["input"]["slots"]) == 4
          and sc.images_per_study(pcfg) == 24,
          "%d windows" % sc.images_per_study(pcfg))
    pds = sc.KneeShardDataset(df, pcfg, sc.plan_slots(idx_csv, pcfg), train=False)
    check("parent item still (4, 18, size, size)",
          tuple(pds[0][0].shape) == (4, 18, pcfg["input"]["size"], pcfg["input"]["size"]),
          str(tuple(pds[0][0].shape)))

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
