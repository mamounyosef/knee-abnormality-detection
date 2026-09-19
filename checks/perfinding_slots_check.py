r"""perfinding_slots_check.py - prove each finding reads ITS views and no others.

The whole claim of convnext_base_25d_perfinding_slots_and_pooling is that the ACL logit
never sees a coronal window and the OA logits never see a sagittal one. Nothing about the
shapes can show that: the model produces a correct-looking (B, 12) whether the window
indices are right, shifted by one slot, or reversed. A wrong SLOT_MAP would train happily
and score plausibly, and we would attribute the result to an idea that was never tested.

So this tests DEPENDENCE directly. For each finding, perturb the windows it is supposed to
ignore and assert its logit does not move; perturb the windows it is supposed to read and
assert it does. Run on the real model with a stub backbone, so the window bookkeeping under
test is the real one.

Also checks the slot-major layout assumption the index arithmetic rests on, and that the
pooling kinds behave as named.

    python checks/perfinding_slots_check.py
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = "convnext_base_25d_perfinding_slots_and_pooling"
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def load_train():
    d = ROOT / "runs" / RUN
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("t_" + RUN, d / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubBackbone(torch.nn.Module):
    """Identity-ish trunk: window i -> a feature vector that depends only on window i.

    Real weights would work too but cost a minute per forward; what is under test is the
    index bookkeeping, and this makes any leak between windows immediately visible.
    """

    num_features = 32

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, self.num_features)

    def forward(self, x):                      # (N, 3, 4, 4) -> (N, 32)
        return self.lin(x[:, 0].reshape(x.shape[0], -1))

    def set_grad_checkpointing(self, v):       # never called here, present for parity
        pass


def main():
    mod = load_train()
    cfg = sc.load_config(ROOT / "runs" / RUN, verbose=False)
    tasks, slots = list(cfg["tasks"]), list(cfg["input"]["slots"])
    per_study = sc.images_per_study(cfg)
    # slots may contribute different window counts (coronal samples more slices), so a
    # slot's first window is the running total of the ones before it, not si * stride
    n_win = [sc.slot_groups(cfg, s) for s in slots]
    start, acc = [], 0
    for n in n_win:
        start.append(acc)
        acc += n
    print("  %d windows over %d slots: %s"
          % (per_study, len(slots),
             ", ".join("%s=%d" % (s, n) for s, n in zip(slots, n_win))))

    # ---- the pooling primitives do what their names say ---------------------------------
    t = torch.tensor([[[1.0], [5.0], [3.0], [2.0]]])          # (1, 4, 1)
    check("_reduce mean", abs(float(mod._reduce(t, "mean")) - 2.75) < 1e-6)
    check("_reduce max", abs(float(mod._reduce(t, "max")) - 5.0) < 1e-6)
    check("_reduce top2 averages the two largest", abs(float(mod._reduce(t, "top2")) - 4.0) < 1e-6)

    # ---- every finding is assigned, and to real slots ----------------------------------
    for t_name in tasks:
        kind = next((k for k, v in mod.POOL_MAP.items() if t_name in v), None)
        check("%-17s has a pooling kind" % t_name, kind is not None, str(kind))
        allowed = mod.SLOT_MAP.get(t_name)
        check("%-17s has views, all of them real" % t_name,
              bool(allowed) and all(s in slots for s in allowed),
              ",".join(allowed or []))

    # ---- build the real model with the stub trunk --------------------------------------
    model = mod.MILModel.__new__(mod.MILModel)
    torch.nn.Module.__init__(model)
    real_create = mod.timm.create_model
    mod.timm.create_model = lambda *a, **k: StubBackbone()
    try:
        model.__init__(cfg)
    finally:
        mod.timm.create_model = real_create
    model.eval()

    # the index arithmetic assumes windows are laid out SLOT-MAJOR
    for j, t_name in enumerate(tasks):
        allowed = [s for s in mod.SLOT_MAP[t_name] if s in slots]
        want = sorted(start[si] + k
                      for si, s in enumerate(slots) if s in allowed
                      for k in range(n_win[si]))
        check("%-17s window indices are its slots, slot-major" % t_name,
              sorted(model.windows[j]) == want,
              "%d windows" % len(model.windows[j]))

    # ---- THE REAL TEST: perturb the windows it should ignore ---------------------------
    torch.manual_seed(0)
    x = torch.randn(1, per_study, 3, 4, 4)
    with torch.no_grad():
        base = model(x.view(-1, 3, 4, 4))
    check("baseline forward is (1, 12) and finite",
          tuple(base.shape) == (1, len(tasks)) and bool(torch.isfinite(base).all()))

    for j, t_name in enumerate(tasks):
        mine = set(model.windows[j])
        others = [w for w in range(per_study) if w not in mine]

        xb = x.clone()
        for w in others:
            xb[0, w] += 10.0                       # blow up every window it must ignore
        with torch.no_grad():
            out = model(xb.view(-1, 3, 4, 4))
        moved = float((out[0, j] - base[0, j]).abs())
        check("%-17s IGNORES the views it was not given" % t_name, moved < 1e-4,
              "moved %.2e over %d other windows" % (moved, len(others)))

        xa = x.clone()
        for w in mine:
            xa[0, w] += 10.0
        with torch.no_grad():
            out = model(xa.view(-1, 3, 4, 4))
        moved = float((out[0, j] - base[0, j]).abs())
        check("%-17s DOES read the views it was given" % t_name, moved > 1e-4,
              "moved %.2e over %d own windows" % (moved, len(mine)))

    # ---- one head per finding, not one per group ---------------------------------------
    check("one norm and one head per finding",
          len(model.norms) == len(tasks) and len(model.heads) == len(tasks),
          "%d norms, %d heads" % (len(model.norms), len(model.heads)))
    check("each head emits exactly one logit",
          all(h.out_features == 1 for h in model.heads))

    # ---- effective batch still matches every other run ---------------------------------
    dl = cfg["dataloader"]
    check("effective batch is still 8 studies per update",
          dl["batch_size"] * dl["accum_steps"] == 8,
          "%d x %d" % (dl["batch_size"], dl["accum_steps"]))

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
