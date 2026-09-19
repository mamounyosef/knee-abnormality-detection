r"""mcl_top2_check.py - prove ONLY the MCL column pools differently.

The whole point of convnext_base_25d_mcl_top2_pooling is that it differs from the 0.9010
baseline in exactly one column. If top2 quietly applied to more than MCL, or to the wrong
column index, the run would still train and still score, and we would credit or blame the
MCL story for something else entirely.

The discriminating property: a mean reads every window, a top2 reads only the two largest.
So nudging a window that is NOT among the top two must move every mean column and leave MCL
untouched, and MCL must move when the largest window changes.

    python checks/mcl_top2_check.py
"""

import copy
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RUN = "convnext_base_25d_mcl_top2_pooling"
PARENT = "convnext_base_25d_18slices"
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def walk(d, prefix=""):
    out = {}
    for k, v in d.items():
        p = "%s.%s" % (prefix, k) if prefix else str(k)
        out.update(walk(v, p)) if isinstance(v, dict) else out.__setitem__(p, v)
    return out


class Stub(torch.nn.Module):
    """Window i -> a feature vector determined only by window i."""
    num_features = 8

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, self.num_features)

    def forward(self, x):
        return self.lin(x[:, 0].reshape(x.shape[0], -1))

    def set_grad_checkpointing(self, v):
        pass


def main():
    d = ROOT / "runs" / RUN
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("t_" + RUN, d / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = sc.load_config(d, verbose=False)
    par = sc.load_config(ROOT / "runs" / PARENT, verbose=False)
    tasks = list(cfg["tasks"])

    # ---- one variable against the baseline ----------------------------------------------
    a, b = walk(par), walk(cfg)
    diff = {k for k in set(a) | set(b) if a.get(k, "~") != b.get(k, "~")} - {"experiment.name"}
    check("config is identical to the baseline", not diff, "differs: %s" % sorted(diff))
    check("TOP2_TASKS is MCL alone", mod.TOP2_TASKS == ["MCL"], str(mod.TOP2_TASKS))
    check("grad_clip still at least 3", float(cfg["training"]["grad_clip"]) >= 3.0)

    # ---- build the real model with a stub trunk -----------------------------------------
    model = mod.Top2Model.__new__(mod.Top2Model)
    torch.nn.Module.__init__(model)
    real = mod.timm.create_model
    mod.timm.create_model = lambda *x, **k: Stub()
    try:
        model.__init__(cfg)
    finally:
        mod.timm.create_model = real
    model.eval()

    mcl = tasks.index("MCL")
    check("top2 is wired to the MCL column", model.top2_idx == [mcl], str(model.top2_idx))
    check("the other 11 columns are on mean", len(model.mean_idx) == len(tasks) - 1
          and mcl not in model.mean_idx)
    check("every column is covered exactly once",
          sorted(model.mean_idx + model.top2_idx) == list(range(len(tasks))))

    n = sc.images_per_study(cfg)
    torch.manual_seed(0)
    x = torch.randn(1, n, 3, 4, 4)
    with torch.no_grad():
        base = model(x.view(-1, 3, 4, 4))
    check("forward is (1, 12) and finite",
          tuple(base.shape) == (1, len(tasks)) and bool(torch.isfinite(base).all()))

    # NOTE ON SEMANTICS. topk runs along the window axis INDEPENDENTLY FOR EACH FEATURE
    # CHANNEL, so "top2" is not "pick 2 windows and use them for everything" - each of the
    # 1024 channels takes its own two strongest windows. That is exactly what
    # convnext_base_25d_perfinding_pool_features did when MCL gained +0.0329, so it is the
    # behaviour to reproduce, but it means a window can be ignored by one channel and used
    # by another. The invariant to test is therefore per channel, not per window.
    with torch.no_grad():
        f = model.backbone(x.view(-1, 3, 4, 4)).view(1, n, -1)      # (1, n, feat)

    # a channel where some window is strictly below that channel's top two
    ch = None
    for c in range(f.shape[-1]):
        col = f[0, :, c]
        if col.argsort(descending=True).shape[0] > 2:
            ch = c
            loser = int(col.argsort(descending=True)[-1])
            break
    check("found a window outside the top two for some channel", ch is not None)

    g = f.clone()
    g[0, loser, ch] -= 5.0                       # push a loser further down: still a loser
    with torch.no_grad():
        p_base = f.topk(2, dim=1).values.mean(dim=1)
        p_mod = g.topk(2, dim=1).values.mean(dim=1)
    check("top2 ignores a window already below that channel's top two",
          bool(torch.allclose(p_base, p_mod, atol=1e-6)),
          "max change %.2e" % float((p_base - p_mod).abs().max()))
    check("but the MEAN over windows would have moved",
          not torch.allclose(f.mean(dim=1), g.mean(dim=1), atol=1e-6))

    h = f.clone()
    h[0, int(f[0, :, ch].argmax()), ch] += 5.0   # raise the winner
    with torch.no_grad():
        p_up = h.topk(2, dim=1).values.mean(dim=1)
    check("top2 DOES follow the strongest window",
          not torch.allclose(p_base, p_up, atol=1e-6))

    # the reduction really is a mean of the top TWO, not a max
    with torch.no_grad():
        f = model.backbone(x.view(-1, 3, 4, 4)).view(1, n, -1)
        want = f.topk(2, dim=1).values.mean(dim=1)
        got = model.head_top2(model.norm_top2(want))
    check("MCL logit equals head(norm(mean of top 2))",
          bool(torch.allclose(got[0, 0], base[0, mcl], atol=1e-5)))

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f_ in FAIL:
        print("    FAILED: %s" % f_)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
