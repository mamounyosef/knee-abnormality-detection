r"""confidence_weight_check.py - prove the confidence-weighted loss does what it claims.

Four things that would fail silently and be discovered only as a bad val score:

  1. The TARGET is untouched. This weights cells, it does not sharpen labels. A run that
     quietly moved 0.95 toward 1.0 would be a different experiment than the one intended.
  2. A p=0.5 cell contributes ZERO gradient at floor=0, and a p=0.95 cell contributes 0.9.
  3. Normalising by sum(w) keeps the loss SCALE comparable to plain BCE, so the effective
     learning rate does not drop just because cells were discounted.
  4. With hard 0/1 targets every weight is 1, which makes the whole thing a no-op - so the
     config is refused rather than running a null experiment for an hour on an A100.

CPU only.

    python checks/confidence_weight_check.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


base = nn.BCEWithLogitsLoss(reduction="none")
cw = sc.ConfidenceWeighted(base, power=1.0, floor=0.0)

# --- 1. the weights themselves -----------------------------------------------------
print("\nweights from |2p-1|")
t = torch.tensor([[0.95, 0.75, 0.50, 0.25, 0.05]])
w = (2 * t - 1).abs()
print("    p      %s" % "  ".join("%.2f" % x for x in t[0]))
print("    weight %s" % "  ".join("%.2f" % x for x in w[0]))
check("p=0.95 -> 0.90", abs(w[0, 0].item() - 0.90) < 1e-6)
check("p=0.50 -> 0.00 (a coin flip is ignored)", w[0, 2].item() == 0.0)
check("p=0.05 -> 0.90 (confident NEGATIVE counts too)", abs(w[0, 4].item() - 0.90) < 1e-6)

# --- 2. targets are NOT modified ---------------------------------------------------
print("\ntargets are untouched")
tgt = torch.tensor([[0.95, 0.50, 0.05]])
before = tgt.clone()
logits = torch.zeros_like(tgt, requires_grad=True)
cw(logits, tgt).backward()
check("the target tensor is unchanged after a backward pass",
      torch.equal(tgt, before), str(tgt.tolist()))

# --- 3. a coin-flip cell gets no gradient -----------------------------------------
print("\ngradient reaches confident cells only")
logits = torch.zeros(1, 3, requires_grad=True)
cw(logits, torch.tensor([[0.95, 0.50, 0.05]])).backward()
g = logits.grad[0]
print("    grad: %s" % ["%.4f" % x for x in g])
check("p=0.50 cell has EXACTLY zero gradient", g[1].item() == 0.0)
check("p=0.95 and p=0.05 cells do get gradient", g[0].item() != 0 and g[2].item() != 0)
check("the two confident cells pull in OPPOSITE directions",
      g[0].item() * g[2].item() < 0)

# --- 4. scale is comparable to plain BCE ------------------------------------------
print("\nloss scale")
torch.manual_seed(0)
lg = torch.randn(64, 12)
# all-confident targets: weighting should then be a no-op up to the constant weight
conf = torch.where(torch.rand(64, 12) > 0.5, torch.full((64, 12), 0.95),
                   torch.full((64, 12), 0.05))
plain = nn.BCEWithLogitsLoss()(lg, conf)
weighted = cw(lg, conf)
check("with ALL cells equally confident, matches plain BCE",
      abs(plain.item() - weighted.item()) < 1e-5,
      "plain %.5f vs weighted %.5f" % (plain.item(), weighted.item()))

mixed = torch.rand(64, 12)
m_plain = nn.BCEWithLogitsLoss()(lg, mixed).item()
m_w = cw(lg, mixed).item()
check("with mixed confidence the loss stays the same order of magnitude",
      0.3 < m_w / m_plain < 3.0, "plain %.4f vs weighted %.4f" % (m_plain, m_w))

# --- 5. floor keeps ambiguous cells alive -----------------------------------------
print("\nfloor")
cwf = sc.ConfidenceWeighted(base, power=1.0, floor=0.2)
# the logit must NOT be 0 here. At target=0.5 and logit=0 the BCE gradient is
# sigmoid(0) - 0.5 = 0 no matter what the weight is, so a zero gradient would say nothing
# about the floor. Offset the logit so the cell has a real error to propagate.
logits = torch.full((1, 3), 1.5, requires_grad=True)
cwf(logits, torch.tensor([[0.95, 0.50, 0.05]])).backward()
g_floor = logits.grad[0, 1].item()

logits0 = torch.full((1, 3), 1.5, requires_grad=True)
sc.ConfidenceWeighted(base, power=1.0, floor=0.0)(
    logits0, torch.tensor([[0.95, 0.50, 0.05]])).backward()
check("floor=0.2 gives a p=0.50 cell a non-zero gradient", g_floor != 0.0,
      "%.5f" % g_floor)
check("floor=0.0 gives that same cell EXACTLY zero", logits0.grad[0, 1].item() == 0.0,
      "%.5f" % logits0.grad[0, 1].item())

# --- 6. refuses hard targets -------------------------------------------------------
print("\nguard against a null experiment")
cfg = {"training": {"loss": {"name": "bce", "pos_weight": None,
                             "confidence_weight": {"enable": True}}},
       "target": {"mode": "hard"}, "tasks": ["a"]}
try:
    sc.build_loss(cfg, None, "cpu")
    check("hard targets are refused", False, "it built a no-op loss instead")
except SystemExit as e:
    check("hard targets are refused with a clear message", "soft" in str(e))

cfg["target"]["mode"] = "soft"
got = sc.build_loss(cfg, None, "cpu")
check("soft targets build the wrapper", isinstance(got, sc.ConfidenceWeighted))

print("\n" + "=" * 72)
if fails:
    print("FAILED: " + ", ".join(fails))
    raise SystemExit(1)
print("all checks passed - cells are weighted, targets are not touched")
