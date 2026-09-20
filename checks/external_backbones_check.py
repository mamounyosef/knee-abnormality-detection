r"""external_backbones_check.py - prove the non-timm weights actually land in the model.

The failure this guards against is silent. `load_state_dict(..., strict=False)` returns
happily when nothing matched, leaving the backbone at random init, and a randomly
initialised network still trains and still produces a plausible loss curve. We would then
attribute the result to MedSAM or Barlow Twins pretraining that never loaded.

So this does not check that loading "worked". It checks that every parameter tensor of the
model changed from its random initialisation, that the resampled position tensors are the
right shape AND carry the checkpoint's structure rather than noise, and that a deliberately
corrupted checkpoint is REJECTED rather than silently half-loaded.

    python checks/external_backbones_check.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import external_backbones as eb  # noqa: E402

WEIGHTS = ROOT / "data" / "pretrained"
OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def fingerprint(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def changed_fraction(before, model):
    after = model.state_dict()
    same = [k for k in before if torch.equal(before[k], after[k])]
    return 1.0 - len(same) / max(len(before), 1), same


def main():
    medsam = WEIGHTS / "medsam_vit_b.pth"
    barlow = WEIGHTS / "barlowtwins_resnet50.pth"
    if not medsam.exists() or not barlow.exists():
        print("  missing checkpoints under %s" % WEIGHTS)
        return 1

    # ---- MedSAM at both target resolutions ---------------------------------------------
    for size, want_grid, want_rel in ((336, 21, 41), (512, 32, 63)):
        torch.manual_seed(0)
        m = timm.create_model("samvit_base_patch16", pretrained=False,
                              num_classes=0, img_size=size)
        before = fingerprint(m)
        n = eb.load_medsam(m, medsam)
        frac, same = changed_fraction(before, m)
        check("MedSAM @%d fills every tensor (%d)" % (size, n),
              n == len(before), "%d of %d" % (n, len(before)))
        check("MedSAM @%d every tensor moved off random init" % size,
              frac > 0.999, "%.1f%% changed, unchanged: %s" % (100 * frac, same[:3]))

        pe = m.state_dict()["pos_embed"]
        check("MedSAM @%d pos_embed resampled to %dx%d" % (size, want_grid, want_grid),
              tuple(pe.shape) == (1, want_grid, want_grid, 768), str(tuple(pe.shape)))
        rp = m.state_dict()["blocks.2.attn.rel_pos_h"]
        check("MedSAM @%d rel_pos resampled to %d" % (size, want_rel),
              tuple(rp.shape) == (want_rel, 64), str(tuple(rp.shape)))

        # a resampled tensor must be a SMOOTH reduction of the original, not noise:
        # neighbouring entries of a real table correlate; random init does not.
        d = rp[1:] - rp[:-1]
        rnd = torch.randn_like(rp)
        check("MedSAM @%d resampled rel_pos is smooth, not noise" % size,
              float(d.abs().mean()) < float((rnd[1:] - rnd[:-1]).abs().mean()),
              "step %.4f vs random %.4f"
              % (float(d.abs().mean()), float((rnd[1:] - rnd[:-1]).abs().mean())))

        m.eval()
        with torch.no_grad():
            o = m(torch.zeros(2, 3, size, size))
        check("MedSAM @%d forward -> (2, 256), finite" % size,
              tuple(o.shape) == (2, 256) and bool(torch.isfinite(o).all()),
              str(tuple(o.shape)))

    # ---- Barlow Twins --------------------------------------------------------------------
    torch.manual_seed(0)
    r = timm.create_model("resnet50", pretrained=False, num_classes=0)
    before = fingerprint(r)
    n = eb.load_barlowtwins(r, barlow)
    frac, same = changed_fraction(before, r)
    check("Barlow Twins fills the backbone (%d tensors)" % n, n > 300, str(n))
    check("Barlow Twins every tensor moved off random init", frac > 0.999,
          "%.1f%% changed, unchanged: %s" % (100 * frac, same[:3]))
    r.eval()
    with torch.no_grad():
        o = r(torch.zeros(2, 3, 336, 336))
    check("Barlow Twins forward -> (2, 2048), finite",
          tuple(o.shape) == (2, 2048) and bool(torch.isfinite(o).all()), str(tuple(o.shape)))

    # the loaded weights must differ from timm's SUPERVISED resnet50, or we have just
    # re-added a member we already have
    sup = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0)
    a, b = r.state_dict(), sup.state_dict()
    diff = float(torch.stack([(a[k].float() - b[k].float()).abs().mean()
                              for k in a if k in b and a[k].shape == b[k].shape]).mean())
    check("Barlow Twins differs from supervised resnet50", diff > 1e-3,
          "mean abs diff %.4f" % diff)

    # ---- a broken checkpoint must be REJECTED, not half-loaded -------------------------
    import tempfile
    bad = dict(torch.load(str(barlow), map_location="cpu", weights_only=False))
    bad = {("zzz." + k): v for k, v in list(bad.items())[:20]}     # rename so nothing matches
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as fh:
        torch.save(bad, fh.name)
        tmp = fh.name
    r2 = timm.create_model("resnet50", pretrained=False, num_classes=0)
    try:
        eb.load_barlowtwins(r2, tmp)
        check("a non-matching checkpoint raises instead of silently loading nothing", False)
    except RuntimeError:
        check("a non-matching checkpoint raises instead of silently loading nothing", True)
    Path(tmp).unlink(missing_ok=True)

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
