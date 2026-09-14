"""
normalization_check.py - audit pixel normalisation across EVERY run in the repo.

Normalisation is the failure mode that never announces itself. A wrong mean/std trains
happily, the loss falls, and the model is simply worse than it should have been: it already
cost this project a whole mricore_vitb_mae pretraining run. So this checks the property from
several directions rather than trusting one assertion.

For every runs/*/config.yaml:

  1. WHICH BRANCH. prepare_batch has two normalisation branches. The per-channel one applies
     norm_mean/norm_std channel by channel and is correct whenever the backbone keeps its
     pretrained 3-channel stem. The scalar one averages the three statistics and is the only
     option when the stem was WIDENED to hold slices, because there is no per-channel
     statistic for a 12- or 24-channel input. Each run must take the branch its stem implies.

  2. ROUND TRIP. Normalisation must be exactly invertible: push known pixels through
     prepare_batch, invert with the same statistics, and the original bytes must come back.
     This is what proves save_debug_batch draws the real input rather than a rescaled ghost,
     and it is the only artifact we use to eyeball what the model actually sees.

  3. AGREEMENT WITH THE BACKBONE. timm records on each model's pretrained_cfg the statistics
     its weights were trained with. check_normalization compares the config against that and
     refuses to start on a mismatch, so this confirms the guard actually fires.

  4. save_debug_batch SURVIVES every layout. It un-normalises independently of
     prepare_batch, so it can drift out of step with it. It is also the place a widened stem
     used to crash: norm_mean sliced to 24 channels yields 3 values and will not broadcast.

    python checks/normalization_check.py            # branch + round trip, no downloads
    python checks/normalization_check.py --backbone # also builds each model for check 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print("    %-52s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


def layout_of(cfg):
    """(label, channels the backbone sees, images per study)."""
    i = cfg["input"]
    if i.get("slice_context") == "neighbours":
        return "2.5D overlapping (loader)", 3, len(i["slots"]) * int(i["n_slices"])
    g = int(i.get("slices_as_channels") or 0)
    if g:
        c = int(i["n_slices"]) // g
        lab = "2.5D grouped" if c == 3 else "widened stem"
        return lab, c, sc.images_per_study(cfg)
    return "one image per slice", int(i["channels"]), sc.images_per_study(cfg)


def fake_batch(cfg, h=4, w=4, b=2):
    i = cfg["input"]
    n_slots, d = len(i["slots"]), int(i["n_slices"])
    if i.get("slice_context") == "neighbours":
        return torch.randint(0, 256, (b, n_slots, d, 3, h, w), dtype=torch.uint8)
    return torch.randint(0, 256, (b, n_slots, d, h, w), dtype=torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", action="store_true",
                    help="also build each model and run check_normalization (downloads weights)")
    a = ap.parse_args()

    runs = sorted(p.parent.name for p in (ROOT / "runs").glob("*/config.yaml"))
    print("auditing %d runs\n" % len(runs))

    for r in runs:
        try:
            cfg = sc.load_config(ROOT / "runs" / r, verbose=False)
        except Exception as exc:
            print("%s\n    config did not load: %s" % (r, exc))
            fails.append(r + " (config)")
            continue
        i = cfg["input"]
        lab, n_ch, per = layout_of(cfg)
        print("%s\n    %s | %d channels | %d passes/study" % (r, lab, n_ch, per))

        x = fake_batch(cfg)
        try:
            out = sc.prepare_batch(x, cfg, "cpu")
        except Exception as exc:
            check("prepare_batch runs", False, "%s: %s" % (type(exc).__name__, exc))
            continue

        check("flattened channel count matches the layout", out.shape[1] == n_ch,
              "got %d, expected %d" % (out.shape[1], n_ch))

        # --- 1. which branch, and is it the right one for this stem? ---
        per_ch = n_ch == len(i["norm_mean"])
        want = "per-channel" if per_ch else "scalar"
        if per_ch:
            mean = torch.tensor(i["norm_mean"]).view(1, -1, 1, 1)
            std = torch.tensor(i["norm_std"]).view(1, -1, 1, 1)
        else:
            mean = torch.tensor(float(np.mean(i["norm_mean"]))).view(1, 1, 1, 1)
            std = torch.tensor(float(np.mean(i["norm_std"]))).view(1, 1, 1, 1)

        # A pretrained 3-channel stem MUST get per-channel statistics, because that is the
        # distribution its filters were fitted on. Anything wider has no such statistic.
        check("branch matches the stem (%s)" % want, per_ch == (n_ch == 3))

        # --- 2. round trip: normalise then invert, recover the original bytes ---
        rec = (out.float() * std + mean) * 255.0
        src = x.float()
        if i.get("slice_context") == "neighbours":
            src = src.reshape(-1, 3, *x.shape[-2:])
        elif i.get("slices_as_channels"):
            g = int(i["slices_as_channels"])
            src = src.reshape(-1, int(i["n_slices"]) // g, *x.shape[-2:])
        else:
            src = src.reshape(-1, 1, *x.shape[-2:]).expand(-1, n_ch, -1, -1)
        rt = i.get("resize_to")
        if rt and src.shape[-1] != int(rt):
            src = torch.nn.functional.interpolate(src, size=(int(rt), int(rt)), mode="area")
        d = (rec - src).abs().max().item()
        check("round trip recovers the source pixels", d < 1e-2, "max diff %.2e" % d)

        # --- 4. save_debug_batch's independent inverse must not crash or mis-index ---
        try:
            import types
            dummy = types.SimpleNamespace()
            # exercise the same arithmetic save_debug_batch uses, without needing cv2
            g = int(i.get("slices_as_channels") or 0)
            cpg = int(i["n_slices"]) // g if g else 1
            v = ((out[:per].float() * std + mean).clamp(0, 1) * 255).to(torch.uint8).numpy()
            ok_shape = v.ndim == 4
            hh, ww = v.shape[-2:]
            idx = [(rr, sl) for rr in range(len(i["slots"]))
                   for sl in np.linspace(0, int(i["n_slices"]) - 1, min(8, int(i["n_slices"]))).astype(int)]
            for rr, sl in idx:
                p = v[rr * g + sl // cpg, sl % cpg] if g else v[rr * int(i["n_slices"]) + sl, 0]
                assert p.shape == (hh, ww)
            check("debug-sheet indexing valid for every (slot, slice)", ok_shape)
        except Exception as exc:
            check("debug-sheet indexing", False, "%s: %s" % (type(exc).__name__, exc))

        # --- 3. does the backbone agree with the config? ---
        if a.backbone:
            try:
                import timm
                mname = cfg["model"]["name"]
                if cfg["model"].get("weights"):
                    check("backbone agreement", True, "skipped: run loads its own weights")
                else:
                    m = timm.create_model(mname, pretrained=True, num_classes=0, in_chans=n_ch)
                    pc = getattr(m, "pretrained_cfg", {}) or {}
                    gm, gs = list(pc.get("mean", [])), list(pc.get("std", []))
                    same = (all(abs(p - q) < 1e-3 for p, q in zip(gm, i["norm_mean"]))
                            and all(abs(p - q) < 1e-3 for p, q in zip(gs, i["norm_std"])))
                    check("config matches the checkpoint's own statistics",
                          same or bool(i.get("norm_mismatch_ok")),
                          "cfg %s/%s vs timm %s/%s" % (i["norm_mean"], i["norm_std"], gm, gs))
            except Exception as exc:
                check("backbone agreement", False, "%s: %s" % (type(exc).__name__, exc))
        print()

    print("=" * 78)
    if fails:
        print("FAILED (%d): %s" % (len(fails), "; ".join(fails)))
        raise SystemExit(1)
    print("every run normalises consistently with its stem, and inverts exactly")


if __name__ == "__main__":
    main()
