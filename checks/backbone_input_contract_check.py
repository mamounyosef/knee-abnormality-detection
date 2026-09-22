"""backbone_input_contract_check.py - does each run feed its backbone what it was trained on?

A pretrained trunk has an input contract: the pixel normalisation it was trained with and
the resolution it was trained at. Break the first and every input is shifted by a fraction
of a standard deviation, silently. Break the second with a LEARNED position table and the
table has to be resampled, which is the bug class that cost us on Qwen (wrong interpolation
mode) and MedSAM (missing antialias).

This reads the contract from timm's own pretrained_cfg for the EXACT weight tag each run
loads, and compares it to what the run's config actually does.

    python checks/backbone_input_contract_check.py
"""
import contextlib
import io
import sys
from pathlib import Path

import timm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

RUNS = ["siglip2_base_perslice_384", "efficientnetv2_l_384", "eva02_large_448", "maxvit_base_384",
        "focalnet_large_336", "medimageinsight_davit_huge_336"]
PASS, FAIL = [], []


def ck(n, ok, d=""):
    (PASS if ok else FAIL).append(n)
    print("  %s %-54s %s" % ("PASS" if ok else "FAIL", n, d))


def close(a, b, tol=1e-4):
    return len(a) == len(b) and all(abs(float(x) - float(y)) < tol for x, y in zip(a, b))


for run in RUNS:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        c = sc.load_config(ROOT / "runs" / run)
    name, i = c["model"]["name"], c["input"]
    print("\n%s  (%s)" % (run, name))
    if name == "davit_huge_fl":
        print("  external weights (MedImageInsight); contract taken from its own config")
        continue
    m = timm.create_model(name, pretrained=False, num_classes=0)
    pc = m.pretrained_cfg
    native = pc.get("input_size")[-1]
    test = (pc.get("test_input_size") or pc.get("input_size"))[-1]
    ck("normalisation mean matches the pretrained weights",
       close(i["norm_mean"], pc["mean"]), "%s vs %s" % (list(i["norm_mean"]), list(pc["mean"])))
    ck("normalisation std matches the pretrained weights",
       close(i["norm_std"], pc["std"]), "%s vs %s" % (list(i["norm_std"]), list(pc["std"])))
    run_res = int(c["input"].get("resize_to") or i["size"])
    pos = [k for k, _ in m.named_parameters()
           if "pos_embed" in k or "relative_position" in k or "rel_pos" in k]
    learned_abs = any(k.endswith("pos_embed") for k in pos)
    print("  resolution: trained at %d%s, this run feeds %d"
          % (native, "" if test == native else " (evaluated at %d)" % test, run_res))
    print("  position tensors: %s" % (pos[:2] if pos else "none"))
    ck("run resolution equals the trained resolution",
       run_res == native, "%d vs %d" % (run_res, native))
    if learned_abs and run_res != native:
        print("  WARNING: a LEARNED absolute position table resampled %d -> %d" % (native, run_res))

print("\n" + "=" * 72)
print("  %d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("    FAILED: %s" % f)
sys.exit(1 if FAIL else 0)
