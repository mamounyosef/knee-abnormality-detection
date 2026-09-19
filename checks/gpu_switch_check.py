r"""gpu_switch_check.py - prove swapping the GPU mid-training resumes onto the same trajectory.

Changing `modal.gpu` on a run that is already training is only safe because the resume guard
does not watch it: the guard raises when a resumed config differs in anything that would make
the two halves of the run describe different experiments. The GPU is not one of those things.

The flip side is the danger this check exists for. The guard ALSO does not watch
`dataloader.batch_size` or `accum_steps`, so changing those mid-run would silently resume
under a different effective batch and nobody would be told. This asserts they did not move.

    python checks/gpu_switch_check.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runs"))
import shared_code as sc  # noqa: E402

# run -> (batch_size, accum_steps, peak GB at THAT batch, expected card)
RUNS = {"convnext_base_25d_perfinding_slots_and_pooling": (8, 1, 88.0, "H200"),
        "convnext_base_25d_mcl_top2_pooling": (8, 1, 53.5, "H100"),
        "convnext_base_25d_left_knees_mirrored": (8, 1, 53.5, "H100")}
CARD_GB = {"H100": 80, "A100-80GB": 80, "H200": 141, "A100-40GB": 40, "B200": 180}

OK, FAIL = [], []


def check(name, cond, detail=""):
    (OK if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, ("  " + detail) if detail else ""))


def main():
    # the guard's watch list, read from the engine rather than copied - a copy would drift
    src = (ROOT / "runs" / "shared_code.py").read_text(encoding="utf-8")
    block = src[src.index("_guard = ["):]
    block = block[:block.index("]")]
    watched = re.findall(r'\("([^"]+)",', block)
    check("resume guard does NOT watch the GPU",
          not any(w.startswith("modal") or "gpu" in w for w in watched),
          "watches: %s" % ", ".join(watched))
    check("resume guard does NOT watch batch_size or accum_steps",
          not any("batch" in w or "accum" in w for w in watched))
    print("    -> so a GPU swap resumes cleanly, and a batch change would NOT be caught."
          "\n       That is exactly why the next assertions matter.\n")

    for run, (bs, accum, peak, card) in RUNS.items():
        cfg = sc.load_config(ROOT / "runs" / run, verbose=False)
        dl, gpu = cfg["dataloader"], cfg["modal"]["gpu"]
        check("%-46s batch %d x accum %d = 8 studies" % (run[:46], dl["batch_size"],
                                                         dl["accum_steps"]),
              dl["batch_size"] == bs and dl["accum_steps"] == accum
              and dl["batch_size"] * dl["accum_steps"] == 8)
        check("%-46s gpu is %s" % (run[:46], card), gpu == card, gpu)
        cap = CARD_GB.get(gpu, 0)
        check("%-46s peak %.1f GB fits %s (%d GB)" % (run[:46], peak, gpu, cap),
              peak < cap * 0.95, "%.0f%% of the card" % (100 * peak / cap))

        # the whole optimizer step is resident in one pass when accum is 1, so the peak
        # above IS the peak; with accum > 1 only a fraction is resident at a time
        n_img = dl["batch_size"] * sc.images_per_study(cfg)
        check("%-46s %d images resident per pass" % (run[:46], n_img), True,
              "%.3f GB/image" % (peak / n_img))

    print("\n  %d passed, %d failed" % (len(OK), len(FAIL)))
    for f in FAIL:
        print("    FAILED: %s" % f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
