r"""prune_artifacts.py - reclaim disk without losing anything that matters.

Three jobs, each recoverable:

  1. DEBUG IMAGES, thinned to ~25% PER RUN, evenly spaced in step order. Not random: the
     first and last are always kept and the survivors are spread across the run, so the
     history still reads as a trajectory (early garbage -> late clean batches) instead of a
     random handful. These are previews of preprocessed batches; nothing reads them and a
     run regenerates them.
  2. KAGGLE BUNDLES, deleted outright. Already gitignored as "rebuilt from checkpoints on
     demand" - submit_ensemble.py repackages them on every submission.
  3. RESUME-ONLY CHECKPOINTS (last.pt, step*.pt) for runs that are finished or idle. The
     Modal volumes hold these; local copies exist only to hand a run to another workspace.

NEVER touched: every best.pt (what submit_ensemble packages), the resume files of runs that
are mid-flight, predictions/*.npz (every ensembling script reads them), logs and configs.

    python runs/others/prune_artifacts.py            # dry run, prints what would go
    python runs/others/prune_artifacts.py --apply
"""

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs"

# runs whose last.pt / step*.pt are still needed for a hand-off
IN_FLIGHT = {"dinov2_vits14_25d_f0", "dinov2_vits14_25d_f4"}
KEEP_FRACTION = 0.25


def step_of(name):
    """Sort key from a debug image name like run008_e10_s4400.png."""
    m = re.search(r"_e(\d+)_s(\d+)", name)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(\d+)", name)
    return (0, int(m.group(1)) if m else 0)


def evenly_spaced(items, frac):
    """Indices to KEEP: first, last, and the rest spread evenly through the ordering."""
    n = len(items)
    if n <= 2:
        return set(range(n))
    k = max(2, round(n * frac))
    # linspace over [0, n-1] inclusive, so both ends always survive
    idx = {round(i * (n - 1) / (k - 1)) for i in range(k)}
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete")
    ap.add_argument("--fraction", type=float, default=KEEP_FRACTION)
    a = ap.parse_args()

    doomed = []

    # --- 1. debug images, thinned per run folder ----------------------------------
    print("debug images (keeping ~%d%% per folder, evenly spaced, ends always kept)"
          % round(a.fraction * 100))
    for d in sorted(RUNS.glob("*/results/debug*images")):
        imgs = sorted([p for p in d.iterdir() if p.is_file()], key=lambda p: step_of(p.name))
        if not imgs:
            continue
        keep = evenly_spaced(imgs, a.fraction)
        drop = [p for i, p in enumerate(imgs) if i not in keep]
        doomed += drop
        gb = sum(p.stat().st_size for p in drop) / 1e9
        run = d.relative_to(RUNS).parts[0]
        print("  %-42s %4d -> %3d kept, %4d dropped (%.2f GB)"
              % (run + "/" + d.name, len(imgs), len(imgs) - len(drop), len(drop), gb))

    # --- 2. kaggle bundles --------------------------------------------------------
    # rglob, not glob("*/kaggle/**/*"): the biggest bundle by far is the submission staging
    # directory runs/others/ensembling/kaggle, which sits two levels down and a one-level
    # pattern silently misses it (1.42 GB reported instead of 5.68 GB).
    kag = [p for p in RUNS.rglob("*") if p.is_file() and "kaggle" in p.relative_to(RUNS).parts]
    print("\nkaggle bundles: %d files, %.2f GB"
          % (len(kag), sum(p.stat().st_size for p in kag) / 1e9))
    for d in sorted({p.relative_to(RUNS).parts[0] + "/" + "/".join(
            p.relative_to(RUNS).parts[1:p.relative_to(RUNS).parts.index("kaggle") + 1])
            for p in kag}):
        n = [p for p in kag if d in p.relative_to(RUNS).as_posix()]
        print("    %-52s %4d files  %5.2f GB"
              % (d, len(n), sum(x.stat().st_size for x in n) / 1e9))
    doomed += kag

    # --- 3. resume-only checkpoints of runs that are not mid-flight ---------------
    print("\nresume-only checkpoints (last.pt / step*.pt) of finished or idle runs:")
    for p in sorted(RUNS.glob("*/results/checkpoints/*.pt")):
        run = p.relative_to(RUNS).parts[0]
        if run in IN_FLIGHT or p.name == "best.pt":
            continue
        if p.name == "last.pt" or p.name.startswith(("step", "top_")):
            doomed.append(p)
            print("  %-42s %-16s %6.0f MB" % (run, p.name, p.stat().st_size / 1e6))

    total = sum(p.stat().st_size for p in doomed) / 1e9
    print("\n" + "=" * 74)
    print("  %d files, %.2f GB" % (len(doomed), total))
    kept_best = list(RUNS.glob("*/results/checkpoints/best.pt"))
    print("  untouched: %d best.pt, %d in-flight resume files, %d predictions"
          % (len(kept_best),
             len([p for p in RUNS.glob("*/results/checkpoints/*.pt")
                  if p.relative_to(RUNS).parts[0] in IN_FLIGHT]),
             len(list(RUNS.glob("*/results/predictions/*.npz")))))
    print("=" * 74)

    if not a.apply:
        print("\nDRY RUN. Re-run with --apply to delete.")
        return
    freed = 0
    for p in doomed:
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError as e:
            print("  could not delete %s: %s" % (p, e))
    # drop the directories that are now empty
    for d in sorted(RUNS.glob("*/kaggle/**"), key=lambda x: -len(x.parts)):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    for d in sorted(RUNS.glob("*/kaggle"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    print("\nfreed %.2f GB" % (freed / 1e9))


if __name__ == "__main__":
    main()
