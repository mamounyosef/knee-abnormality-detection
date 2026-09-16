r"""make_fold_runs.py - build the cross-validation fold runs from a parent run.

WHY A SCRIPT. Five runs that must differ in EXACTLY the split (and, for the last two, the
seed) and be identical in every other respect. Hand-editing five config.yaml files is how one
fold quietly ends up with a different learning rate or slice count, and a fold ensemble built
from inconsistent members is worse than useless - it looks like a result.

WHY THE CHECKS AT THE END. Editing the configs is not enough and proving they differ is not a
test. Every .py in a run folder hardcodes `EXP_NAME = "<run>"` and builds EXP_DIR from it, so
copied files make a fold load the PARENT's config: parent's split, parent's name, parent's
results path, silently. Three folds each burned ~1h09m of A100 doing exactly that.

    python runs/others/make_fold_runs.py --patience 6
    python runs/others/make_fold_runs.py --parent <run> --folds 5
"""

import argparse
import difflib
import io
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs"

# what a fold needs to train and be evaluated. results/ and kaggle/ are NOT copied: they
# belong to the parent, and a stale checkpoint in a fold's results/ is exactly the failure
# that `resume: last` would silently pick up.
COPY = ["config.yaml", "train.py", "evaluate.py", "plot_results.py", "submit.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="convnext_base_25d_18slices")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--patience", type=int, default=7,
                    help="validations without improvement before stopping")
    ap.add_argument("--seed-base", type=int, default=42,
                    help="folds 0-2 all use seed-base; folds 3 and 4 get seed-base+1000 "
                         "and +2000, far enough apart to be obviously intentional in a log")
    a = ap.parse_args()

    src = RUNS / a.parent
    if not src.is_dir():
        raise SystemExit("no such run: %s" % src)
    parent_cfg = io.open(src / "config.yaml", encoding="utf-8").read()

    made = []
    for k in range(a.folds):
        tr = ROOT / "data" / ("13_fold%d_train.csv" % k)
        va = ROOT / "data" / ("13_fold%d_val.csv" % k)
        for f in (tr, va):
            if not f.exists():
                raise SystemExit("missing fold csv: %s" % f)

        name = "%s_f%d" % (a.parent, k)
        dst = RUNS / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in COPY:
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)

        # EXP_NAME has to stay a LITERAL, rewritten per fold.
        #
        # Deriving it from the file's own folder (`Path(__file__).parent.name`) looks more
        # robust and BREAKS ON MODAL: the entrypoint is copied to /root/train.py, so the
        # folder is "root" and EXP_DIR resolves to /root/runs/root/config.yaml, which does
        # not exist. It passes every local test and dies in the container.
        for f in COPY:
            p = dst / f
            if not p.exists() or p.suffix != ".py":
                continue
            txt = io.open(p, encoding="utf-8").read()
            if 'EXP_NAME = "' not in txt:
                continue
            io.open(p, "w", encoding="utf-8", newline="\n").write(
                re.sub(r'EXP_NAME = "[^"]*"', 'EXP_NAME = "%s"' % name, txt, count=1))

        cfg = parent_cfg
        # experiment.name is the FIRST `name:` in the file, under `experiment:`
        cfg = re.sub(r'(\n\s*name:\s*)"%s"' % re.escape(a.parent),
                     lambda m: '%s"%s"' % (m.group(1), name), cfg, count=1)
        cfg = re.sub(r'(\n\s*train_csv:\s*)"[^"]*"',
                     lambda m: '%s"13_fold%d_train.csv"' % (m.group(1), k), cfg, count=1)
        cfg = re.sub(r'(\n\s*val_csv:\s*)\s*"[^"]*"',
                     lambda m: '%s"13_fold%d_val.csv"' % (m.group(1), k), cfg, count=1)
        cfg = re.sub(r'(\n\s*patience:\s*)\d+',
                     lambda m: "%s%d" % (m.group(1), a.patience), cfg, count=1)

        # SEEDS: folds 0-2 SHARE the base seed, folds 3 and 4 each get their own.
        #
        # Folds 0-2 are the ones the go/no-go call is made on, and holding the seed fixed
        # there means their score spread measures SPLIT variance alone - a weak fold cannot
        # be blamed on an unlucky seed. Folds 3-4 are only ever ensemble members, so they
        # take trajectory diversity (different init and augmentation/shuffle stream) on top
        # of their different splits: free, and it cannot lower a fold's expected score.
        #
        # Seed diversity is weak alone here (two seeds of one config correlate 0.952), which
        # is why it is spent on the folds that are not also a measurement.
        #
        # `reproducibility` lives in shared_config.yaml, not the parent's config, so the
        # block is appended rather than patched.
        seed = a.seed_base if k < 3 else a.seed_base + 1000 * (k - 2)
        if re.search(r"^reproducibility:", cfg, flags=re.M):
            cfg = re.sub(r"(^reproducibility:\n(?:.*\n)*?\s*seed:\s*)\d+",
                         lambda m: "%s%d" % (m.group(1), seed), cfg, count=1, flags=re.M)
        else:
            note = ("shared with folds 0-2: their spread measures the SPLIT alone"
                    if k < 3 else "this fold's own seed: adds trajectory diversity")
            cfg = cfg.rstrip("\n") + (
                "\n\nreproducibility:\n  seed: %d                          # %s\n"
                % (seed, note))
        io.open(dst / "config.yaml", "w", encoding="utf-8", newline="\n").write(cfg)
        made.append((name, cfg))

    # --- 1. only the intended lines differ from the parent ---------------------------
    print("\nverifying each fold's config against %s" % a.parent)
    plines = parent_cfg.splitlines()
    allowed = ("name:", "train_csv:", "val_csv:", "patience:", "seed:",
               "reproducibility:", "#")
    bad = 0
    for name, cfg in made:
        diff = [d for d in difflib.unified_diff(plines, cfg.splitlines(), lineterm="", n=0)
                if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))]
        changed = [d for d in diff if d.startswith("+")]
        # a bare "+" is the blank line separating the appended reproducibility block
        unexpected = [d for d in changed
                      if d.strip() != "+" and not any(t in d for t in allowed)]
        print("  %-36s %d changed line(s)%s"
              % (name, len(changed), "" if not unexpected else "  UNEXPECTED: %s" % unexpected))
        if unexpected:
            bad += 1
    if bad:
        raise SystemExit("\n%d fold(s) differ from the parent in unintended ways" % bad)
    print("  all differ only in name, csvs, patience and seed")

    # --- 2. resolve EXP_DIR THE WAY THE CONTAINER DOES -------------------------------
    # This is the check that was missing. A local-only check passes for both the bug that
    # wasted three A100 hours (EXP_NAME still the parent's) and the "fix" that replaced it
    # (EXP_NAME derived from a folder that is "root" on Modal). Neither survives asking
    # what the CONTAINER would open: /root/runs/<EXP_NAME>/config.yaml.
    print("\nwhat each fold declares, resolved the way the container does")
    import sys
    sys.path.insert(0, str(RUNS))
    import shared_code as sc

    fail = 0
    for idx, (name, _) in enumerate(made):
        d = RUNS / name
        declared = set()
        for f in sorted(d.glob("*.py")):
            txt = f.read_text(encoding="utf-8")
            m = re.search(r'EXP_NAME = "([^"]*)"', txt)
            if m:
                declared.add(m.group(1))
            elif "EXP_NAME" in txt and "EXP_NAME = " in txt:
                declared.add("NOT-A-LITERAL")

        literal_ok = declared == {name}
        one = list(declared)[0] if len(declared) == 1 else "?"
        # local runs/ stands in for /root/runs
        cfg_there = (RUNS / one / "config.yaml").exists() if literal_ok else False

        c = sc.load_config(d, verbose=False)
        trc = (c.get("paths") or {}).get("train_csv")
        sd = (c.get("reproducibility") or {}).get("seed")
        ok = (literal_ok and cfg_there
              and c["experiment"]["name"] == name
              and trc == "13_fold%d_train.csv" % idx)
        print("  %-36s EXP_NAME=%-34s csv=%-22s seed=%-6s %s"
              % (name, "/".join(sorted(declared)) or "MISSING", trc, sd,
                 "ok" if ok else "FAIL"))
        if not literal_ok:
            print("       every .py must declare a LITERAL EXP_NAME = %r; found %s"
                  % (name, sorted(declared)))
        elif not cfg_there:
            print("       /root/runs/%s/config.yaml would not exist in the container" % one)
        fail += not ok
    if fail:
        raise SystemExit("%d fold(s) would load the wrong config, or none at all" % fail)
    print("\nevery fold declares its own name and resolves in the container layout")


if __name__ == "__main__":
    main()
