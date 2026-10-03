"""audit_stop_rules.py - which ONLINE stopping rule on audit-58 alone is safe for full-data runs?

Setting: a full-data relabel trains on everything except the 58 audit studies and validates only on
them. It must decide WHILE TRAINING which checkpoint to keep and when to stop. Tested on every
finished split-data run with a 704-study val curve and per-step audit predictions:

  regret = the run's best val AUROC - val AUROC at the checkpoint the rule kept
           (val, 12x larger, is the reference for where the run really peaked)

Rules (grid): grade = mean of the audit score over a window of checks (centred = one check of lag,
or trailing = no lag); best.pt moves when the grade beats the best so far; stop after PATIENCE EPOCHS
(not checks: runs validate at different rates) with no new best, never before MIN_EPOCH.
Truncation: a source run ends where its own val early stopping ended it; a rule that would train
longer just ends there too (that only flatters "stop late" rules slightly).

Noise: the 58 studies are resampled B times (paired across steps), as a real run would face.
Overfitting the grid: the rule is CHOSEN on a random half of the runs and SCORED on the other half,
repeated 200 times. Cohort: the ensemble's own architectures are reported separately.

    python runs/others/audit_stop_rules.py            # builds the cache on first use (~10 min)
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent.parent
D = RUNS.parent / "data"
CACHE = Path(__file__).with_name("audit_stop_rules_cache.npz")
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
B = 200
ENSEMBLE_FAMILIES = ("convnext_base_relabel", "mricore_vitb_cross_attention", "mricore_vitb_mae_cross_attention",
                     "dinov3_convnext_base", "convnext_base_25d_all_knees_same_orientation", "swinv2_base_in22k",
                     "convnext_base_25d_poolattn", "convnext_base_25d_perfinding_slots_and_pooling",
                     "medimageinsight_davit_huge_336", "medimageinsight_davit_huge_perslice_480_warm")


def auc(y, s):
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    d = p[:, None] - n[None, :]
    return (d > 0).mean() + 0.5 * (d == 0).mean()


def build_cache():
    rng = np.random.default_rng(0)
    aud = pd.read_csv(D / "05_audit58.csv").set_index("StudyInstanceUID")
    U = [str(s) for s in aud.index]
    Y = (aud[["official_" + t for t in T]].values >= 0.5).astype(int)
    boot = rng.integers(0, len(U), (B, len(U)))
    store = {}
    for vl in sorted(RUNS.glob("*/results/val_log.csv")):
        run = vl.parent.parent.name
        cfg = (vl.parent.parent / "config.yaml").read_text(encoding="utf-8", errors="replace")
        if "train_full" in cfg:
            continue                                 # full-data runs: their val is in-sample
        v = pd.read_csv(vl)
        if "val_macro_auroc" not in v or "step" not in v or "epoch" not in v:
            continue
        v = v.dropna(subset=["val_macro_auroc"]).drop_duplicates("step", keep="last").sort_values("step")
        steps, A = [], []
        for st in v.step.astype(int):
            fs = sorted((vl.parent / "predictions").glob(f"*_audit_s{st}.npz"), key=lambda f: f.stat().st_mtime)
            if not fs:
                continue
            try:
                z = np.load(fs[-1], allow_pickle=True)
                df = pd.DataFrame(z["logits"], index=[str(s) for s in z["study_ids"]])
            except Exception:
                continue
            if not set(U) <= set(df.index) or df.shape[1] != 12:
                continue
            steps.append(st)
            A.append(df.loc[U].values)
        if len(steps) < 8:
            continue
        vv = v.set_index("step").loc[steps]
        a = np.array([np.nanmean([auc(Y[:, j], P[:, j]) for j in range(12)]) for P in A])
        ab = np.array([[np.nanmean([auc(Y[b][:, j], P[b][:, j]) for j in range(12)]) for P in A] for b in boot])
        store[run] = dict(step=np.array(steps), epoch=vv.epoch.values.astype(float),
                          val=vv.val_macro_auroc.values, audit=a, boot=ab)
        print(f"  cached {run} ({len(steps)} checks)", flush=True)
    flat = {}
    for r, d in store.items():
        for k, x in d.items():
            flat[f"{r}|{k}"] = x
    np.savez_compressed(CACHE, **flat)


def load_cache():
    z = np.load(CACHE)
    out = {}
    for key in z.files:
        r, k = key.split("|")
        out.setdefault(r, {})[k] = z[key]
    return out


def kept(a, ep, window, centred, pat_ep, min_ep, why=None):
    """index of the checkpoint the rule keeps (why: list that receives "stopped" or "ran out")"""
    n = len(a)
    best, bj, last_best_ep = -np.inf, 0, ep[0]
    for i in range(n):
        if centred:
            j = i - 1                                 # graded once check i exists
            if j < 0:
                continue
            h = window // 2
            g = np.nanmean(a[max(0, j - h):min(n, j + h + 1)])
        else:
            j = i
            g = np.nanmean(a[max(0, j - window + 1):j + 1])
        if g > best:
            best, bj, last_best_ep = g, j, ep[j]
        if ep[i] >= min_ep and ep[i] - last_best_ep >= pat_ep:
            if why is not None:
                why.append("stopped")
            return bj
    if centred:                                       # run ended: grade the final check on what exists
        h = window // 2
        g = np.nanmean(a[max(0, n - 1 - h):n])
        if g > best:
            bj = n - 1
    if why is not None:
        why.append("ran out")
    return bj


def main():
    if not CACHE.exists():
        build_cache()
    R = load_cache()
    runs = sorted(R)
    fam = [r for r in runs if any(r == f or r.startswith(f + "_") and "fold" not in r for f in ENSEMBLE_FAMILIES)]
    print(f"{len(runs)} runs ({len(fam)} in the ensemble's families)\n")

    grid = list(itertools.product((1, 3, 5), (True, False), (0.75, 1.0, 1.5, 2.0, 3.0), (4, 6, 8)))
    # regret[c, run, b]: b=0 is the real audit, 1..B the resampled ones
    reg = np.zeros((len(grid), len(runs), B + 1))
    for ri, r in enumerate(runs):
        d = R[r]
        best = d["val"].max()
        curves = [d["audit"]] + list(d["boot"])
        for ci, (w, c, p, m) in enumerate(grid):
            for b, a in enumerate(curves):
                reg[ci, ri, b] = best - d["val"][kept(a, d["epoch"], w, c, p, m)]

    def summ(x):
        return x.mean(), np.median(x), np.percentile(x, 90), x.max(), (x <= 0.002).mean()

    print("ALL CONFIGURATIONS, sorted by mean regret over runs x resamples (real audit + 200 resampled):")
    print(f"{'window':>6s} {'grading':>9s} {'patience ep':>11s} {'min ep':>6s} | {'mean':>7s} {'median':>7s} {'p90':>7s} {'max':>7s} {'<=.002':>7s} | real-audit mean  p90")
    order = np.argsort(reg.mean(axis=(1, 2)))
    for ci in order[:15]:
        w, c, p, m = grid[ci]
        s = summ(reg[ci].ravel())
        sr = summ(reg[ci, :, 0])
        print(f"{w:6d} {'centred' if c else 'trailing':>9s} {p:11.2f} {m:6d} | {s[0]:7.4f} {s[1]:7.4f} {s[2]:7.4f} {s[3]:7.4f} {s[4]:7.0%} | {sr[0]:.4f} {sr[2]:.4f}")
    print("  ...")
    for ci in order[-3:]:
        w, c, p, m = grid[ci]
        s = summ(reg[ci].ravel())
        print(f"{w:6d} {'centred' if c else 'trailing':>9s} {p:11.2f} {m:6d} | {s[0]:7.4f} {s[1]:7.4f} {s[2]:7.4f} {s[3]:7.4f} {s[4]:7.0%}")

    # honest estimate: pick the config on half the runs, score it on the other half
    rng = np.random.default_rng(1)
    held, picks = [], []
    for _ in range(200):
        perm = rng.permutation(len(runs))
        a_, b_ = perm[: len(runs) // 2], perm[len(runs) // 2:]
        ci = int(np.argmin(reg[:, a_, :].mean(axis=(1, 2))))
        picks.append(ci)
        held.append(reg[ci][b_].ravel())
    held = np.concatenate(held)
    s = summ(held)
    print(f"\nCHOSEN ON HALF THE RUNS, SCORED ON THE OTHER HALF (200 splits): mean {s[0]:.4f}  median {s[1]:.4f}  "
          f"p90 {s[2]:.4f}  max {s[3]:.4f}  within 0.002 {s[4]:.0%}")
    top = pd.Series(picks).value_counts().head(3)
    for ci, n in top.items():
        print(f"   picked {n:3d}/200 times: window {grid[ci][0]}, {'centred' if grid[ci][1] else 'trailing'}, "
              f"patience {grid[ci][2]} epochs, min epoch {grid[ci][3]}")

    best_ci = int(order[0])
    w, c, p, m = grid[best_ci]
    print(f"\nBEST RULE: window {w}, {'centred' if c else 'trailing'}, patience {p} epochs, min epoch {m}")
    fi = [runs.index(r) for r in fam]
    for lab, idx in (("all runs", list(range(len(runs)))), ("ensemble families", fi)):
        s = summ(reg[best_ci, idx].ravel())
        print(f"  {lab:18s} mean {s[0]:.4f}  median {s[1]:.4f}  p90 {s[2]:.4f}  max {s[3]:.4f}  within 0.002 {s[4]:.0%}")
    print("\n  per ensemble-family run (real audit regret | resampled median, p90 | val peak epoch -> kept epoch):")
    for r in fam:
        ri = runs.index(r)
        d = R[r]
        k = kept(d["audit"], d["epoch"], w, c, p, m)
        x = reg[best_ci, ri, 1:]
        print(f"   {r:58s} {reg[best_ci, ri, 0]:.4f} | {np.median(x):.4f}, {np.percentile(x, 90):.4f} | "
              f"ep {d['epoch'][d['val'].argmax()]:.0f} -> {d['epoch'][k]:.0f}")
    worst = np.argsort(-reg[best_ci, :, 0])[:5]
    print("\n  worst runs under the best rule (real audit):")
    for ri in worst:
        d = R[runs[ri]]
        k = kept(d["audit"], d["epoch"], w, c, p, m)
        print(f"   {runs[ri]:58s} regret {reg[best_ci, ri, 0]:.4f}  val peak ep {d['epoch'][d['val'].argmax()]:.0f}, "
              f"kept ep {d['epoch'][k]:.0f}, run ended ep {d['epoch'][-1]:.0f}")

    # reference: what a fixed epoch costs when the labels shift (the alternative to an audit rule)
    print("\nREFERENCE (no audit at all): stop at a FIXED epoch E for every run")
    for E in (6, 8, 9, 10, 12):
        x = []
        for r in runs:
            d = R[r]
            i = int(np.argmin(np.abs(d["epoch"] - E)))
            x.append(d["val"].max() - d["val"][i])
        s = summ(np.array(x))
        print(f"  E={E:2d}: mean {s[0]:.4f}  median {s[1]:.4f}  p90 {s[2]:.4f}  max {s[3]:.4f}  within 0.002 {s[4]:.0%}")


def complete_runs(R):
    """runs that trained their WHOLE configured schedule: no val early stopping cut their curve, so a
    rule that 'runs out of curve' there gets no help from val"""
    sys.path.insert(0, str(RUNS))
    import shared_code as sc
    out = []
    for r, d in R.items():
        try:
            e = int(sc.load_config(RUNS / r, verbose=False)["training"]["epochs"])
        except Exception:
            continue
        if d["epoch"][-1] >= e:
            out.append(r)
    return sorted(out)


def unbiased(R, grid):
    """the grid again, on complete runs only, with how often each rule stopped by its own trigger"""
    comp = complete_runs(R)
    print(f"\nUNBIASED CHECK: {len(comp)} runs that trained their full schedule (curve not cut by val early stopping)")
    print(f"{'window':>6s} {'grading':>9s} {'patience ep':>11s} {'min ep':>6s} | {'mean':>7s} {'median':>7s} {'p90':>7s} {'max':>7s} {'<=.002':>7s} | stopped itself")
    res = []
    for (w, c, p, m) in grid:
        x, why = [], []
        for r in comp:
            d = R[r]
            best = d["val"].max()
            for a in [d["audit"]] + list(d["boot"]):
                x.append(best - d["val"][kept(a, d["epoch"], w, c, p, m, why)])
        x = np.array(x)
        res.append(((w, c, p, m), x.mean(), np.median(x), np.percentile(x, 90), x.max(), (x <= 0.002).mean(),
                    np.mean([y == "stopped" for y in why])))
    res.sort(key=lambda t: t[1])
    for g, mn, md, p9, mx, w2, st in res[:12]:
        print(f"{g[0]:6d} {'centred' if g[1] else 'trailing':>9s} {g[2]:11.2f} {g[3]:6d} | {mn:7.4f} {md:7.4f} {p9:7.4f} {mx:7.4f} {w2:7.0%} | {st:.0%}")
    print("  reference, fixed epoch E on the same complete runs:")
    for E in (6, 8, 9, 10, 12):
        x = np.array([R[r]["val"].max() - R[r]["val"][int(np.argmin(np.abs(R[r]["epoch"] - E)))] for r in comp])
        print(f"   E={E:2d}: mean {x.mean():.4f}  median {np.median(x):.4f}  p90 {np.percentile(x, 90):.4f}  max {x.max():.4f}  within 0.002 {(x <= 0.002).mean():.0%}")
    print("  reference, train to the end of the schedule (last check):")
    x = np.array([R[r]["val"].max() - R[r]["val"][-1] for r in comp])
    print(f"   end: mean {x.mean():.4f}  median {np.median(x):.4f}  p90 {np.percentile(x, 90):.4f}  max {x.max():.4f}  within 0.002 {(x <= 0.002).mean():.0%}")
    print("  complete runs:", ", ".join(comp))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--unbiased":
        unbiased(load_cache(), list(itertools.product((1, 3, 5), (True, False), (0.75, 1.0, 1.5, 2.0, 3.0), (4, 6, 8))))
        sys.exit(0)
    sys.exit(main())
